from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
SHARED_TNG_DIR = REPO_DIR.parent / "tng-computer-persona"
SHARED_SOURCE_DIALOGUE = (
    SHARED_TNG_DIR
    / "data"
    / "dialogue.jsonl"
)
OUTPUT_DIR = REPO_DIR / "data"
OUTPUT_EXAMPLES = OUTPUT_DIR / "riva_examples.jsonl"
OUTPUT_TRAIN = OUTPUT_DIR / "riva_train.jsonl"
OUTPUT_EVAL = OUTPUT_DIR / "riva_eval.jsonl"
OUTPUT_COUNTEREXAMPLES = OUTPUT_DIR / "riva_counterexamples.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "riva_extract_report.md"

PERSONA_SYSTEM = (
    "Respond as Riva from Star Trek: The Next Generation. "
    "Keep the tone calm, deliberate, and emotionally restrained. "
    "Speak in concise phrasing that works through interpretation or sign language. "
    "Make the meaning legible without becoming theatrical."
)

RIVA_SOURCE_SPEAKERS = {"RIVA", "SCHOLAR", "WOMAN", "ADONIS", "CHORUS"}

SCENE_RE = re.compile(r"^(?:INT\.|EXT\.|FADE IN|FADE OUT|ACT\b|TAG\b|CONTINUED:|TEASER\b|[0-9]+[A-Z]?\s+CONTINUED:)")
SPEAKER_RE = re.compile(r"^[A-Z][A-Z0-9 '\-./()]+$")
RIVA_STAGE_RE = re.compile(r"\b(riva|speaking for riva|translating for riva|continuing to speak for riva|translating)\b", re.IGNORECASE)


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def default_source() -> Path:
    if not SHARED_SOURCE_DIALOGUE.exists():
        raise FileNotFoundError(f"Missing shared source corpus: {SHARED_SOURCE_DIALOGUE}")
    return SHARED_SOURCE_DIALOGUE


def extract_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing source transcript: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def load_blocks(path: Path, episode: str | None = None) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        blocks: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if episode and row.get("episode") != episode:
                    continue
                blocks.append(
                    {
                        "speaker": row["speaker"],
                        "stages": [],
                        "text": row["text"],
                        "line_start": row["line_num"],
                        "line_end": row["line_num"],
                    }
                )
        return blocks

    return parse_blocks(extract_text(path))


def is_speaker_heading(raw_line: str) -> bool:
    stripped = raw_line.strip()
    if not stripped or SCENE_RE.match(stripped):
        return False
    if ":" in stripped:
        return False
    if stripped.startswith(("(", "[")):
        return False
    if len(stripped) > 40:
        return False
    if "," in stripped:
        return False
    return bool(SPEAKER_RE.match(stripped))


def parse_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text_value = " ".join(line.strip() for line in current["text_lines"] if line.strip())
        text_value = re.sub(r"\s+", " ", text_value).strip()
        if text_value:
            blocks.append(
                {
                    "speaker": current["speaker"],
                    "stages": list(current["stages"]),
                    "text": text_value,
                    "line_start": current["line_start"],
                    "line_end": current["line_end"],
                }
            )
        current = None

    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if SCENE_RE.match(stripped):
            flush()
            continue
        if is_speaker_heading(raw_line):
            flush()
            current = {
                "speaker": stripped,
                "stages": [],
                "text_lines": [],
                "line_start": line_num,
                "line_end": line_num,
            }
            continue
        if current is None:
            continue
        current["line_end"] = line_num
        if stripped.startswith(("(", "[")) and stripped.endswith((")", "]")):
            current["stages"].append(stripped.strip("()[]"))
            continue
        current["text_lines"].append(stripped)

    flush()
    return blocks


def is_riva_block(block: dict) -> tuple[bool, str]:
    speaker = block["speaker"].upper()
    stage_blob = " ".join(block["stages"])
    text_blob = block["text"]
    if speaker in RIVA_SOURCE_SPEAKERS:
        return True, speaker
    if speaker == "DATA" and RIVA_STAGE_RE.search(stage_blob):
        return True, "DATA"
    if speaker == "DATA" and RIVA_STAGE_RE.search(text_blob):
        return True, "DATA"
    return False, ""


def build_examples(
    blocks: list[dict],
    source_file: str,
    context_lines: int,
    eval_every: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    examples: list[dict] = []
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    counterexamples: list[dict] = []
    riva_count = 0

    for index, block in enumerate(blocks):
        is_riva, source_speaker = is_riva_block(block)
        if not is_riva:
            continue

        riva_count += 1
        context: list[dict] = []
        cursor = index - 1
        while cursor >= 0 and len(context) < context_lines:
            candidate = blocks[cursor]
            if is_riva_block(candidate)[0]:
                cursor -= 1
                continue
            context.append(candidate)
            cursor -= 1

        context.reverse()
        context_text = "\n".join(f"{item['speaker']}: {item['text']}" for item in context)
        record = {
            "id": stable_id(block["speaker"], block["line_start"], block["text"]),
            "messages": [
                {"role": "system", "content": PERSONA_SYSTEM},
                {"role": "user", "content": context_text or "Please continue."},
                {"role": "assistant", "content": block["text"]},
            ],
            "metadata": {
                "speaker": "RIVA",
                "source_speaker": source_speaker,
                        "source_file": source_file,
                "line_start": block["line_start"],
                "line_end": block["line_end"],
                "stages": block["stages"],
            },
        }
        examples.append(record)
        target = eval_rows if eval_every > 0 and riva_count % eval_every == 0 else train_rows
        target.append(record)

        for item in context:
            counterexamples.append(
                {
                    "id": stable_id("counter", block["line_start"], item["line_start"], item["speaker"], item["text"]),
                    "speaker": item["speaker"],
                    "source_file": source_file,
                    "line_start": item["line_start"],
                    "line_end": item["line_end"],
                    "text": item["text"],
                    "context": context_text,
                    "reason": "near-miss line surrounding a Riva utterance",
                }
            )

    summary = {
        "source_blocks": len(blocks),
        "riva_blocks": riva_count,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "counterexamples": len(counterexamples),
    }
    return examples, train_rows, eval_rows, counterexamples, summary


def build_report(examples: list[dict], counterexamples: list[dict], blocks: list[dict], summary: dict) -> str:
    from collections import Counter

    source_mix = Counter(row["metadata"]["source_speaker"] for row in examples)
    samples = [row["messages"][2]["content"] for row in examples[:5]]
    return "\n".join(
        [
            "# Riva Persona Extract Report",
            "",
            f"- Source transcript: `{summary['source_file']}`",
            f"- Episode: `{summary['episode']}`",
            f"- Parsed dialogue blocks: {summary['source_blocks']}",
            f"- Extracted Riva examples: {summary['train_rows'] + summary['eval_rows']}",
            f"- Counterexamples: {summary['counterexamples']}",
            "",
            "## Source mix",
            "",
            "| Source speaker | Count |",
            "|---|---:|",
            *[f"| {speaker} | {count} |" for speaker, count in sorted(source_mix.items())],
            "",
            "## Sample assistant lines",
            "",
            *[f"- {line}" for line in samples],
            "",
            "## Notes",
            "",
            "- The extractor captures both direct Riva dialogue and Data lines explicitly marked as speaking for Riva.",
            "- The resulting examples are intentionally narrow and mediated, which matches the persona scope.",
        ]
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a Riva persona corpus from Loud as a Whisper.")
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source(),
        help="Path to the Loud as a Whisper transcript. Defaults to the shared TNG corpus transcript when available.",
    )
    parser.add_argument("--episode", type=str, default="100132.txt", help="Episode identifier to filter when the source is a shared dialogue JSONL corpus.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for the generated Riva persona artifacts.")
    parser.add_argument("--context-lines", type=int, default=2, help="Number of preceding non-Riva blocks to include as prompt context.")
    parser.add_argument("--eval-every", type=int, default=6, help="Put every Nth Riva example into the eval split.")
    args = parser.parse_args()

    blocks = load_blocks(args.source, episode=args.episode)
    source_file = (
        str(args.source.relative_to(REPO_DIR.parent))
        if args.source.is_relative_to(REPO_DIR.parent)
        else str(args.source)
    )
    examples, train_rows, eval_rows, counterexamples, summary = build_examples(
        blocks,
        source_file=source_file,
        context_lines=args.context_lines,
        eval_every=args.eval_every,
    )
    summary["source_file"] = source_file
    summary["episode"] = args.episode

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "riva_examples.jsonl", examples)
    write_jsonl(output_dir / "riva_train.jsonl", train_rows)
    write_jsonl(output_dir / "riva_eval.jsonl", eval_rows)
    write_jsonl(output_dir / "riva_counterexamples.jsonl", counterexamples)
    (output_dir / "riva_extract_report.md").write_text(
        build_report(examples, counterexamples, blocks, summary) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'riva_examples.jsonl'}")
    print(f"Wrote {output_dir / 'riva_train.jsonl'}")
    print(f"Wrote {output_dir / 'riva_eval.jsonl'}")
    print(f"Wrote {output_dir / 'riva_counterexamples.jsonl'}")
    print(f"Wrote {output_dir / 'riva_extract_report.md'}")


if __name__ == "__main__":
    main()
