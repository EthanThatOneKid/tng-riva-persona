# tng-riva-persona

Dedicated persona repo for Riva from *Star Trek: The Next Generation*.

This repo stays focused on Riva-specific prompting, examples, and evaluation. The shared TNG corpus lives in `../tng-computer-persona`, and the extractor here pulls episode `100132.txt` from `../tng-computer-persona/data/dialogue.jsonl`.

## Extractor

```bash
python -m scripts.extract_riva_persona
```

Outputs:

- `data/riva_examples.jsonl`
- `data/riva_train.jsonl`
- `data/riva_eval.jsonl`
- `data/riva_counterexamples.jsonl`
- `data/riva_extract_report.md`

## Scope

- Riva voice prompt
- Riva-specific examples and counterexamples
- Notes about mediated communication and sign-language interpretation
- Links back to the shared TNG corpus and episode source material
