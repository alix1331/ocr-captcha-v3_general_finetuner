# TrOCR Captcha Fine-Tuning

A general-purpose, dataset-agnostic fine-tuning script for local
TrOCR-style captcha OCR models. Write it once, then fine-tune it on any
number of different captcha datasets by changing a few command-line
flags — not by writing a new script each time.

Built around a **prepare → correct → train → evaluate** loop: the model
predicts a batch of your images, you fix its mistakes in a CSV, and the
script fine-tunes on your corrections and reports before/after accuracy.

## Features

- **One script, many datasets.** A pluggable `--task_type` handles
  dataset-specific label logic (`math_expression`, `plain_text`); the
  model loading, training loop, and CSV workflow are fully shared.
- **Local model only.** Nothing is downloaded from Hugging Face at
  runtime — point `--model_dir` at a local checkpoint folder (e.g. a
  local copy of [`anuashok/ocr-captcha-v3`](https://huggingface.co/anuashok/ocr-captcha-v3))
  and it stays fully offline.
- **Human-in-the-loop correction, not blind trust in filenames.** Labels
  always come from a `corrected_text` column you fill in — image
  filenames are never treated as answers.
- **Multi-script math support.** Persian, Arabic, and ASCII digits, and
  configurable operators (`+ - * /`).
- **Namespaced runs.** Every `--run_name` gets its own output folder, so
  fine-tuning multiple datasets never overwrites another dataset's CSVs
  or model.
- **No count required.** Point it at a folder and it uses every image
  by default, or a random deterministic subset if you ask for one.

## Requirements

- Python 3.9+
- See [`requirements.txt`](requirements.txt): `torch`, `transformers`,
  `pandas`, `numpy`, `Pillow`.

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-matched `torch` build first (see
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)).

## Quick start

```bash
# 1) Generate predictions for a batch of images to manually correct
python finetune_general.py \
    --task_type math_expression \
    --model_dir  ./ocr-captcha-v3 \
    --train_dir  ./datasets/dissel/train \
    --test_dir   ./datasets/dissel/test \
    --run_name   dissel \
    --prepare

# 2) Open outputs/dissel/initial_predictions.csv and fill in
#    the corrected_text column with the true label for each image.

# 3) Fine-tune on your corrections and evaluate
python finetune_general.py \
    --task_type math_expression \
    --model_dir  ./ocr-captcha-v3 \
    --train_dir  ./datasets/dissel/train \
    --test_dir   ./datasets/dissel/test \
    --run_name   dissel \
    --train
```

Switching to a different, non-math dataset is the same two commands
with a different `--task_type` and different folders — the script
itself doesn't change:

```bash
python finetune_general.py --task_type plain_text \
    --model_dir ./ocr-captcha-v3 \
    --train_dir ./datasets/alnum/train --test_dir ./datasets/alnum/test \
    --run_name alnum --prepare
```

## Project structure

```
.
├── finetune_general.py   # the fine-tuning script
├── requirements.txt
├── MANUAL.md              # full usage manual (all flags, CSV schema, etc.)
└── outputs/                # created at runtime, one subfolder per --run_name
    └── <run_name>/
        ├── initial_predictions.csv
        ├── after_finetune_predictions.csv
        ├── test_predictions.csv
        └── finetuned_model/
```

## Task types

| `--task_type`      | Use for...                                              |
|---------------------|-----------------------------------------------------------|
| `math_expression`   | Captchas showing a short sum, e.g. `35-7`, `۱۲+۸`.         |
| `plain_text`        | Captchas showing plain text/alphanumeric strings.          |

Adding a third task type is a matter of subclassing `TaskAdapter` and
registering it — see the "Adding a new task type" section of
[`MANUAL.md`](MANUAL.md).

## Documentation

See [`MANUAL.md`](MANUAL.md) for the full manual: every command-line
flag, the CSV column schema, output file layout, and tips for
troubleshooting corrections.

## How it works, briefly

The base model is a TrOCR-style vision-encoder-decoder
(`VisionEncoderDecoderModel` + `TrOCRProcessor`). Fine-tuning is a
plain PyTorch loop (AdamW, linear warmup/decay, gradient clipping) over
whatever images you've corrected — no `Trainer` or `accelerate`
dependency. Generation uses an explicit `GenerationConfig` passed to
`generate()` rather than mutating `model.config`, to stay compatible
with current `transformers` versions.

## License

This project is licensed under the MIT License. See LICENSE for the complete license text.

The project uses Hugging Face transformers, the TrOCR architecture, and potentially pretrained model files. Their respective licenses and terms remain applicable to those dependencies and model files. Check the original model repository for its specific license and usage conditions.

## Acknowledgements

Built on top of Hugging Face [`transformers`](https://github.com/huggingface/transformers)
and the TrOCR architecture. Tested against
[`anuashok/ocr-captcha-v3`](https://huggingface.co/anuashok/ocr-captcha-v3).
