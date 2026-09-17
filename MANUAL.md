# finetune_general.py — Manual

A dataset-agnostic fine-tuning script for local TrOCR-style captcha OCR
models (e.g. `anuashok/ocr-captcha-v3`, or the same kind of checkpoint
you already have). One script handles every dataset you throw at it —
what changes between datasets is a handful of command-line flags, not
the code.

---

## 1. What it does

For any dataset of captcha images, the script runs a 4-step loop:

1. **Prepare** — run the current model on a batch of your images and
   write its raw guesses to a CSV.
2. **You correct** — open that CSV and type the true answer for each
   image into one column (`corrected_text`). Nothing else needs editing.
3. **Train** — the script reads your corrections, fine-tunes the model
   on them, and saves the fine-tuned weights.
4. **Evaluate** — it re-predicts the training images (to show how much
   accuracy improved) and predicts a separate, unseen test set (with no
   labels required).

What makes a "correct answer" is decided by a **task type**, which is
the one thing you choose per dataset:

| Task type         | For datasets where...                                   | The "value" compared for accuracy |
|--------------------|----------------------------------------------------------|-------------------------------------|
| `math_expression`  | the image shows a short sum, e.g. `35-7`, `۱۲+۸`          | the *solved result* (e.g. `28`)     |
| `plain_text`       | the image shows text/alphanumeric characters with nothing to solve | the *text itself*                   |

Everything else — model loading, image augmentation, the training loop,
the CSV bookkeeping — is shared code and never needs to change.

---

## 2. Requirements

See `requirements.txt`:

```
torch>=2.2
transformers>=4.41
pandas>=2.0
numpy>=1.24
Pillow>=10.0
```

Install with:

```bash
pip install -r requirements.txt
```

If you're training on a GPU, install a CUDA-matched `torch` build first
(see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)),
then install the rest.

**No internet access is used at runtime.** The model is loaded strictly
from a local folder — nothing is downloaded from Hugging Face.

---

## 3. Folder layout you need

For **each** dataset you fine-tune on, you need three things:

```
your_model_folder/          <- a local copy of the base OCR model
    config.json
    generation_config.json
    preprocessor_config.json
    tokenizer_config.json
    model.safetensors        (or pytorch_model.bin)
    tokenizer.json            (or vocab.json + merges.txt)
    ...

your_dataset/train/          <- images you will manually correct + train on
    0001.png
    0002.png
    ...

your_dataset/test/           <- held-out images, predicted but not trained on
    0001.png
    ...
```

The **same** `your_model_folder` can be reused across every dataset —
you don't need a separate copy per dataset. Each dataset just needs its
own `train/` and `test/` image folders.

> The image filename is only used to match rows in the CSV back to
> files. It is **never** treated as the answer, no matter what it's
> named.

---

## 4. Quick start

### 4a. Math-expression captchas

```bash
# Step 1 — generate the CSV to correct
python finetune_general.py \
    --task_type math_expression \
    --model_dir  ./ocr-captcha-v3 \
    --train_dir  ./datasets/dissel/train \
    --test_dir   ./datasets/dissel/test \
    --run_name   dissel \
    --prepare
```

Open `outputs/dissel/initial_predictions.csv`. For each row, fill in
`corrected_text` with the expression as shown in the image (not the
answer):

| filename | recognized_text | corrected_text |
|----------|------------------|-----------------|
| 0001.png | ۷۷-۱ (wrong)      | 77-10           |

Save the file, then:

```bash
# Step 2 — fine-tune on your corrections and evaluate
python finetune_general.py \
    --task_type math_expression \
    --model_dir  ./ocr-captcha-v3 \
    --train_dir  ./datasets/dissel/train \
    --test_dir   ./datasets/dissel/test \
    --run_name   dissel \
    --train
```

### 4b. Plain-text / alphanumeric captchas

Same two commands, different `--task_type` and a different dataset —
**the script itself is untouched**:

```bash
python finetune_general.py \
    --task_type plain_text \
    --model_dir  ./ocr-captcha-v3 \
    --train_dir  ./datasets/alnum/train \
    --test_dir   ./datasets/alnum/test \
    --run_name   alnum \
    --prepare

# ...edit outputs/alnum/initial_predictions.csv, corrected_text column...

python finetune_general.py \
    --task_type plain_text \
    --model_dir  ./ocr-captcha-v3 \
    --train_dir  ./datasets/alnum/train \
    --test_dir   ./datasets/alnum/test \
    --run_name   alnum \
    --train
```

Each `--run_name` writes into its own subfolder under `--output_dir`
(default: the script's own folder), so datasets never overwrite each
other's CSVs or fine-tuned models:

```
outputs/
  dissel/
    initial_predictions.csv
    after_finetune_predictions.csv
    test_predictions.csv
    finetuned_model/
  alnum/
    initial_predictions.csv
    ...
```

---

## 5. Command-line reference

### Mode (pick one)

| Flag        | Meaning |
|-------------|---------|
| `--prepare` | Run the model once, write `initial_predictions.csv`, then stop. |
| `--train`   | Read your corrections from that CSV, fine-tune, save, evaluate, and predict the test set. |

Running with neither flag just prints usage help.

### Required for every run

| Flag           | Meaning |
|----------------|---------|
| `--task_type`  | `math_expression` or `plain_text`. |
| `--train_dir`  | Folder of images to correct/train on. |
| `--test_dir`   | Folder of held-out images for final prediction. |

### Paths (optional)

| Flag           | Default                      | Meaning |
|----------------|-------------------------------|---------|
| `--model_dir`  | the script's own folder       | Where the local HF model files live. |
| `--output_dir` | the script's own folder       | Where run subfolders are created. |
| `--run_name`   | the `train_dir` folder's name  | Namespaces this run's outputs. |

### Data selection (optional)

| Flag            | Default        | Meaning |
|-----------------|----------------|---------|
| `--train_count` | all images     | Use only a random subset of `train_dir` (deterministic given `--seed`). |
| `--test_count`  | all images     | Same, for `test_dir`. |
| `--seed`        | `42`           | Controls the shuffling above and general reproducibility. |

### Training hyperparameters (optional)

| Flag                  | Default | Meaning |
|-----------------------|---------|---------|
| `--epochs`            | `15`    | Fine-tuning epochs. |
| `--train_batch_size`  | `4`     | Batch size while training. |
| `--eval_batch_size`   | `8`     | Batch size while predicting. |
| `--learning_rate`     | `2e-5`  | AdamW learning rate. |
| `--weight_decay`      | `0.01`  | AdamW weight decay. |
| `--warmup_ratio`      | `0.1`   | Fraction of steps spent warming up the LR. |
| `--num_beams`         | `4`     | Beam search width during generation. |
| `--max_target_length` | `16`    | Max token length of a label. |
| `--no_augmentation`   | off     | Disable the mild rotate/brightness/contrast/blur augmentation used while training. |

### `math_expression`-specific (optional)

| Flag                    | Default          | Meaning |
|--------------------------|------------------|---------|
| `--math_digit_scripts`   | `persian,arabic` | Comma list of digit scripts folded into ASCII before parsing. ASCII digits are always accepted regardless. Choices: `persian`, `arabic`, `ascii`. |
| `--math_display_script`  | `persian`        | Script used when showing `recognized_text` in the CSV, purely for your readability. Choices: `persian`, `arabic`, `ascii`. |
| `--math_operators`       | `+,-`            | Comma list of operators to support. Add `*` and/or `/` if your captchas use them, e.g. `+,-,*,/`. |

### `plain_text`-specific (optional)

| Flag                          | Default    | Meaning |
|-------------------------------|------------|---------|
| `--plain_text_case`           | `preserve` | `preserve`, `lower`, or `upper` — case-folding applied before comparison. |
| `--plain_text_charset`        | (none)     | If set, only these characters survive normalization; everything else is dropped as OCR garbage. E.g. `"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"`. |
| `--plain_text_keep_whitespace`| off        | By default, runs of whitespace collapse to a single space. Set this to preserve whitespace exactly. |

---

## 6. The CSV files

All three CSVs (`initial_predictions.csv`, `after_finetune_predictions.csv`,
`test_predictions.csv`) share the same columns:

| Column              | Meaning |
|---------------------|---------|
| `filename`          | Image file, for matching only — never the label. |
| `recognized_text`   | What the model read from the image (display-formatted). |
| `predicted_value`   | The task's "solved" value from `recognized_text` (e.g. the arithmetic result, or the text itself for `plain_text`). |
| `confidence`        | Beam search log-score (not a percentage; higher is more confident). |
| `corrected_text`    | **The only column you fill in.** The true label, as read from the image. |
| `expected_value`    | The task's "solved" value from `corrected_text`, computed automatically. |
| `correct`           | `1` if `predicted_value == expected_value`, else `0`; blank if not yet corrected. |
| `text_exact_match`  | `1` if `recognized_text` exactly matches `corrected_text` character-for-character; blank/`0` if not yet corrected. |

You only ever type into `corrected_text`. Everything else is
recalculated by the script — if you edit `expected_value` yourself, it
will be silently overwritten from `corrected_text` the next time you
run `--train`.

**Why two accuracy numbers?** `correct` tells you whether the model got
the right *answer* (useful for math, where a slightly garbled OCR read
can still solve correctly by luck, or a correct-looking read can be off
by one digit and solve wrong). `text_exact_match` tells you whether the
model's raw reading was character-perfect. For `plain_text` tasks the
two numbers are effectively the same, since the "value" *is* the text.

---

## 7. Output files (after `--train`)

Inside `outputs/<run_name>/`:

| File                              | Contents |
|------------------------------------|----------|
| `initial_predictions.csv`          | Created by `--prepare`; your corrections live here. |
| `after_finetune_predictions.csv`   | Training images re-predicted after fine-tuning, with accuracy printed to the console. |
| `test_predictions.csv`             | Predictions on the unseen test set (unlabeled — `corrected_text`/`expected_value`/`correct` are left blank for you to fill in later if you want to score it). |
| `finetuned_model/`                 | The fine-tuned model + processor, saved with `save_pretrained`, ready to reload as `--model_dir` for further fine-tuning or for inference elsewhere. |

---

## 8. Tips

- **Reusing a fine-tuned model as the next starting point:** pass a
  previous run's `outputs/<run_name>/finetuned_model` as `--model_dir`
  for the next dataset, if you want continual fine-tuning across
  datasets rather than always starting from the original base model.
- **Only some images corrected?** That's fine — `--train` will print a
  warning and fine-tune on whatever rows have a non-empty
  `corrected_text`, skipping the rest.
- **Invalid corrections are skipped, not silently accepted.** If a row's
  `corrected_text` can't be parsed by the task (e.g. `math_expression`
  can't find a `digits-operator-digits` pattern in it), the script
  prints `[correction] SKIP <filename>: ...` and leaves that image out
  of training.
- **Rerunning `--prepare`** overwrites `initial_predictions.csv`, so
  don't rerun it after you've already started filling in corrections
  unless you mean to start over.

---

## 9. Adding a new task type

If a future dataset doesn't fit `math_expression` or `plain_text`, add a
new `TaskAdapter` subclass near the top of the script and register it in
`TASK_REGISTRY`. It only needs to implement:

- `normalize(raw)` — turn raw OCR output or a manual correction into a
  canonical string.
- `is_valid(normalized)` — whether that string is an acceptable label.
- `solve(normalized)` — the "value" to compare for correctness.
- `display(normalized)` *(optional)* — cosmetic formatting for the CSV.

Nothing else in the file needs to change — the dataset loading, model
loading, training loop, and CSV workflow already work generically
against this interface.
