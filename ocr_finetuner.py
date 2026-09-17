#!/usr/bin/env python3
"""
General local TrOCR captcha fine-tuning script.

This is a dataset-agnostic version of a fine-tuning workflow originally
written for one specific captcha type (Persian math expressions). Instead
of hardcoding the dataset's quirks into the script, all dataset-specific
logic (digit normalization, validity checks, "solving" the label, display
formatting) lives behind a pluggable TaskAdapter, selected with --task_type.
Everything else (model loading, dataset/dataloader, augmentation,
generation config, training loop, the prepare -> manual-correction -> train
-> evaluate workflow, CSV handling) is shared and dataset-agnostic.

Built-in task types
--------------------
  math_expression
      For captchas that show a short arithmetic expression (e.g. "35-7").
      Handles Persian/Arabic/ASCII digits and +, -, *, / operators, all
      configurable. The "value" of a sample is the solved result.

  plain_text
      For captchas that just show text/alphanumeric strings with nothing
      to solve. The "value" of a sample is the (normalized) text itself.

Typical workflow for ONE dataset
---------------------------------
  1) Point --model_dir at a local TrOCR checkpoint (config.json,
     model.safetensors, tokenizer files, etc. all present in that folder;
     nothing is downloaded from Hugging Face).

  2) Create the initial CSV for manual correction:

       python finetune_general.py \
           --task_type math_expression \
           --model_dir ./ocr-captcha-v3 \
           --train_dir ./datasets/dissel/train \
           --test_dir  ./datasets/dissel/test \
           --run_name dissel \
           --prepare

  3) Open outputs/dissel/initial_predictions.csv and fill in ONLY the
     corrected_text column with the ground truth for each image.

  4) Fine-tune and evaluate:

       python finetune_general.py \
           --task_type math_expression \
           --model_dir ./ocr-captcha-v3 \
           --train_dir ./datasets/dissel/train \
           --test_dir  ./datasets/dissel/test \
           --run_name dissel \
           --train

To fine-tune on a DIFFERENT dataset (say, a plain alphanumeric captcha),
you do not touch this script -- just change the flags:

       python finetune_general.py \
           --task_type plain_text \
           --model_dir ./ocr-captcha-v3 \
           --train_dir ./datasets/alnum/train \
           --test_dir  ./datasets/alnum/test \
           --run_name alnum \
           --prepare
       # ...correct outputs/alnum/initial_predictions.csv...
       python finetune_general.py --task_type plain_text \
           --model_dir ./ocr-captcha-v3 \
           --train_dir ./datasets/alnum/train \
           --test_dir  ./datasets/alnum/test \
           --run_name alnum \
           --train

Each --run_name gets its own subfolder under --output_dir, so multiple
datasets never overwrite each other's CSVs or fine-tuned model.
"""

import argparse
import os
import random
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import (
    GenerationConfig,
    PreTrainedTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    set_seed,
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ============================================================
# CONSTANTS
# ============================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}

# Generic CSV schema shared by every task type.
CSV_COLUMNS = [
    "filename",
    "recognized_text",     # raw OCR output, task.display()-formatted
    "predicted_value",     # task.solve() applied to the OCR output
    "confidence",          # beam search sequence log-score
    "corrected_text",      # human-provided ground truth (blank until filled in)
    "expected_value",      # task.solve() applied to corrected_text
    "correct",             # predicted_value == expected_value
    "text_exact_match",    # recognized_text == corrected_text (char-exact)
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ============================================================
# TASK ADAPTERS
#
# A TaskAdapter is the ONLY place dataset-specific behaviour lives.
# Add a new subclass + register it in TASK_REGISTRY to support a new
# kind of captcha without touching anything else in this file.
# ============================================================

class TaskAdapter(ABC):
    """Dataset-specific label handling, pluggable into the shared pipeline."""

    name: str = "base"

    @abstractmethod
    def normalize(self, raw: Any) -> str:
        """Convert raw OCR output or a manual correction into canonical form.

        Must be idempotent when applied to its own display() output, since
        recognized_text is stored in display form and later re-normalized
        for comparisons.
        """
        raise NotImplementedError

    @abstractmethod
    def is_valid(self, normalized: str) -> bool:
        """Whether a normalized string is an acceptable label for this task."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, normalized: str) -> Optional[str]:
        """Derive the 'value' used for correctness comparisons.

        For a math task this is the computed result; for a plain-text task
        this is just the normalized text itself. Returns None if the label
        can't be resolved to a value.
        """
        raise NotImplementedError

    def display(self, normalized: str) -> str:
        """Optional cosmetic formatting for CSV/console display.

        Defaults to identity. A task may override this (e.g. to render
        ASCII digits back into Persian digits) purely for human readability;
        it must not change the value normalize() would produce if re-applied.
        """
        return normalized

    def example_hint(self) -> str:
        """Short human-readable example printed after --prepare."""
        return (
            "Example:\n"
            "  image shows:      <whatever the image shows>\n"
            "  corrected_text:   <the same thing, typed exactly>\n"
        )


# ---------------- math_expression ----------------

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

DIGIT_SCRIPTS: Dict[str, str] = {
    "persian": PERSIAN_DIGITS,
    "arabic": ARABIC_DIGITS,
    "ascii": ASCII_DIGITS,
}

# Every character variant that should be folded into a canonical operator.
# Extend these sets directly if a new dataset uses a symbol not listed here.
_OPERATOR_CHAR_MAP: Dict[str, str] = {
    "+": "+", "＋": "+",
    "-": "-", "−": "-", "–": "-", "—": "-", "_": "-", "ـ": "-",
    "*": "*", "×": "*", "x": "*", "X": "*", "٭": "*",
    "/": "/", "÷": "/", "∕": "/",
}


def _format_number(value: float) -> str:
    """Render an int-or-float result as a clean string."""
    if float(value).is_integer():
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


class MathExpressionTask(TaskAdapter):
    """Short arithmetic expressions, e.g. '35-7', '۱۲+۸', '40 ÷ 5'."""

    name = "math_expression"

    def __init__(
        self,
        input_digit_scripts: Sequence[str] = ("persian", "arabic"),
        display_script: str = "persian",
        enabled_ops: Sequence[str] = ("+", "-"),
    ) -> None:
        for script in input_digit_scripts:
            if script not in DIGIT_SCRIPTS:
                raise ValueError(
                    f"Unknown digit script '{script}'. "
                    f"Choices: {sorted(DIGIT_SCRIPTS)}"
                )
        if display_script not in DIGIT_SCRIPTS:
            raise ValueError(
                f"Unknown display script '{display_script}'. "
                f"Choices: {sorted(DIGIT_SCRIPTS)}"
            )

        enabled_ops_set = set(enabled_ops)
        unknown_ops = enabled_ops_set - set("+-*/")
        if unknown_ops:
            raise ValueError(
                f"Unsupported math operator(s): {sorted(unknown_ops)}. "
                "Supported: +, -, *, /"
            )
        self._enabled_ops = enabled_ops_set

        # Build a translate table folding every configured input digit
        # script into ASCII. ASCII digits always pass through unchanged.
        src = "".join(
            DIGIT_SCRIPTS[s] for s in input_digit_scripts if s != "ascii"
        )
        dst = ASCII_DIGITS * (len(src) // 10)
        self._digit_translate = str.maketrans(src, dst)

        self._display_translate = (
            None
            if display_script == "ascii"
            else str.maketrans(ASCII_DIGITS, DIGIT_SCRIPTS[display_script])
        )

        self._char_to_op = {
            ch: op
            for ch, op in _OPERATOR_CHAR_MAP.items()
            if op in enabled_ops_set
        }

        op_class = "".join(re.escape(op) for op in sorted(enabled_ops_set))
        self._pattern = re.compile(rf"(\d+)([{op_class}])(\d+)")

    def normalize(self, raw: Any) -> str:
        if raw is None:
            return ""

        text = str(raw).strip()

        if text.lower() == "nan":
            return ""

        text = text.translate(self._digit_translate)

        result: List[str] = []
        for ch in text:
            if ch in ASCII_DIGITS:
                result.append(ch)
            elif ch in self._char_to_op:
                result.append(self._char_to_op[ch])
            else:
                # Ignore spaces, "=", and OCR garbage.
                continue

        return "".join(result)

    def is_valid(self, normalized: str) -> bool:
        return bool(self._pattern.fullmatch(normalized))

    def solve(self, normalized: str) -> Optional[str]:
        match = self._pattern.fullmatch(normalized)
        if not match:
            return None

        left = int(match.group(1))
        op = match.group(2)
        right = int(match.group(3))

        if op == "+":
            return str(left + right)
        if op == "-":
            return str(left - right)
        if op == "*":
            return str(left * right)
        if op == "/":
            if right == 0:
                return None
            return _format_number(left / right)

        return None

    def display(self, normalized: str) -> str:
        if self._display_translate is None:
            return normalized
        return normalized.translate(self._display_translate)

    def example_hint(self) -> str:
        return (
            "Example:\n"
            "  image shows:      77 - 10 =\n"
            "  corrected_text:   77-10\n\n"
            "expected_value is calculated automatically (67).\n"
            "Do NOT type the answer itself into corrected_text."
        )


# ---------------- plain_text ----------------

class PlainTextTask(TaskAdapter):
    """Plain text / alphanumeric captchas with no arithmetic involved."""

    name = "plain_text"

    def __init__(
        self,
        case: str = "preserve",
        charset: Optional[str] = None,
        keep_whitespace: bool = False,
    ) -> None:
        if case not in ("preserve", "lower", "upper"):
            raise ValueError("case must be one of: preserve, lower, upper")

        self._case = case
        self._charset = set(charset) if charset else None
        self._keep_whitespace = keep_whitespace

    def normalize(self, raw: Any) -> str:
        if raw is None:
            return ""

        text = str(raw).strip()

        if text.lower() == "nan":
            return ""

        if not self._keep_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        if self._case == "lower":
            text = text.lower()
        elif self._case == "upper":
            text = text.upper()

        if self._charset is not None:
            text = "".join(ch for ch in text if ch in self._charset)

        return text

    def is_valid(self, normalized: str) -> bool:
        return len(normalized) > 0

    def solve(self, normalized: str) -> Optional[str]:
        return normalized if normalized else None

    def example_hint(self) -> str:
        return (
            "Example:\n"
            "  image shows:      aB3xQ\n"
            "  corrected_text:   aB3xQ\n\n"
            "Just type exactly what the image shows."
        )


TASK_REGISTRY = {
    "math_expression": MathExpressionTask,
    "plain_text": PlainTextTask,
}


def build_task(args: argparse.Namespace) -> TaskAdapter:
    if args.task_type == "math_expression":
        digit_scripts = [s.strip() for s in args.math_digit_scripts.split(",") if s.strip()]
        ops = [s.strip() for s in args.math_operators.split(",") if s.strip()]
        return MathExpressionTask(
            input_digit_scripts=digit_scripts,
            display_script=args.math_display_script,
            enabled_ops=ops,
        )

    if args.task_type == "plain_text":
        return PlainTextTask(
            case=args.plain_text_case,
            charset=args.plain_text_charset,
            keep_whitespace=args.plain_text_keep_whitespace,
        )

    raise ValueError(f"Unknown task_type '{args.task_type}'")


# ============================================================
# IMAGE
# ============================================================

def load_image(path: str) -> Image.Image:
    img = Image.open(path)

    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)

    return img.convert("RGB")


def augment(img: Image.Image) -> Image.Image:
    """Mild training-only augmentation. Dataset-agnostic."""

    if random.random() < 0.5:
        img = img.rotate(
            random.uniform(-2.0, 2.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(255, 255, 255),
        )

    if random.random() < 0.35:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.10))

    if random.random() < 0.35:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90, 1.10))

    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    return img


# ============================================================
# TOKENIZER
# ============================================================

def get_tokenizer(processor: TrOCRProcessor) -> PreTrainedTokenizer:
    return cast(PreTrainedTokenizer, processor.tokenizer)


# ============================================================
# DATASET
# ============================================================

@dataclass
class Sample:
    path: str
    text: str
    expected_value: Optional[str]


@dataclass
class Batch:
    pixel_values: torch.Tensor
    labels: torch.Tensor
    texts: List[str]
    paths: List[str]
    expected_values: List[Optional[str]]


class OCRDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        processor: TrOCRProcessor,
        is_train: bool,
        use_augmentation: bool,
        max_target_length: int,
    ) -> None:
        self.samples = list(samples)
        self.processor = processor
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        self.max_target_length = max_target_length
        self.tokenizer = get_tokenizer(processor)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        img = load_image(sample.path)

        if self.use_augmentation:
            img = augment(img)

        processor_any = cast(Any, self.processor)
        processor_output = processor_any(images=img, return_tensors="pt")
        pixel_values = processor_output.pixel_values[0]

        tokenizer_any = cast(Any, self.tokenizer)
        encoded = tokenizer_any(
            sample.text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        labels = encoded.input_ids.squeeze(0).clone()

        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": sample.text,
            "path": sample.path,
            "expected_value": sample.expected_value,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Batch:
    return Batch(
        pixel_values=torch.stack([item["pixel_values"] for item in batch]),
        labels=torch.stack([item["labels"] for item in batch]),
        texts=[item["text"] for item in batch],
        paths=[item["path"] for item in batch],
        expected_values=[item["expected_value"] for item in batch],
    )


# ============================================================
# DATA LOADING
# ============================================================

def list_images(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        raise RuntimeError(f"Directory does not exist:\n{directory}")

    paths: List[str] = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(filename)[1].lower()
        if extension in IMG_EXTS:
            paths.append(path)

    return sorted(paths)


def load_image_items(directory: str) -> List[Tuple[str, Optional[str]]]:
    """
    Load (image_path, expected_value) pairs.

    The filename is only an image ID and is NEVER interpreted as the
    label. Ground truth is only ever supplied later via corrected_text.
    """
    paths = list_images(directory)

    if not paths:
        raise RuntimeError(f"No images found in:\n{directory}")

    items: List[Tuple[str, Optional[str]]] = [(path, None) for path in paths]

    print(f"[data] Loaded {len(items)} images from {directory}")

    return items


def select_items(
    items: Sequence[Tuple[str, Optional[str]]],
    count: Optional[int],
    seed: int,
) -> List[Tuple[str, Optional[str]]]:
    """Select `count` items (shuffled, deterministic given `seed`).

    If count is None, every item is used (no shuffling needed).
    """
    if count is None:
        return list(items)

    if count > len(items):
        raise ValueError(
            f"Requested {count} images, but only {len(items)} are available."
        )

    indices = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    return [items[index] for index in indices[:count]]


# ============================================================
# LOCAL MODEL ONLY
# ============================================================

CORE_REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
]
WEIGHT_FILE_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]
# Any ONE of these groups being fully present is enough (different
# tokenizer exports use different file sets).
TOKENIZER_FILE_GROUPS = [
    ["tokenizer.json"],
    ["vocab.json", "merges.txt"],
]


def verify_local_model(model_dir: str) -> None:
    print()
    print("[model] Checking LOCAL model files:")

    missing: List[str] = []

    for filename in CORE_REQUIRED_FILES:
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK       {filename:<28} {size_mb:8.2f} MB")
        else:
            print(f"  MISSING  {filename}")
            missing.append(filename)

    weight_found = None
    for filename in WEIGHT_FILE_CANDIDATES:
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK       {filename:<28} {size_mb:8.2f} MB")
            weight_found = filename
            break
    if weight_found is None:
        print(f"  MISSING  one of {WEIGHT_FILE_CANDIDATES}")
        missing.append(f"one of {WEIGHT_FILE_CANDIDATES}")

    tokenizer_group_found = None
    for group in TOKENIZER_FILE_GROUPS:
        if all(os.path.isfile(os.path.join(model_dir, f)) for f in group):
            for filename in group:
                path = os.path.join(model_dir, filename)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  OK       {filename:<28} {size_mb:8.2f} MB")
            tokenizer_group_found = group
            break
    if tokenizer_group_found is None:
        options = " or ".join(str(g) for g in TOKENIZER_FILE_GROUPS)
        print(f"  MISSING  tokenizer files ({options})")
        missing.append(f"tokenizer files ({options})")

    if missing:
        raise FileNotFoundError(
            "The local model is incomplete.\n\nMissing:\n"
            + "\n".join(missing)
            + f"\n\nPut the missing files in:\n{model_dir}"
        )


def load_local_model(
    model_dir: str,
) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:

    verify_local_model(model_dir)

    print()
    print("[model] Loading processor from LOCAL files...")
    processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)

    print("[model] Loading model weights from LOCAL files...")
    model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)

    print()
    print("[model] Local model loaded successfully.")
    print("[model] No Hugging Face download is used.")

    return processor, model


# ============================================================
# GENERATION CONFIG
# ============================================================

def valid_token_id(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        if value >= 0:
            return int(value)
    return None


def first_valid_token(values: Sequence[Any], default: int) -> int:
    for value in values:
        token_id = valid_token_id(value)
        if token_id is not None:
            return token_id
    return default


def build_generation_config(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    max_target_length: int,
    num_beams: int,
) -> GenerationConfig:
    """
    Do NOT modify model.config.max_length or model.config.num_beams.

    New Transformers versions reject that old generation strategy.
    We use GenerationConfig and pass it explicitly to generate().
    """

    tokenizer = get_tokenizer(processor)

    model_config = model.config
    decoder_config = getattr(model_config, "decoder", None)

    start_candidates: List[Any] = []
    if decoder_config is not None:
        start_candidates.extend([
            getattr(decoder_config, "decoder_start_token_id", None),
            getattr(decoder_config, "bos_token_id", None),
            getattr(decoder_config, "cls_token_id", None),
        ])
    start_candidates.extend([
        getattr(model_config, "decoder_start_token_id", None),
        tokenizer.cls_token_id,
        tokenizer.bos_token_id,
    ])
    decoder_start_token_id = first_valid_token(start_candidates, 0)

    eos_candidates: List[Any] = [tokenizer.sep_token_id, tokenizer.eos_token_id]
    if decoder_config is not None:
        eos_candidates.append(getattr(decoder_config, "eos_token_id", None))
    eos_candidates.append(getattr(model_config, "eos_token_id", None))
    eos_token_id = first_valid_token(eos_candidates, decoder_start_token_id)

    pad_candidates: List[Any] = [tokenizer.pad_token_id]
    if decoder_config is not None:
        pad_candidates.append(getattr(decoder_config, "pad_token_id", None))
    pad_candidates.append(getattr(model_config, "pad_token_id", None))
    pad_token_id = first_valid_token(pad_candidates, 1)

    generation_config = GenerationConfig(
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_length=max_target_length,
        num_beams=num_beams,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    print()
    print("[model] GenerationConfig:")
    print(f"  decoder_start_token_id = {decoder_start_token_id}")
    print(f"  eos_token_id           = {eos_token_id}")
    print(f"  pad_token_id           = {pad_token_id}")
    print(f"  max_length             = {max_target_length}")
    print(f"  num_beams              = {num_beams}")

    return generation_config


# ============================================================
# CONFIDENCE
# ============================================================

def sequence_confidence(generation_output: Any, index: int) -> float:
    """sequences_scores are log scores, not percentages."""
    sequence_scores = getattr(generation_output, "sequences_scores", None)

    if sequence_scores is None:
        return float("nan")

    try:
        return float(sequence_scores[index].item())
    except (AttributeError, IndexError, TypeError):
        return float("nan")


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_items(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    image_items: Sequence[Tuple[str, Optional[str]]],
    device: torch.device,
    generation_config: GenerationConfig,
    batch_size: int,
    max_target_length: int,
    task: TaskAdapter,
) -> pd.DataFrame:

    model.eval()

    if not image_items:
        return pd.DataFrame(columns=CSV_COLUMNS)

    # text="" because this is inference; labels are unused during generate().
    inference_samples = [
        Sample(path=path, text="", expected_value=expected_value)
        for path, expected_value in image_items
    ]

    dataset = OCRDataset(
        samples=inference_samples,
        processor=processor,
        is_train=False,
        use_augmentation=False,
        max_target_length=max_target_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    rows: List[Dict[str, Any]] = []
    global_index = 0

    generation_model = cast(Any, model)
    processor_any = cast(Any, processor)

    for batch in loader:
        pixel_values = batch.pixel_values.to(device)

        generation_output = generation_model.generate(
            pixel_values,
            generation_config=generation_config,
        )

        sequences = getattr(generation_output, "sequences", generation_output)
        sequences = cast(torch.Tensor, sequences)

        predictions = processor_any.batch_decode(sequences, skip_special_tokens=True)

        for local_index, raw_prediction in enumerate(predictions):
            path, expected_value = image_items[global_index]

            canonical = task.normalize(raw_prediction)
            predicted_value = task.solve(canonical)
            display_text = task.display(canonical)

            # No ground-truth label is available at plain inference time.
            # Do not mark an unlabeled sample as wrong.
            answer_correct: Any = ""
            if (
                expected_value is not None
                and predicted_value is not None
                and predicted_value == expected_value
            ):
                answer_correct = 1

            rows.append({
                "filename": os.path.basename(path),
                "recognized_text": display_text,
                "predicted_value": predicted_value if predicted_value is not None else "",
                "confidence": sequence_confidence(generation_output, local_index),
                "corrected_text": "",
                "expected_value": expected_value if expected_value is not None else "",
                "correct": answer_correct,
                "text_exact_match": 0,
            })

            global_index += 1

    return pd.DataFrame(rows)


# ============================================================
# READ MANUAL CORRECTIONS
# ============================================================

def read_corrections(
    csv_path: str,
    selected_items: Sequence[Tuple[str, Optional[str]]],
    task: TaskAdapter,
) -> List[Sample]:
    """
    Read manual corrections.

    ONLY corrected_text is ground truth supplied by the user. expected_value
    is always (re)calculated from corrected_text; any stale value already in
    the CSV is overwritten. The filename is NEVER used as the label.
    """
    if not os.path.isfile(csv_path):
        return []

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    required = {"filename", "corrected_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    path_map: Dict[str, str] = {
        os.path.basename(path): path for path, _ in selected_items
    }

    corrected_samples: List[Sample] = []

    for index, row in df.iterrows():
        filename = str(row["filename"]).strip()
        if filename not in path_map:
            continue

        corrected_raw = str(row.get("corrected_text", "")).strip()
        if not corrected_raw:
            continue

        normalized = task.normalize(corrected_raw)
        if not task.is_valid(normalized):
            print(f"[correction] SKIP {filename}: invalid label '{corrected_raw}'")
            continue

        value = task.solve(normalized)
        if value is None:
            print(f"[correction] SKIP {filename}: could not resolve '{normalized}'")
            continue

        # Always trust the manually corrected text.
        # Never trust a stale expected_value from an older CSV.
        df.at[index, "corrected_text"] = normalized
        if "expected_value" in df.columns:
            df.at[index, "expected_value"] = str(value)

        corrected_samples.append(
            Sample(path=path_map[filename], text=normalized, expected_value=value)
        )

    # Persist the (re)calculated values so the CSV becomes self-consistent.
    save_csv(df, csv_path)

    return corrected_samples


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    samples: Sequence[Sample],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    use_augmentation: bool,
    max_target_length: int,
) -> None:

    if not samples:
        print("[train] No corrected samples.")
        return

    print()
    print(f"[train] Fine-tuning on {len(samples)} manually corrected images.")

    dataset = OCRDataset(
        samples=samples,
        processor=processor,
        is_train=True,
        use_augmentation=use_augmentation,
        max_target_length=max_target_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    if len(loader) == 0:
        print("[train] No batches.")
        return

    total_steps = len(loader) * epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        remaining = total_steps - step
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, remaining / decay_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model_any = cast(Any, model)

    for epoch in range(1, epochs + 1):
        model_any.train()
        running_loss = 0.0

        for batch in loader:
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model_any(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            if loss is None:
                raise RuntimeError("The model did not return a training loss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())

        average_loss = running_loss / len(loader)
        print(f"[train] epoch {epoch:02d}/{epochs:02d} loss={average_loss:.5f}")

    model.eval()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(dataframe: pd.DataFrame, name: str) -> None:
    if dataframe.empty:
        print(f"[eval] {name}: empty dataset")
        return

    total = len(dataframe)

    labeled = dataframe[
        dataframe["expected_value"].astype(str).str.strip() != ""
    ].copy()

    if not labeled.empty:
        labeled_correct = pd.to_numeric(labeled["correct"], errors="coerce").fillna(0)
        accuracy = labeled_correct.mean() * 100.0
        correct_count = int(labeled_correct.sum())
        wrong_count = len(labeled) - correct_count
    else:
        accuracy = float("nan")
        correct_count = 0
        wrong_count = 0

    exact_labeled = labeled[
        labeled["corrected_text"].astype(str).str.strip() != ""
    ]

    if not exact_labeled.empty:
        exact_match = (
            pd.to_numeric(exact_labeled["text_exact_match"], errors="coerce")
            .fillna(0)
            .mean()
            * 100.0
        )
    else:
        exact_match = float("nan")

    print()
    print(f"[eval] {name}")
    print(f"  Total               : {total}")
    if labeled.empty:
        print("  Ground-truth labels : 0")
        print("  Accuracy            : not available (no labels)")
    else:
        print(f"  Ground-truth labels : {len(labeled)}")
        print(f"  Correct value       : {correct_count}")
        print(f"  Wrong value         : {wrong_count}")
        print(f"  Value accuracy      : {accuracy:.2f}%")

    if exact_labeled.empty:
        print("  Text exact-match    : not available")
    else:
        print(f"  Text exact-match    : {exact_match:.2f}%")


# ============================================================
# CSV
# ============================================================

def save_csv(dataframe: pd.DataFrame, path: str) -> None:
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[csv] Saved: {path}")


def update_text_exact_match(dataframe: pd.DataFrame, task: TaskAdapter) -> pd.DataFrame:
    df = dataframe.copy()

    if df.empty:
        return df

    def compare(row: pd.Series) -> int:
        corrected = task.normalize(row.get("corrected_text", ""))
        # recognized_text is stored in display() form; re-normalize it so
        # the comparison is apples-to-apples with corrected_text.
        recognized = task.normalize(row.get("recognized_text", ""))

        if not corrected:
            return 0

        return int(corrected == recognized)

    df["text_exact_match"] = df.apply(compare, axis=1)

    return df


# ============================================================
# SAVE FINE-TUNED MODEL
# ============================================================

def save_finetuned_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    run_output_dir: str,
) -> str:

    model_dir = os.path.join(run_output_dir, "finetuned_model")
    os.makedirs(model_dir, exist_ok=True)

    print()
    print("[model] Saving fine-tuned model...")
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    print(f"[model] Saved to:\n{model_dir}")

    return model_dir


# ============================================================
# MAIN
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "General local TrOCR captcha fine-tuning with a pluggable "
            "task adapter, reusable across different datasets."
        )
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare", action="store_true",
        help="Create initial_predictions.csv and stop.",
    )
    mode.add_argument(
        "--train", action="store_true",
        help="Read the corrected CSV, fine-tune, and evaluate.",
    )

    parser.add_argument(
        "--task_type", required=True, choices=sorted(TASK_REGISTRY),
        help="Which label logic to use for this dataset.",
    )

    parser.add_argument(
        "--model_dir", default=None,
        help="Folder containing the local HF model files. "
             "Defaults to this script's own folder.",
    )
    parser.add_argument(
        "--train_dir", required=True,
        help="Folder of images to prepare/correct/fine-tune on.",
    )
    parser.add_argument(
        "--test_dir", required=True,
        help="Folder of held-out images for final unlabeled prediction.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where run output subfolders are created. "
             "Defaults to this script's own folder.",
    )
    parser.add_argument(
        "--run_name", default=None,
        help="Namespaces this dataset's CSVs/model under output_dir/run_name. "
             "Defaults to the train_dir folder name.",
    )

    parser.add_argument("--train_count", type=int, default=None,
                         help="Subset size from train_dir. Default: use all images.")
    parser.add_argument("--test_count", type=int, default=None,
                         help="Subset size from test_dir. Default: use all images.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_target_length", type=int, default=16)
    parser.add_argument("--no_augmentation", action="store_true")

    # math_expression-specific
    parser.add_argument(
        "--math_digit_scripts", default="persian,arabic",
        help="Comma list of digit scripts to fold into ASCII before parsing. "
             f"Choices: {sorted(DIGIT_SCRIPTS)}",
    )
    parser.add_argument(
        "--math_display_script", default="persian", choices=sorted(DIGIT_SCRIPTS),
        help="Script used to render recognized_text for human review.",
    )
    parser.add_argument(
        "--math_operators", default="+,-",
        help="Comma list of operators to support: any of +,-,*,/",
    )

    # plain_text-specific
    parser.add_argument(
        "--plain_text_case", default="preserve", choices=["preserve", "lower", "upper"],
        help="Case-folding applied during normalization.",
    )
    parser.add_argument(
        "--plain_text_charset", default=None,
        help="If set, only these characters are kept during normalization "
             "(everything else is treated as OCR garbage and dropped).",
    )
    parser.add_argument(
        "--plain_text_keep_whitespace", action="store_true",
        help="By default internal whitespace is collapsed to single spaces; "
             "set this to preserve it as-is.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.prepare and not args.train:
        parser.print_help()
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = args.model_dir or base_dir
    output_dir = args.output_dir or base_dir
    run_name = args.run_name or os.path.basename(os.path.normpath(args.train_dir)) or "run"
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    task = build_task(args)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 60)
    print("General Local TrOCR Captcha Fine-Tuning")
    print("=" * 60)
    print(f"[env] torch        = {torch.__version__}")
    print(f"[env] transformers = {transformers.__version__}")
    print(f"[env] device       = {device}")
    if device.type == "cuda":
        print(f"[env] GPU          = {torch.cuda.get_device_name(0)}")

    print()
    print(f"[task] type         = {task.name}")
    print(f"[run]  name          = {run_name}")
    print(f"[path] model_dir     = {model_dir}")
    print(f"[path] train_dir     = {args.train_dir}")
    print(f"[path] test_dir      = {args.test_dir}")
    print(f"[path] run_output    = {run_output_dir}")

    if not os.path.isdir(args.train_dir):
        raise RuntimeError(f"train_dir not found:\n{args.train_dir}")
    if not os.path.isdir(args.test_dir):
        raise RuntimeError(f"test_dir not found:\n{args.test_dir}")

    processor, model = load_local_model(model_dir)
    model_any = cast(Any, model)
    model = cast(VisionEncoderDecoderModel, model_any.to(device))

    generation_config = build_generation_config(
        model=model,
        processor=processor,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
    )

    initial_csv = os.path.join(run_output_dir, "initial_predictions.csv")
    after_csv = os.path.join(run_output_dir, "after_finetune_predictions.csv")
    test_csv = os.path.join(run_output_dir, "test_predictions.csv")

    # --------------------------------------------------------
    # PREPARE MODE
    # --------------------------------------------------------
    if args.prepare:
        print()
        print("=" * 60)
        print("STEP 1 - CREATE INITIAL CSV")
        print("=" * 60)

        all_train_images = load_image_items(args.train_dir)
        train_items = select_items(all_train_images, args.train_count, args.seed)

        print(f"[data] train_dir contains {len(all_train_images)} images.")
        print(f"[data] Selected {len(train_items)} images.")

        initial_df = predict_items(
            model=model,
            processor=processor,
            image_items=train_items,
            device=device,
            generation_config=generation_config,
            batch_size=args.eval_batch_size,
            max_target_length=args.max_target_length,
            task=task,
        )

        # CRITICAL: corrected_text/expected_value are intentionally blank.
        # The filename is only an ID and is never interpreted as a label.
        initial_df["corrected_text"] = ""
        initial_df["expected_value"] = ""
        initial_df["correct"] = ""
        initial_df["text_exact_match"] = ""

        save_csv(initial_df[CSV_COLUMNS], initial_csv)

        print()
        print("IMPORTANT: edit ONLY the corrected_text column.")
        print()
        print(task.example_hint())
        print()
        print("Do NOT use the filename as the label.")
        print()
        print("When finished, save the CSV and run the same command with --train instead of --prepare.")
        print()
        return

    # --------------------------------------------------------
    # TRAIN MODE
    # --------------------------------------------------------
    if not os.path.isfile(initial_csv):
        raise RuntimeError(
            f"{initial_csv} does not exist.\n\nRun this first with --prepare."
        )

    print()
    print("=" * 60)
    print("STEP 1 - LOAD MANUAL CORRECTIONS")
    print("=" * 60)

    all_train_images = load_image_items(args.train_dir)
    train_items = select_items(all_train_images, args.train_count, args.seed)

    corrected_samples = read_corrections(initial_csv, train_items, task)

    print(
        f"[correction] Valid corrected labels: "
        f"{len(corrected_samples)}/{len(train_items)}"
    )

    if not corrected_samples:
        raise RuntimeError(
            "No valid corrected_text values were found.\n\n"
            f"Open {initial_csv} and fill in the corrected_text column.\n\n"
            + task.example_hint()
        )

    if len(corrected_samples) < len(train_items):
        print()
        print("[warning] Some images have no correction.")
        print("[warning] Only corrected images will be used for training.")

    print()
    print("=" * 60)
    print("STEP 2 - FINE-TUNING")
    print("=" * 60)

    train_model(
        model=model,
        processor=processor,
        samples=corrected_samples,
        device=device,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        use_augmentation=not args.no_augmentation,
        max_target_length=args.max_target_length,
    )

    finetuned_dir = save_finetuned_model(model, processor, run_output_dir)

    print()
    print("=" * 60)
    print("STEP 3 - RE-PREDICT TRAINING IMAGES")
    print("=" * 60)

    after_df = predict_items(
        model=model,
        processor=processor,
        image_items=train_items,
        device=device,
        generation_config=generation_config,
        batch_size=args.eval_batch_size,
        max_target_length=args.max_target_length,
        task=task,
    )

    correction_map = {
        os.path.basename(sample.path): sample.text for sample in corrected_samples
    }
    value_map = {
        os.path.basename(sample.path): str(sample.expected_value)
        for sample in corrected_samples
    }

    after_df["corrected_text"] = after_df["filename"].map(correction_map).fillna("")
    after_df["expected_value"] = after_df["filename"].map(value_map).fillna("")
    after_df = update_text_exact_match(after_df, task)

    def _correct(row: Any) -> Any:
        expected = str(row["expected_value"]).strip()
        predicted = str(row["predicted_value"]).strip()
        return int(bool(expected) and predicted == expected) if expected else ""

    after_df["correct"] = after_df.apply(_correct, axis=1)

    save_csv(after_df[CSV_COLUMNS], after_csv)
    evaluate(after_df, f"AFTER FINE-TUNE / {run_name}")

    print()
    print("=" * 60)
    print("STEP 4 - TEST UNSEEN IMAGES")
    print("=" * 60)

    all_test_images = load_image_items(args.test_dir)
    test_items = select_items(all_test_images, args.test_count, args.seed + 1)

    print(f"[data] test_dir contains {len(all_test_images)} images.")
    print(f"[data] Selected {len(test_items)} TEST images.")

    test_df = predict_items(
        model=model,
        processor=processor,
        image_items=test_items,
        device=device,
        generation_config=generation_config,
        batch_size=args.eval_batch_size,
        max_target_length=args.max_target_length,
        task=task,
    )

    # test_dir is unlabeled by default. Do not invent expected values.
    test_df["corrected_text"] = ""
    test_df["expected_value"] = ""
    test_df["correct"] = ""
    test_df["text_exact_match"] = ""

    save_csv(test_df[CSV_COLUMNS], test_csv)

    print()
    print(f"[test] Saved predictions: {test_csv}")
    print()
    print("Test accuracy is not calculated because test_dir has no labels.")
    print("If you add corrected_text labels for it later, they can be")
    print("evaluated the same way, without ever using filenames as labels.")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"[output] Initial CSV:      {initial_csv}")
    print(f"[output] Fine-tuned CSV:   {after_csv}")
    print(f"[output] Test predictions: {test_csv}")
    print(f"[output] Fine-tuned model: {finetuned_dir}")


if __name__ == "__main__":
    main()