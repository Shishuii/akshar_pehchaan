"""
Download All Models
====================
Downloads all models used by the meter OCR pipeline into ./models/ directory.

Models downloaded:
  ── Tier 1: Best OCR accuracy ─────────────────────────────────────
  1. openbmb/MiniCPM-V-2_6-int4         — #1 OCR accuracy, ~7 GB int4
  2. Qwen/Qwen2.5-VL-3B-Instruct        — Qwen2.5 upgrade, ~7 GB fp16
  3. OpenGVLab/InternVL2-4B             — DocVQA ~91%, ~2.5 GB AWQ
  4. OpenGVLab/InternVL2-2B             — DocVQA 94.1%, ~5 GB fp16
  5. google/paligemma2-3b-mix-448       — 448px resolution, ~8 GB fp16
  ── Tier 2: Current defaults ───────────────────────────────────────
  6. Qwen/Qwen2-VL-2B-Instruct          — Original default (~4.5 GB fp16)
  ── Tier 3: Dedicated OCR mode ─────────────────────────────────────
  7. microsoft/Florence-2-large         — <OCR> task mode (~1.5 GB)
  8. microsoft/Florence-2-base          — Smallest Florence-2 (~0.7 GB)
  ── Tier 4: Lightweight ────────────────────────────────────────────
  9. vikhyatk/moondream2                — Fast edge VLM (~3.6 GB)
 10. HuggingFaceTB/SmolVLM-500M-Instruct — ~1.0 GB
 11. HuggingFaceTB/SmolVLM-256M-Instruct — ~0.5 GB
  ── Non-VLM ────────────────────────────────────────────────────────
 12. microsoft/trocr-base-printed       — Specialized display OCR (~1.5 GB)
 13. EasyOCR english_g2                 — ./easyocr_models/

Usage:
    python download_models.py                  # download ALL models
    python download_models.py --vlm-only       # only VLM models
    python download_models.py --trocr-only     # only TrOCR
    python download_models.py --easyocr-only   # only EasyOCR
    python download_models.py --tier1-only     # only top-4 accuracy models
    python download_models.py --skip-large     # skip models >5 GB fp16
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

EASYOCR_MODEL_DIR = Path(__file__).parent / "easyocr_models"
EASYOCR_MODEL_DIR.mkdir(exist_ok=True)


def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def download_minicpm():
    """Download MiniCPM-V-2_6-int4 (pre-quantised, ~7 GB)."""
    from transformers import AutoModel, AutoTokenizer

    model_id = "openbmb/MiniCPM-V-2_6-int4"
    logger.info("=" * 60)
    logger.info("Downloading: %s (~7 GB int4, best OCR accuracy)", model_id)

    t0 = time.time()
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(MODEL_DIR))
    AutoModel.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(MODEL_DIR))
    logger.info("MiniCPM-V-2_6-int4 done in %.1fs", time.time() - t0)


def download_qwen25vl():
    """Download Qwen2.5-VL-3B-Instruct (~7 GB fp16)."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    logger.info("=" * 60)
    logger.info("Downloading: %s (~7 GB fp16)", model_id)

    t0 = time.time()
    AutoProcessor.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
    Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", cache_dir=str(MODEL_DIR)
    )
    logger.info("Qwen2.5-VL-3B done in %.1fs", time.time() - t0)


def download_internvl(size: str = "2B"):
    """Download InternVL2-2B or InternVL2-4B."""
    from transformers import AutoModel, AutoTokenizer

    model_id = f"OpenGVLab/InternVL2-{size}"
    size_gb = {"2B": 5.0, "4B": 10.0}
    logger.info("=" * 60)
    logger.info("Downloading: %s (~%.0f GB fp16)", model_id, size_gb.get(size, 5.0))

    t0 = time.time()
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(MODEL_DIR))
    AutoModel.from_pretrained(
        model_id, torch_dtype="auto", trust_remote_code=True, cache_dir=str(MODEL_DIR)
    )
    logger.info("InternVL2-%s done in %.1fs", size, time.time() - t0)


def download_paligemma():
    """Download PaliGemma2-3B-mix-448 (~8 GB fp16)."""
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

    model_id = "google/paligemma2-3b-mix-448"
    logger.info("=" * 60)
    logger.info("Downloading: %s (~8 GB fp16)", model_id)

    t0 = time.time()
    AutoProcessor.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
    PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", cache_dir=str(MODEL_DIR)
    )
    logger.info("PaliGemma2-3B done in %.1fs", time.time() - t0)


def download_florence2(size: str = "large"):
    """Download Florence-2-large or Florence-2-base."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = f"microsoft/Florence-2-{size}"
    size_gb = {"large": 1.5, "base": 0.7}
    logger.info("=" * 60)
    logger.info("Downloading: %s (~%.1f GB fp16)", model_id, size_gb.get(size, 1.5))

    t0 = time.time()
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(MODEL_DIR))
    AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", trust_remote_code=True, cache_dir=str(MODEL_DIR)
    )
    logger.info("Florence-2-%s done in %.1fs", size, time.time() - t0)


def download_moondream():
    """Download Moondream2 (~3.6 GB fp16)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "vikhyatk/moondream2"
    logger.info("=" * 60)
    logger.info("Downloading: %s (~3.6 GB fp16)", model_id)

    t0 = time.time()
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(MODEL_DIR))
    AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=str(MODEL_DIR)
    )
    logger.info("Moondream2 done in %.1fs", time.time() - t0)


def download_qwen2vl():
    """Download Qwen2-VL-2B-Instruct model and processor."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    logger.info("=" * 60)
    logger.info("Downloading: %s (~4.5 GB)", model_id)
    logger.info("Saving to: %s", MODEL_DIR)

    t0 = time.time()
    logger.info("Downloading processor...")
    AutoProcessor.from_pretrained(model_id, cache_dir=str(MODEL_DIR))

    logger.info("Downloading model weights...")
    Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        cache_dir=str(MODEL_DIR),
    )
    logger.info("Qwen2-VL-2B done in %.1fs", time.time() - t0)


def download_smolvlm(size: str = "256M"):
    """Download SmolVLM model."""
    from transformers import AutoProcessor, AutoModelForVision2Seq

    size_map = {"256M": "256M-Instruct", "500M": "500M-Instruct"}
    variant = size_map.get(size, "256M-Instruct")
    model_id = f"HuggingFaceTB/SmolVLM-{variant}"
    size_gb = 0.5 if size == "256M" else 1.0

    logger.info("=" * 60)
    logger.info("Downloading: %s (~%.1f GB)", model_id, size_gb)

    t0 = time.time()
    AutoProcessor.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
    AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
    logger.info("SmolVLM-%s done in %.1fs", size, time.time() - t0)


def download_trocr():
    """Download TrOCR-base-printed model."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_id = "microsoft/trocr-base-printed"
    logger.info("=" * 60)
    logger.info("Downloading: %s (~1.5 GB)", model_id)

    t0 = time.time()
    TrOCRProcessor.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
    VisionEncoderDecoderModel.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
    logger.info("TrOCR done in %.1fs", time.time() - t0)


def download_easyocr():
    """Trigger EasyOCR to download its models."""
    import easyocr

    logger.info("=" * 60)
    logger.info("Downloading EasyOCR models to: %s", EASYOCR_MODEL_DIR)

    t0 = time.time()
    # Just instantiating triggers the download
    _reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(EASYOCR_MODEL_DIR),
        recog_network="english_g2",
        verbose=True,
    )
    logger.info("EasyOCR done in %.1fs", time.time() - t0)


def check_disk_space(required_gb: float):
    import shutil
    _, _, free = shutil.disk_usage(str(MODEL_DIR))
    free_gb = free / 1e9
    if free_gb < required_gb:
        logger.warning(
            "Low disk space: %.1f GB free, %.1f GB required", free_gb, required_gb
        )
        answer = input(f"Continue anyway? [y/N]: ")
        if answer.lower() != "y":
            sys.exit(0)
    else:
        logger.info("Disk space OK: %.1f GB free (need ~%.1f GB)", free_gb, required_gb)


def main():
    parser = argparse.ArgumentParser(description="Download all meter OCR models")
    parser.add_argument("--vlm-only",     action="store_true", help="Only download VLM models")
    parser.add_argument("--trocr-only",   action="store_true", help="Only download TrOCR")
    parser.add_argument("--easyocr-only", action="store_true", help="Only download EasyOCR")
    parser.add_argument("--tier1-only",   action="store_true",
                        help="Only Tier-1 accuracy models (MiniCPM, InternVL2-2B, InternVL2-4B)")
    parser.add_argument("--skip-large",   action="store_true",
                        help="Skip models with fp16 VRAM > 5 GB (PaliGemma, Qwen2.5-VL-3B, MiniCPM)")
    parser.add_argument("--skip-qwen2vl", action="store_true", help="Skip Qwen2-VL-2B (~4.5 GB)")
    parser.add_argument("--skip-smolvlm", action="store_true", help="Skip SmolVLM models")
    args = parser.parse_args()

    logger.info("Model download directory: %s", MODEL_DIR.resolve())

    total_start = time.time()
    errors = []

    def _try(name, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Exception as e:
            logger.error("%s download failed: %s", name, e)
            errors.append((name, str(e)))

    # Determine what to download
    download_all = not any([args.vlm_only, args.trocr_only, args.easyocr_only, args.tier1_only])

    # ── TrOCR ────────────────────────────────────────────────────────────────
    if args.trocr_only or download_all:
        _try("TrOCR", download_trocr)

    # ── VLMs ─────────────────────────────────────────────────────────────────
    if args.vlm_only or args.tier1_only or download_all:

        # Tier 1
        if not args.skip_large:
            _try("MiniCPM-V-2_6-int4", download_minicpm)

        _try("InternVL2-2B", download_internvl, "2B")
        _try("InternVL2-4B", download_internvl, "4B")

        if not args.tier1_only:
            # Tier 1 continued (large)
            if not args.skip_large:
                check_disk_space(8.0)
                _try("Qwen2.5-VL-3B", download_qwen25vl)
                _try("PaliGemma2-3B-448", download_paligemma)

            # Tier 2
            if not args.skip_qwen2vl:
                check_disk_space(5.0)
                _try("Qwen2-VL-2B", download_qwen2vl)

            # Tier 3
            _try("Florence-2-large", download_florence2, "large")
            _try("Florence-2-base",  download_florence2, "base")

            # Tier 4
            _try("Moondream2", download_moondream)

            if not args.skip_smolvlm:
                _try("SmolVLM-256M", download_smolvlm, "256M")
                _try("SmolVLM-500M", download_smolvlm, "500M")

    # ── EasyOCR ───────────────────────────────────────────────────────────────
    if args.easyocr_only or download_all:
        _try("EasyOCR", download_easyocr)

    # Summary
    elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("Download complete in %.1fs", elapsed)
    logger.info("All models saved under: %s", MODEL_DIR.resolve())

    if errors:
        logger.warning("Some downloads failed:")
        for name, err in errors:
            logger.warning("  %-20s %s", name, err)
        sys.exit(1)
    else:
        logger.info("All downloads successful!")
        logger.info("")
        logger.info("To start the UI:")
        logger.info("  python app.py")
        logger.info("")
        logger.info("To run EasyOCR only:")
        logger.info("  python meter_ocr.py <image.jpg>")


if __name__ == "__main__":
    main()
