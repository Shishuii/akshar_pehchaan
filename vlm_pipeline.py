"""
VLM-based Electric Meter OCR Pipeline
======================================
Uses Qwen2-VL (primary) or SmolVLM (lightweight alternative) to extract
structured information from electric meter images via natural-language Q&A.

Models are stored in ./models/ directory.
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

METER_PROMPT = """\
You are a specialist in reading 7-segment LCD/LED electric utility meter displays with extreme precision.

## STEP 1 — READ THE 7-SEGMENT DISPLAY (most critical task)

The main display uses 7-segment digits (like a digital clock). Read it with these rules:

**Digit-by-digit analysis:**
- Scan from LEFT to RIGHT across all digit positions, including leading zeros
- Each digit is formed by up to 7 lit segments (a=top, b=upper-right, c=lower-right, d=bottom, e=lower-left, f=upper-left, g=middle)
- Segments that are OFF appear dark/dim; segments that are ON appear bright/glowing

**Common 7-segment confusion pairs — read carefully:**
- 0 vs 8: "0" has NO middle segment (g); "8" has ALL 7 segments lit
- 6 vs 8: "6" has top segment (a) dark; "8" has top segment lit
- 1 vs 7: "1" uses only right two segments; "7" also has the top segment (a) lit
- 5 vs 6: "5" has lower-left (e) dark; "6" has lower-left lit
- 2 vs 3: "2" has upper-left (f) dark; "3" has upper-right (b) dark side unlit on left

**Decimal point detection (critical):**
- A decimal point is a SMALL DOT located at the BOTTOM-RIGHT corner BETWEEN two digit positions
- Look carefully for any tiny illuminated dot between digits — it is easy to miss
- Count digit positions: e.g. if dot is after position 5 of 8 digits, reading is XXXXX.XXX
- If you see a decimal point, include it in the reading at the exact position (e.g. "00523.40")
- If no decimal point is visible, do NOT add one

**Units on or near the display:**
- Look for unit labels ON the display glass itself or printed directly adjacent to it
- Common units: kWh, KWH, kVAh, KVAh, MWh, kW, kVA, W, VA, A, V, Hz
- Some displays show the active unit as a lit label segment (e.g. "kWh" lights up)
- Record exactly what unit label is visible on or adjacent to the display

## STEP 2 — READ METER LABEL INFORMATION

- Manufacturer name: printed on front panel (Landis+Gyr, Elster, Itron, Genus, Secure, HPL, L&T, Schneider, Havells, Iskraemeco)
- Serial number: alphanumeric code on sticker/embossed label, often starts with letters (e.g. GE7422324)
- Meter type: single phase (1P, 1-phase) or three phase (3P, 3-phase)
- Voltage rating: e.g. 230V, 240V, 415V, 3×230V/400V
- Current rating: e.g. 5-30A, 10-60A, 5(60)A

## STEP 3 — OUTPUT JSON ONLY

Return this exact JSON structure with NO other text, NO markdown fences, NO explanation.
For each field, also include a confidence score (0.0 = not visible/uncertain, 1.0 = clearly visible and certain):

{
  "manufacturer": "brand name or null",
  "manufacturer_confidence": 0.0,
  "serial_number": "serial number string or null",
  "serial_number_confidence": 0.0,
  "display_reading": "exact digits with decimal point as shown, e.g. 00523.40 or 1234567 — NO spaces, NO units",
  "display_reading_confidence": 0.0,
  "display_unit": "unit label ON or immediately adjacent to display (kWh/kVAh/kW/kVA/MWh/etc.) or null",
  "display_unit_confidence": 0.0,
  "power_unit": "primary energy unit of the meter (kWh, kVAh, MWh) or null",
  "power_unit_confidence": 0.0,
  "md_kw": "Maximum Demand kW value if a separate reading is visible, else null",
  "md_kw_confidence": 0.0,
  "demand_kva": "Demand kVA value if a separate reading is visible, else null",
  "demand_kva_confidence": 0.0,
  "kvah_reading": "kVAh reading if a separate reading is visible, else null",
  "kvah_reading_confidence": 0.0,
  "meter_type": "single phase / three phase / other or null",
  "meter_type_confidence": 0.0,
  "voltage_rating": "voltage rating string or null",
  "voltage_rating_confidence": 0.0,
  "current_rating": "current rating string or null",
  "current_rating_confidence": 0.0,
  "decimal_point_position": "digit index (0-based from left) AFTER which decimal appears, e.g. 5 means XXXXX.XX, or null if none",
  "digit_count": "total number of digit positions visible on main display (integer)",
  "notes": "any ambiguous segments, dim digits, or other observations"
}

CRITICAL REMINDERS:
- display_reading must contain ONLY the digits and decimal point — no units, no spaces
- Include ALL leading zeros exactly as shown on the display
- A missed decimal point causes a 10x or 100x error — double-check for small dots
- If a digit is ambiguous, pick the most likely value and note it in "notes"
- Confidence scores: 0.9-1.0 = clearly legible, 0.6-0.8 = readable but some uncertainty, 0.3-0.5 = partially visible, 0.0-0.2 = not visible or guessed
- Return ONLY the JSON object, nothing else"""


# ── RTX 2080 Ti (10.76 GB VRAM) — ranked by OCR accuracy ──────────────────────
#
#  Rank  Model                  fp16 VRAM  int4 VRAM  DocVQA  OCRBench  Notes
#  ────  ─────────────────────  ─────────  ─────────  ──────  ────────  ─────
#   1    MiniCPM-V-2_6-int4     7 GB       7 GB*      ~92%    ~89%      pre-quantised; best meter OCR
#   2    Qwen2.5-VL-3B          ~7 GB      ~3.5 GB    ~85%    ~81%      drop-in upgrade from 2B
#   3    InternVL2-4B           ~10 GB     ~2.5 GB    ~91%    ~82%      tight fp16; use int4
#   4    InternVL2-2B           ~5 GB      ~1.5 GB    94.1%   ~75%      strong DocVQA, small
#   5    PaliGemma2-3B-448      ~8 GB      ~3 GB      ~80%    ~78%      best at 448px resolution
#   6    Qwen2-VL-2B            ~4.5 GB    ~2.5 GB    ~80%    ~72%      current default
#   7    Florence-2-large       ~1.5 GB    <1 GB      —       —         dedicated <OCR> prompt mode
#   8    Florence-2-base        ~0.7 GB    <0.7 GB    —       —         smallest, same OCR capability
#   9    Moondream2             ~3.6 GB    ~2 GB      79.3%   61.2%     very fast, decent OCR
#  10    SmolVLM-500M           ~1 GB      <1 GB      ~67%    ~55%      lightweight fallback
#  11    SmolVLM-256M           ~0.5 GB    <0.5 GB    58.3%   ~50%      edge-device only
#
#  * MiniCPM-V-2_6-int4 ships pre-quantised so no runtime quantisation overhead.
# ───────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS: dict[str, dict] = {
    # ── Tier 1: Best OCR accuracy, fits in 10.76 GB VRAM ──────────────────
    "MiniCPM-V-2_6-int4": {
        "model_id":    "openbmb/MiniCPM-V-2_6-int4",
        "loader":      "minicpm",
        "description": "#1 OCR accuracy (OCRBench ~89%). Pre-quantised int4 — 7GB VRAM. "
                       "Handles fine-grained 7-segment text extremely well.",
        "vram_fp16":   "16 GB (not recommended)",
        "vram_int4":   "~7 GB  ← use this",
        "docvqa":      "~92%",
        "ocrbench":    "~89%",
    },
    "Qwen2.5-VL-3B": {
        "model_id":    "Qwen/Qwen2.5-VL-3B-Instruct",
        "loader":      "qwen2vl",
        "description": "3B upgrade of Qwen2-VL. Native high-res support. "
                       "~7GB fp16 or ~3.5GB int4. Excellent at structured JSON extraction.",
        "vram_fp16":   "~7 GB",
        "vram_int4":   "~3.5 GB",
        "docvqa":      "~85%",
        "ocrbench":    "~81%",
    },
    "InternVL2-4B": {
        "model_id":    "OpenGVLab/InternVL2-4B",
        "loader":      "internvl",
        "description": "4B InternVL2. Very strong DocVQA (~91%). "
                       "Needs AWQ int4 to fit — ~2.5GB. Use trust_remote_code=True.",
        "vram_fp16":   "~10 GB (tight)",
        "vram_int4":   "~2.5 GB (AWQ)",
        "docvqa":      "~91%",
        "ocrbench":    "~82%",
    },
    "InternVL2-2B": {
        "model_id":    "OpenGVLab/InternVL2-2B",
        "loader":      "internvl",
        "description": "2B InternVL2. 94.1% DocVQA — punches above its size. "
                       "~5GB fp16 or ~1.5GB int4. Strong meter label reader.",
        "vram_fp16":   "~5 GB",
        "vram_int4":   "~1.5 GB",
        "docvqa":      "94.1%",
        "ocrbench":    "~75%",
    },
    "PaliGemma2-3B-448": {
        "model_id":    "google/paligemma2-3b-mix-448",
        "loader":      "paligemma",
        "description": "PaliGemma2 at 448×448px resolution — the sweet spot for LCD displays. "
                       "+33pt DocVQA vs 224px. ~8GB fp16.",
        "vram_fp16":   "~8 GB",
        "vram_int4":   "~3 GB",
        "docvqa":      "~80%",
        "ocrbench":    "~78%",
    },
    # ── Tier 2: Current defaults ────────────────────────────────────────────
    "Qwen2-VL-2B": {
        "model_id":    "Qwen/Qwen2-VL-2B-Instruct",
        "loader":      "qwen2vl",
        "description": "Original 2B Qwen2-VL. Solid baseline, 4.5GB fp16. "
                       "Good structured JSON extraction. Well-tested in this pipeline.",
        "vram_fp16":   "~4.5 GB",
        "vram_int4":   "~2.5 GB",
        "docvqa":      "~80%",
        "ocrbench":    "~72%",
    },
    # ── Tier 3: Dedicated OCR mode (no chat, task-based prompt) ────────────
    "Florence-2-large": {
        "model_id":    "microsoft/Florence-2-large",
        "loader":      "florence2",
        "description": "770M params. Dedicated <OCR> and <OCR_WITH_REGION> prompts "
                       "return text + bounding-box coordinates. 1.5GB fp16. "
                       "Best for pure text extraction without chat overhead.",
        "vram_fp16":   "~1.5 GB",
        "vram_int4":   "<1 GB",
        "docvqa":      "N/A (task-based)",
        "ocrbench":    "N/A",
    },
    "Florence-2-base": {
        "model_id":    "microsoft/Florence-2-base",
        "loader":      "florence2",
        "description": "230M params. Same <OCR> capability as large, smallest possible. "
                       "Under 0.7GB fp16. Ideal when VRAM is shared with other models.",
        "vram_fp16":   "~0.7 GB",
        "vram_int4":   "<0.7 GB",
        "docvqa":      "N/A (task-based)",
        "ocrbench":    "N/A",
    },
    # ── Tier 4: Lightweight / fast inference ───────────────────────────────
    "Moondream2": {
        "model_id":    "vikhyatk/moondream2",
        "loader":      "moondream",
        "description": "1.8B. Very fast (edge-device level). ~3.6GB fp16. "
                       "79.3% DocVQA, 61.2% OCRBench. Good for quick checks.",
        "vram_fp16":   "~3.6 GB",
        "vram_int4":   "~2 GB",
        "docvqa":      "79.3%",
        "ocrbench":    "61.2%",
    },
    "SmolVLM-500M": {
        "model_id":    "HuggingFaceTB/SmolVLM-500M-Instruct",
        "loader":      "smolvlm",
        "description": "500M. ~1GB fp16. Good balance of speed and accuracy for labels.",
        "vram_fp16":   "~1 GB",
        "vram_int4":   "<1 GB",
        "docvqa":      "~67%",
        "ocrbench":    "~55%",
    },
    "SmolVLM-256M": {
        "model_id":    "HuggingFaceTB/SmolVLM-256M-Instruct",
        "loader":      "smolvlm",
        "description": "256M ultra-lightweight. 0.5GB fp16. Edge/CPU fallback only.",
        "vram_fp16":   "~0.5 GB",
        "vram_int4":   "<0.5 GB",
        "docvqa":      "58.3%",
        "ocrbench":    "~50%",
    },
}


class VLMPipeline:
    """Vision Language Model pipeline for electric meter OCR."""

    def __init__(self, model_key: str = "Qwen2-VL-2B"):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. Choose from: {list(SUPPORTED_MODELS)}"
            )
        self.model_key = model_key
        self.model_info = SUPPORTED_MODELS[model_key]
        self.model_id = self.model_info["model_id"]
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """Load model and processor (lazy — call before first inference)."""
        if self._loaded:
            return
        loader = self.model_info.get("loader", "smolvlm")
        logger.info("Loading VLM: %s (%s) from %s ...", self.model_key, loader, self.model_id)
        dispatch = {
            "qwen2vl":   self._load_qwen2vl,
            "smolvlm":   self._load_smolvlm,
            "minicpm":   self._load_minicpm,
            "internvl":  self._load_internvl,
            "paligemma": self._load_paligemma,
            "florence2": self._load_florence2,
            "moondream": self._load_moondream,
        }
        load_fn = dispatch.get(loader)
        if load_fn is None:
            raise ValueError(f"Unknown loader '{loader}' for model '{self.model_key}'")
        load_fn()
        self._loaded = True
        logger.info("%s ready.", self.model_key)

    def _device_dtype(self):
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32

    def _load_qwen2vl(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        device, dtype = self._device_dtype()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(MODEL_DIR),
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            cache_dir=str(MODEL_DIR),
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    def _load_smolvlm(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        device, dtype = self._device_dtype()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(MODEL_DIR),
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=str(MODEL_DIR),
        )
        self.model = self.model.to(device)
        self.model.eval()

    def _load_minicpm(self):
        """MiniCPM-V-2_6-int4 — pre-quantised, trust_remote_code required."""
        from transformers import AutoModel, AutoTokenizer

        device, _ = self._device_dtype()
        self.processor = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=str(MODEL_DIR),
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            cache_dir=str(MODEL_DIR),
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    def _load_internvl(self):
        """InternVL2 — dynamic high-res, trust_remote_code required."""
        from transformers import AutoModel, AutoTokenizer

        device, dtype = self._device_dtype()
        self.processor = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=str(MODEL_DIR),
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            cache_dir=str(MODEL_DIR),
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    def _load_paligemma(self):
        """PaliGemma2 — standard HF VLM, no trust_remote_code needed."""
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        device, dtype = self._device_dtype()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(MODEL_DIR),
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            cache_dir=str(MODEL_DIR),
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    def _load_florence2(self):
        """Florence-2 — task-based VLM, trust_remote_code required."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        device, dtype = self._device_dtype()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=str(MODEL_DIR),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            cache_dir=str(MODEL_DIR),
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    def _load_moondream(self):
        """Moondream2 — trust_remote_code required, uses custom chat API."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = self._device_dtype()
        self.processor = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=str(MODEL_DIR),
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            cache_dir=str(MODEL_DIR),
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def extract(self, image_path: str) -> dict:
        """
        Run VLM inference on a meter image.

        Returns:
            {
              "model": str,
              "success": bool,
              "fields": {manufacturer, serial_number, display_reading,
                         display_unit, power_unit, md_kw, demand_kva,
                         kvah_reading, meter_type, voltage_rating,
                         current_rating, decimal_point_position,
                         digit_count, notes},
              "raw_response": str,
              "error": str | None
            }
        """
        if not self._loaded:
            self.load()

        try:
            image = Image.open(image_path).convert("RGB")
            loader = self.model_info.get("loader", "smolvlm")
            infer_dispatch = {
                "qwen2vl":   self._infer_qwen2vl,
                "smolvlm":   self._infer_smolvlm,
                "minicpm":   self._infer_minicpm,
                "internvl":  self._infer_internvl,
                "paligemma": self._infer_paligemma,
                "florence2": self._infer_florence2,
                "moondream": self._infer_moondream,
            }
            infer_fn = infer_dispatch.get(loader, self._infer_smolvlm)
            raw = infer_fn(image)

            fields = self._parse_json(raw)
            return {
                "model": self.model_key,
                "success": True,
                "fields": fields,
                "raw_response": raw,
                "error": None,
            }
        except Exception as exc:
            logger.exception("VLM inference failed: %s", exc)
            return {
                "model": self.model_key,
                "success": False,
                "fields": _empty_fields(),
                "raw_response": "",
                "error": str(exc),
            }

    def _infer_qwen2vl(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": METER_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        trimmed = [
            out_ids[i][len(inputs.input_ids[i]) :]
            for i in range(len(out_ids))
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

    def _infer_smolvlm(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": METER_PROMPT},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        device = next(self.model.parameters()).device
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
            )

        return self.processor.decode(out_ids[0], skip_special_tokens=True)

    def _infer_minicpm(self, image: Image.Image) -> str:
        """MiniCPM-V-2_6 uses model.chat() with a msgs list."""
        msgs = [{"role": "user", "content": [image, METER_PROMPT]}]
        with torch.no_grad():
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.processor,
            )
        return response if isinstance(response, str) else str(response)

    def _infer_internvl(self, image: Image.Image) -> str:
        """InternVL2 uses model.chat() with pixel_values + generation_config."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import GenerationConfig

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        pixel_values = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

        generation_config = GenerationConfig(
            max_new_tokens=768,
            do_sample=False,
        )
        question = f"<image>\n{METER_PROMPT}"
        with torch.no_grad():
            response = self.model.chat(
                self.processor,
                pixel_values,
                question,
                generation_config,
            )
        return response if isinstance(response, str) else str(response)

    def _infer_paligemma(self, image: Image.Image) -> str:
        """PaliGemma2 — standard processor + generate."""
        device = next(self.model.parameters()).device
        # PaliGemma uses a short task prefix; the detailed prompt follows
        prompt = METER_PROMPT
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(out_ids[0][input_len:], skip_special_tokens=True)

    def _infer_florence2(self, image: Image.Image) -> str:
        """Florence-2 uses task-based prompts, not chat.

        We run three tasks and stitch the results into a JSON-like string
        that _parse_json can handle.
        """
        device = next(self.model.parameters()).device

        def _run_task(task_token: str) -> str:
            inputs = self.processor(
                text=task_token,
                images=image,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=3,
                )
            raw = self.processor.batch_decode(out_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                raw,
                task=task_token,
                image_size=(image.width, image.height),
            )
            # parsed is a dict like {"<OCR>": "text ..."}
            return parsed.get(task_token, "")

        ocr_text = _run_task("<OCR>")
        # Build a minimal JSON so _parse_json can extract display_reading etc.
        # Florence-2 returns raw concatenated text — we surface it in notes
        # and let the user see the display_reading heuristic extraction.
        reading = self._florence_extract_reading(ocr_text)
        result_json = json.dumps({
            "manufacturer": None,
            "serial_number": None,
            "display_reading": reading,
            "display_unit": None,
            "power_unit": None,
            "md_kw": None,
            "demand_kva": None,
            "kvah_reading": None,
            "meter_type": None,
            "voltage_rating": None,
            "current_rating": None,
            "decimal_point_position": None,
            "digit_count": None,
            "notes": f"Florence-2 raw OCR: {ocr_text[:400]}",
        })
        return result_json

    @staticmethod
    def _florence_extract_reading(ocr_text: str) -> Optional[str]:
        """Pull the longest digit+decimal sequence from Florence-2 OCR text."""
        candidates = re.findall(r"\d[\d.]{2,}\d", ocr_text)
        if not candidates:
            return None
        return max(candidates, key=len)

    def _infer_moondream(self, image: Image.Image) -> str:
        """Moondream2 uses model.answer_question() with an encoded image."""
        with torch.no_grad():
            enc = self.model.encode_image(image)
            response = self.model.answer_question(
                enc,
                METER_PROMPT,
                self.processor,
            )
        return response if isinstance(response, str) else str(response)

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract and parse the first JSON object found in text."""
        # Try to find JSON block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())

                def _conf(key):
                    v = data.get(f"{key}_confidence")
                    if v is None:
                        return None
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None

                # Normalize keys
                return {
                    "manufacturer":                   data.get("manufacturer"),
                    "manufacturer_confidence":         _conf("manufacturer"),
                    "serial_number":                   data.get("serial_number"),
                    "serial_number_confidence":        _conf("serial_number"),
                    "display_reading":                 data.get("display_reading"),
                    "display_reading_confidence":      _conf("display_reading"),
                    "display_unit":                    data.get("display_unit"),
                    "display_unit_confidence":         _conf("display_unit"),
                    "power_unit":                      data.get("power_unit"),
                    "power_unit_confidence":           _conf("power_unit"),
                    "md_kw":                           data.get("md_kw"),
                    "md_kw_confidence":                _conf("md_kw"),
                    "demand_kva":                      data.get("demand_kva"),
                    "demand_kva_confidence":           _conf("demand_kva"),
                    "kvah_reading":                    data.get("kvah_reading"),
                    "kvah_reading_confidence":         _conf("kvah_reading"),
                    "meter_type":                      data.get("meter_type"),
                    "meter_type_confidence":           _conf("meter_type"),
                    "voltage_rating":                  data.get("voltage_rating"),
                    "voltage_rating_confidence":       _conf("voltage_rating"),
                    "current_rating":                  data.get("current_rating"),
                    "current_rating_confidence":       _conf("current_rating"),
                    "decimal_point_position":          data.get("decimal_point_position"),
                    "digit_count":                     data.get("digit_count"),
                    "notes":                           data.get("notes"),
                }
            except json.JSONDecodeError:
                pass
        return _empty_fields(notes=f"JSON parse failed. Raw: {text[:300]}")


def _empty_fields(notes: Optional[str] = None) -> dict:
    return {
        "manufacturer":                 None,
        "manufacturer_confidence":      None,
        "serial_number":                None,
        "serial_number_confidence":     None,
        "display_reading":              None,
        "display_reading_confidence":   None,
        "display_unit":                 None,
        "display_unit_confidence":      None,
        "power_unit":                   None,
        "power_unit_confidence":        None,
        "md_kw":                        None,
        "md_kw_confidence":             None,
        "demand_kva":                   None,
        "demand_kva_confidence":        None,
        "kvah_reading":                 None,
        "kvah_reading_confidence":      None,
        "meter_type":                   None,
        "meter_type_confidence":        None,
        "voltage_rating":               None,
        "voltage_rating_confidence":    None,
        "current_rating":               None,
        "current_rating_confidence":    None,
        "decimal_point_position":       None,
        "digit_count":                  None,
        "notes":                        notes,
    }
