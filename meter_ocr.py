#!/usr/bin/env python3
"""
Electrical Meter OCR
====================
Extracts per-field readings with confidence scores from smart meter images.

Fields extracted:
  - meter_serial_number  (e.g. GE7422324)
  - serial_reading       (current LCD display value)
  - kWh
  - kVAh
  - MD_kW
  - Demand_kVA

Engine: EasyOCR (open-source, PyTorch-based, accurate, confidence scores)
Models are stored in ./easyocr_models/ inside this project folder.

Install dependencies:
    pip install easyocr opencv-python-headless numpy
"""

import re
import json
import logging
import sys
import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import easyocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent.resolve()
_MODEL_DIR = _HERE / "easyocr_models"
_MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Regex patterns for each target field
# ---------------------------------------------------------------------------
_PATTERNS = {
    "meter_serial_number": [
        r"[Ss]r\.?\s*[Nn]o\.?\s*[:\-]?\s*([A-Z]{1,3}\d{6,10})",
        r"[Ss]\.?\s*[Nn]o\.?\s*[:\-]?\s*([A-Z]{1,3}\d{6,10})",
        r"\b([A-Z]{1,3}\d{7,8})\b",
    ],
    "kWh": [
        r"(\d[\d\s]*[.,]\d+|\d+)\s*kWh",
        r"kWh\s*[:\-]?\s*(\d[\d\s]*[.,]?\d*)",
    ],
    "kVAh": [
        r"(\d[\d\s]*[.,]\d+|\d+)\s*kVAh",
        r"kVAh\s*[:\-]?\s*(\d[\d\s]*[.,]?\d*)",
    ],
    "MD_kW": [
        r"MD\s*kW\s*[:\-]?\s*(\d[\d\s]*[.,]?\d*)",
        r"(\d[\d\s]*[.,]?\d*)\s*MD\s*kW",
    ],
    "Demand_kVA": [
        r"[Dd]emand\s*kVA\s*[:\-]?\s*(\d[\d\s]*[.,]?\d*)",
        r"[Dd]md\s*kVA\s*[:\-]?\s*(\d[\d\s]*[.,]?\d*)",
        r"(\d[\d\s]*[.,]?\d*)\s*[Dd]emand\s*kVA",
    ],
}

_LABEL_TRIGGERS = {
    "kWh":       [r"kWh", r"KWH"],
    "kVAh":      [r"kVAh", r"KVAH"],
    "MD_kW":     [r"MD\s*kW", r"MD\s*KW"],
    "Demand_kVA":[r"[Dd]emand\s*kVA", r"[Dd]MD\s*kVA"],
}


# ---------------------------------------------------------------------------
class MeterOCR:
    """
    Extracts structured meter data from a photo of a smart electricity meter.

    Two OCR passes:
      1. Full image  — captures printed labels and serial numbers.
      2. LCD crop    — isolated green display region with enhanced preprocessing
                       for accurate segmented-digit + decimal reading.
    """

    def __init__(self, use_gpu: bool = False):
        logger.info("Loading EasyOCR (models → %s) …", _MODEL_DIR)
        self.reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            model_storage_directory=str(_MODEL_DIR),
            # Use the most accurate recognition network
            recog_network="english_g2",
            verbose=False,
        )
        logger.info("EasyOCR ready.")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_full_image(img: np.ndarray) -> np.ndarray:
        """Mild denoise + sharpen for printed-text regions."""
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(denoised, -1, kernel)

    @staticmethod
    def _extract_lcd_roi(img: np.ndarray):
        """
        Detect the LCD/LED display panel via colour masking.
        Tries multiple colour ranges to handle green, amber, cyan, and blue displays.
        Returns the RAW crop (no preprocessing) so _extract_lcd_reading can apply
        multiple preprocessing strategies.
        Returns (raw_crop_bgr, (x1, y1, w, h)) or (None, None).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Ordered by prevalence: green, amber/yellow, cyan, blue
        colour_ranges = [
            (np.array([35, 40, 50]),  np.array([90, 255, 255])),   # green
            (np.array([15, 80, 80]),  np.array([35, 255, 255])),   # amber / yellow
            (np.array([80, 40, 80]),  np.array([100, 255, 255])),  # cyan / teal
            (np.array([100, 40, 60]), np.array([130, 255, 255])),  # blue
        ]

        best_roi, best_rect, best_area = None, None, 0
        cl_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        for lo, hi in colour_ranges:
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cl_kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < 500 or area <= best_area:
                continue
            best_area = area
            x, y, w, h = cv2.boundingRect(largest)
            pad = 12
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad); y2 = min(img.shape[0], y + h + pad)
            best_roi = img[y1:y2, x1:x2].copy()
            best_rect = (x1, y1, x2 - x1, y2 - y1)

        if best_roi is not None:
            return best_roi, best_rect

        # Fallback: look for a dark rectangular panel with bright digits
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ih, iw = gray.shape
        cx_lo, cx_hi = iw // 6, 5 * iw // 6
        cy_lo, cy_hi = ih // 5, 4 * ih // 5
        patch = gray[cy_lo:cy_hi, cx_lo:cx_hi]
        blurred = cv2.GaussianBlur(patch, (5, 5), 0)
        _, dark = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, dk)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 2000:
                x, y, w, h = cv2.boundingRect(largest)
                x += cx_lo; y += cy_lo
                pad = 10
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
                return img[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)

        logger.warning("LCD display region not found.")
        return None, None

    # ------------------------------------------------------------------
    # Multi-strategy preprocessing for LCD crops
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_lcd_variants(roi_bgr: np.ndarray) -> list[tuple[np.ndarray, str]]:
        """
        Generate 5 preprocessed versions of the LCD crop for multi-pass OCR.

        Key insight: the decimal point on a 7-segment display is a tiny dot.
        A 2×2 dilation at 3× scale is effectively a 6×6 dilation in the
        original image — it fills the dot and merges it with adjacent strokes.

        Strategies here deliberately avoid isotropic dilation so the dot survives.
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        H = roi_bgr.shape[0]
        # Upscale to ~200-250 px tall; 4× is good for tiny displays
        scale = max(3, min(5, 200 // max(H, 1)))

        def up(g):
            return cv2.resize(g, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        def to_bgr(g):
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        def ensure_dark_on_light(b):
            return b if np.mean(b) >= 127 else cv2.bitwise_not(b)

        variants = []

        # ── 1. CLAHE + Otsu, no dilation ────────────────────────────────
        g = up(clahe.apply(gray))
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((to_bgr(ensure_dark_on_light(b)), "otsu_plain"))

        # ── 2. CLAHE + Otsu + vertical-only dilation ────────────────────
        #    Reconnects broken horizontal digit strokes; the decimal dot is
        #    circular so a (1×3) kernel does NOT bridge it to its neighbour.
        g = up(clahe.apply(gray))
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b = ensure_dark_on_light(b)
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        b = cv2.dilate(b, vk, iterations=1)
        variants.append((to_bgr(b), "otsu_vdil"))

        # ── 3. Adaptive threshold (tolerates uneven backlighting) ────────
        g = up(clahe.apply(gray))
        g = cv2.GaussianBlur(g, (3, 3), 0)
        b = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6)
        variants.append((to_bgr(ensure_dark_on_light(b)), "adaptive"))

        # ── 4. Raw CLAHE grayscale (no binarisation) ────────────────────
        #    EasyOCR handles internal thresholding; this avoids all our errors.
        g = up(clahe.apply(gray))
        g = cv2.GaussianBlur(g, (3, 3), 0)
        variants.append((to_bgr(g), "clahe_gray"))

        # ── 5. Green channel (amplifies contrast on green LCD panels) ────
        _, g_ch, _ = cv2.split(roi_bgr)
        g = up(clahe.apply(g_ch))
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((to_bgr(ensure_dark_on_light(b)), "green_ch"))

        return variants

    # ------------------------------------------------------------------
    # OCR runner
    # ------------------------------------------------------------------

    def _run_ocr(self, img: np.ndarray) -> list[dict]:
        """
        Run EasyOCR on an ndarray.
        Returns list of {text, confidence, bbox, center}.
        EasyOCR result: [(bbox, text, conf), ...]
        bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        raw = self.reader.readtext(
            img,
            detail=1,
            paragraph=False,
            # Prefer higher accuracy over speed
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=1.5,
        )

        detections = []
        for bbox_pts, text, conf in raw:
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            detections.append({
                "text":       text.strip(),
                "confidence": round(float(conf), 4),
                "bbox":       bbox_pts,
                "center":     (sum(xs) / 4, sum(ys) / 4),
            })
        return detections

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_number(raw: str) -> Optional[str]:
        """Remove spaces, normalise comma→dot, return None if not numeric."""
        raw = raw.replace(" ", "").replace(",", ".")
        parts = raw.split(".")
        if len(parts) == 1:
            return parts[0] if parts[0].isdigit() else None
        integer_part = parts[0]
        decimal_part = "".join(parts[1:])
        if integer_part.isdigit() and decimal_part.isdigit():
            return f"{integer_part}.{decimal_part}"
        return None

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------

    def _extract_serial_number(self, detections: list[dict]) -> dict:
        """Match 'Sr. No. GE7422324' or standalone alphanumeric serial."""
        candidates = []

        for i, det in enumerate(detections):
            text = det["text"]

            # Single-token match
            for pat in _PATTERNS["meter_serial_number"]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    candidates.append({
                        "value":      m.group(1).upper(),
                        "confidence": det["confidence"],
                    })
                    break

            # Two-token: label + next detection
            if re.search(r"[Ss]r\.?\s*[Nn]o|[Ss]\.?\s*[Nn]o", text, re.IGNORECASE):
                for j in range(i + 1, min(i + 5, len(detections))):
                    nxt = detections[j]["text"].strip().upper()
                    if re.match(r"^[A-Z]{1,3}\d{6,10}$", nxt):
                        candidates.append({
                            "value":      nxt,
                            "confidence": min(det["confidence"],
                                             detections[j]["confidence"]),
                        })

        if not candidates:
            return {"value": None, "confidence": 0.0}

        best = max(candidates, key=lambda c: c["confidence"])
        return {"value": best["value"], "confidence": best["confidence"]}

    def _extract_lcd_reading(self, img: np.ndarray) -> dict:
        """
        Crop + enhance the LCD panel, run multi-variant OCR, return the best reading.

        Why multi-variant?
        - A single Otsu threshold often fails on dimly lit or overexposed displays.
        - The decimal point is a tiny dot that isotropic dilation destroys.
        - Running 5 preprocessing strategies and scoring + voting the results
          consistently outperforms a single-pass approach.
        """
        roi_bgr, _ = self._extract_lcd_roi(img)

        if roi_bgr is None:
            # Last resort: search the full image for any long numeric string
            dets = self._run_ocr(img)
            return self._best_reading_from_detections(dets)

        variants = self._preprocess_lcd_variants(roi_bgr)
        all_candidates: list[dict] = []

        for variant_img, variant_name in variants:
            dets = self._run_ocr_display(variant_img)
            for d in dets:
                val = self._parse_7seg_text(d["text"])
                if val:
                    score = self._score_reading(val, d["confidence"])
                    all_candidates.append({
                        "value":      val,
                        "confidence": d["confidence"],
                        "score":      score,
                        "variant":    variant_name,
                    })

        if not all_candidates:
            return {"value": None, "confidence": 0.0}

        # Sort by composite score (decimal bonus + length bonus + OCR confidence)
        all_candidates.sort(key=lambda c: -c["score"])
        best = all_candidates[0]

        # Confidence boost when multiple independent variants agree
        n_agree = sum(1 for c in all_candidates if c["value"] == best["value"])
        boosted = min(1.0, best["confidence"] + 0.05 * (n_agree - 1))

        logger.debug(
            "LCD reading '%s' (conf %.2f, agreed %d/%d variants)",
            best["value"], boosted, n_agree, len(variants),
        )
        return {"value": best["value"], "confidence": round(boosted, 4)}

    def _run_ocr_display(self, img: np.ndarray) -> list[dict]:
        """
        EasyOCR pass tuned for 7-segment / LCD numeric displays.

        Key differences from the full-image pass:
        - Lower link_threshold (0.2) so the decimal dot is NOT merged into its
          neighbour digit and is preserved as a separate character.
        - Lower text_threshold (0.5) to catch the small dot.
        - Both paragraph=True and paragraph=False passes are merged so we get
          the full string AND individual tokens.
        """
        common = dict(
            detail=1,
            contrast_ths=0.05,
            adjust_contrast=0.65,
            text_threshold=0.5,
            low_text=0.25,
            link_threshold=0.2,   # critical: keeps decimal dot separate
            canvas_size=3200,
            mag_ratio=2.0,
        )
        # paragraph=True returns (bbox, text) 2-tuples — no confidence score.
        # paragraph=False returns the normal (bbox, text, conf) 3-tuples.
        raw_para  = self.reader.readtext(img, paragraph=True,  **common)
        raw_token = self.reader.readtext(img, paragraph=False, **common)

        seen: set[str] = set()
        detections: list[dict] = []

        for item in raw_para:
            bbox_pts, text = item[0], item[1]
            conf = item[2] if len(item) == 3 else 1.0   # paragraph mode omits conf
            t = text.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            detections.append({
                "text":       t,
                "confidence": round(float(conf), 4),
                "center":     (sum(xs) / 4, sum(ys) / 4),
            })

        for bbox_pts, text, conf in raw_token:
            t = text.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            detections.append({
                "text":       t,
                "confidence": round(float(conf), 4),
                "center":     (sum(xs) / 4, sum(ys) / 4),
            })

        return detections

    @staticmethod
    def _parse_7seg_text(text: str) -> Optional[str]:
        """
        Map common EasyOCR confusions on 7-segment displays and extract
        the numeric reading (digits + at most one decimal point).

        Handles all standard digit-lookalike errors:
          O/Q/D → 0,  I/l/| → 1,  Z/z → 2,  S → 5,
          b/G → 6,   B → 8,   comma → decimal point.
        """
        subs = {
            "O": "0", "o": "0", "Q": "0", "D": "0",
            "I": "1", "l": "1", "|": "1",
            "Z": "2", "z": "2",
            "S": "5",
            "b": "6", "G": "6",
            "B": "8",
            " ": "",
            ",": ".",
        }
        t = text.strip()
        for wrong, right in subs.items():
            t = t.replace(wrong, right)
        t = t.strip(".-+")

        # Accept if the whole token is a valid reading (e.g. "00523.40")
        if re.match(r"^\d{2,}\.?\d*$", t):
            # Reject plausible years to avoid false positives (e.g. "2024")
            if not re.match(r"^(19|20)\d{2}$", t):
                return t

        # Fall back: extract the longest embedded numeric substring
        m = re.search(r"\d{2,}\.?\d*", t)
        if m and not re.match(r"^(19|20)\d{2}$", m.group()):
            return m.group()

        return None

    @staticmethod
    def _score_reading(value: str, confidence: float) -> float:
        """
        Composite score for a candidate meter reading.

        Rewards:
          + OCR engine confidence
          + Typical meter reading length (4–8 total digits)
          + Having a decimal point  (nearly all meter readings do)
          + 2 decimal places        (most common format: XXXXX.XX)
        Penalises:
          - Very short readings (< 3 digits)
        """
        score = confidence
        digits = len(value.replace(".", ""))

        if 4 <= digits <= 8:
            score += 0.20
        elif digits == 3:
            score += 0.05

        if "." in value:
            score += 0.15
            dec = value.split(".")[1]
            score += 0.10 if len(dec) == 2 else 0.05 if len(dec) == 1 else 0.0

        if digits < 3:
            score -= 0.30

        return score

    def _best_reading_from_detections(self, detections: list[dict]) -> dict:
        """Fallback: find the best numeric reading in full-image detections."""
        candidates = []
        for d in detections:
            val = self._parse_7seg_text(d["text"])
            if val:
                score = self._score_reading(val, d["confidence"])
                candidates.append({"value": val, "confidence": d["confidence"],
                                   "score": score})
        if not candidates:
            return {"value": None, "confidence": 0.0}
        best = max(candidates, key=lambda c: c["score"])
        return {"value": best["value"], "confidence": best["confidence"]}

    def _extract_labeled_field(self, detections: list[dict], field: str) -> dict:
        """
        Extract numeric value for a labeled field (kWh, kVAh, MD kW, Demand kVA).
        Strategy 1: regex match within a single detection.
        Strategy 2: label detected → scan spatially adjacent detections for value.
        """
        candidates = []

        for i, det in enumerate(detections):
            text = det["text"]

            # Strategy 1
            for pat in _PATTERNS[field]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    cleaned = self._clean_number(m.group(1))
                    if cleaned:
                        candidates.append({
                            "value":      cleaned,
                            "confidence": det["confidence"],
                            "priority":   2,
                        })
                    break

            # Strategy 2
            for trig in _LABEL_TRIGGERS[field]:
                if re.search(trig, text, re.IGNORECASE):
                    cx, cy = det["center"]
                    for j in range(max(0, i - 3), min(len(detections), i + 6)):
                        if j == i:
                            continue
                        nd = detections[j]
                        nx, ny = nd["center"]
                        dy = abs(ny - cy)
                        dx = abs(nx - cx)
                        if dy < 80 and dx < 400:
                            raw = nd["text"].replace(" ", "").replace(",", ".")
                            if re.match(r"^\d+\.?\d*$", raw):
                                cleaned = self._clean_number(raw)
                                if cleaned:
                                    candidates.append({
                                        "value":      cleaned,
                                        "confidence": min(det["confidence"],
                                                         nd["confidence"]),
                                        "priority":   1,
                                        "dy":         dy,
                                    })
                    break

        if not candidates:
            return {"value": None, "confidence": 0.0}

        candidates.sort(key=lambda c: (-c["priority"], -c["confidence"],
                                        c.get("dy", 999)))
        best = candidates[0]
        return {"value": best["value"], "confidence": best["confidence"]}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> dict:
        """
        Process a single meter image.

        Returns:
            {
                "image": str,
                "fields": {
                    "meter_serial_number": {"value": ..., "confidence": 0-1},
                    "serial_reading":      {"value": ..., "confidence": 0-1},
                    "kWh":                 {"value": ..., "confidence": 0-1},
                    "kVAh":                {"value": ..., "confidence": 0-1},
                    "MD_kW":               {"value": ..., "confidence": 0-1},
                    "Demand_kVA":          {"value": ..., "confidence": 0-1},
                },
                "raw_detections": [{"text": ..., "confidence": ...}, ...]
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        logger.info("Processing: %s", image_path)

        # Pass 1: full image (printed text — serial number, labels, values)
        enhanced = self._enhance_full_image(img)
        detections = self._run_ocr(enhanced)
        logger.info("Full-image OCR: %d regions detected.", len(detections))
        for d in detections:
            logger.debug("  [%.3f] %r", d["confidence"], d["text"])

        # Pass 2: LCD crop (segmented display digits + decimal)
        serial_reading = self._extract_lcd_reading(img)

        # Capture display bounds for crop-saving (cheap second call — no OCR)
        _, display_bbox = self._extract_lcd_roi(img)

        return {
            "image": image_path,
            "fields": {
                "meter_serial_number": self._extract_serial_number(detections),
                "serial_reading":      serial_reading,
                "kWh":                 self._extract_labeled_field(detections, "kWh"),
                "kVAh":                self._extract_labeled_field(detections, "kVAh"),
                "MD_kW":               self._extract_labeled_field(detections, "MD_kW"),
                "Demand_kVA":          self._extract_labeled_field(detections, "Demand_kVA"),
            },
            # bbox included so save_crops / callers can draw or crop regions
            "raw_detections": [
                {"text": d["text"], "confidence": d["confidence"], "bbox": d["bbox"]}
                for d in detections
            ],
            "display_bbox": display_bbox,   # (x1, y1, w, h) or None
        }

    # ------------------------------------------------------------------
    # Crop saving
    # ------------------------------------------------------------------

    def save_crops(self, image_path: str, result: dict, output_dir: str) -> list[str]:
        """
        Save every extracted region from a processed meter image.

        Saved files (all under output_dir/):
          {stem}_annotated.jpg           — full image; OCR boxes in green,
                                           display region highlighted in blue,
                                           matched field boxes in distinct colours
          {stem}_display_raw.jpg         — raw LCD/LED panel crop
          {stem}_display_{variant}.jpg   — each of the 5 preprocessing variants
                                           (otsu_plain, otsu_vdil, adaptive,
                                            clahe_gray, green_ch)
          {stem}_field_{name}.jpg        — tight crop around each matched field
                                           (meter_serial_number, kWh, kVAh, …)

        Returns a list of absolute paths for every file written.
        """
        import json as _json

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        stem = Path(image_path).stem
        out  = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []

        # ── colours per field (BGR) ───────────────────────────────────
        FIELD_COLORS = {
            "meter_serial_number": (0,  165, 255),   # orange
            "kWh":                 (0,  255, 255),   # yellow
            "kVAh":                (255,  0, 255),   # magenta
            "MD_kW":               (255,  128,  0),  # sky blue
            "Demand_kVA":          (128, 255,   0),  # lime
        }
        DISPLAY_COLOR  = (255, 80, 0)    # deep blue for the display panel
        DETECT_COLOR   = (0, 200, 80)    # green for generic detections

        # ── 1. Annotated full image ───────────────────────────────────
        annotated = img.copy()

        # Draw all raw OCR detections (light green, thin)
        for det in result.get("raw_detections", []):
            bbox = det.get("bbox")
            if not bbox:
                continue
            pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
            cv2.polylines(annotated, [pts], True, DETECT_COLOR, 1)
            cx = int(sum(p[0] for p in bbox) / 4)
            cy = int(sum(p[1] for p in bbox) / 4) - 5
            cv2.putText(annotated, det["text"][:20],
                        (cx, max(cy, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, DETECT_COLOR, 1,
                        cv2.LINE_AA)

        # Highlight matched field regions (thick, coloured)
        fields = result.get("fields", {})
        for field_name, color in FIELD_COLORS.items():
            fval = (fields.get(field_name) or {}).get("value")
            if not fval:
                continue
            for det in result.get("raw_detections", []):
                if str(fval) in det["text"] or det["text"] in str(fval):
                    bbox = det.get("bbox")
                    if bbox:
                        pts = np.array([[int(p[0]), int(p[1])] for p in bbox],
                                       dtype=np.int32)
                        cv2.polylines(annotated, [pts], True, color, 3)
                        x0 = int(min(p[0] for p in bbox))
                        y0 = int(min(p[1] for p in bbox)) - 8
                        label = field_name.replace("_", " ")
                        cv2.putText(annotated, label,
                                    (x0, max(y0, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                                    cv2.LINE_AA)
                    break

        # Highlight the display region (blue rectangle)
        display_bbox = result.get("display_bbox")
        if display_bbox:
            x1, y1, w, h = display_bbox
            cv2.rectangle(annotated, (x1, y1), (x1 + w, y1 + h), DISPLAY_COLOR, 3)
            cv2.putText(annotated, "DISPLAY",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, DISPLAY_COLOR, 2,
                        cv2.LINE_AA)

        ann_path = str(out / f"{stem}_annotated.jpg")
        cv2.imwrite(ann_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(ann_path)

        # ── 2. Raw display crop ───────────────────────────────────────
        if display_bbox:
            x1, y1, w, h = display_bbox
            raw_crop = img[y1:y1 + h, x1:x1 + w]
            if raw_crop.size > 0:
                raw_path = str(out / f"{stem}_display_raw.jpg")
                cv2.imwrite(raw_path, raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved.append(raw_path)

                # ── 3. Preprocessing variants of the display ─────────
                for variant_img, variant_name in self._preprocess_lcd_variants(raw_crop):
                    vpath = str(out / f"{stem}_display_{variant_name}.jpg")
                    cv2.imwrite(vpath, variant_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved.append(vpath)

        # ── 4. Individual field crops ─────────────────────────────────
        for field_name in FIELD_COLORS:
            fval = (fields.get(field_name) or {}).get("value")
            if not fval:
                continue
            for det in result.get("raw_detections", []):
                if str(fval) in det["text"] or det["text"] in str(fval):
                    bbox = det.get("bbox")
                    if not bbox:
                        continue
                    xs = [int(p[0]) for p in bbox]
                    ys = [int(p[1]) for p in bbox]
                    pad = 10
                    cx1 = max(0, min(xs) - pad)
                    cy1 = max(0, min(ys) - pad)
                    cx2 = min(img.shape[1], max(xs) + pad)
                    cy2 = min(img.shape[0], max(ys) + pad)
                    crop = img[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        fpath = str(out / f"{stem}_field_{field_name}.jpg")
                        cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved.append(fpath)
                    break

        # ── 5. Write a JSON sidecar with the extracted values ─────────
        meta = {
            "source_image": image_path,
            "fields": {
                k: v for k, v in fields.items()
            },
            "display_bbox": display_bbox,
            "files_saved":  [Path(p).name for p in saved],
        }
        json_path = str(out / f"{stem}_metadata.json")
        with open(json_path, "w") as f:
            _json.dump(meta, f, indent=2)
        saved.append(json_path)

        logger.info("Saved %d crop files → %s", len(saved), out)
        return saved


# ---------------------------------------------------------------------------
# Display-only OCR pipeline
# ---------------------------------------------------------------------------

# Power-unit canonical names and their regex triggers
_UNIT_MAP: list[tuple[str, str]] = [
    (r"kWh",  "kWh"),
    (r"KWH",  "kWh"),
    (r"kwh",  "kWh"),
    (r"kVAh", "kVAh"),
    (r"KVAH", "kVAh"),
    (r"kvah", "kVAh"),
    (r"kVA",  "kVA"),
    (r"KVA",  "kVA"),
    (r"kW",   "kW"),
    (r"KW",   "kW"),
    (r"MWh",  "MWh"),
    (r"MW",   "MW"),
    (r"VAh",  "VAh"),
    (r"Wh",   "Wh"),
]


class DisplayOCR:
    """
    Standalone OCR pipeline for a pre-cropped meter display image.

    Unlike the full-meter pipeline, the entire input image IS the display —
    no LCD region detection step. This allows users to pass in a tight crop
    of just the LCD/LED screen and get back:

      • reading         — numeric value (e.g. "00523.40")
      • decimal_detected — whether a decimal point was found
      • unit            — power unit (kWh, kVAh, kW …)
      • per-field confidence scores
      • full candidate table for transparency

    The pipeline applies the same 5-variant preprocessing strategy as
    MeterOCR._preprocess_lcd_variants and uses the same tuned EasyOCR params
    (low link_threshold so the decimal dot is never merged into digits).
    """

    def __init__(self, use_gpu: bool = False):
        logger.info("Loading DisplayOCR EasyOCR (models → %s) …", _MODEL_DIR)
        self.reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            model_storage_directory=str(_MODEL_DIR),
            recog_network="english_g2",
            verbose=False,
        )
        logger.info("DisplayOCR ready.")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, image_path: str) -> dict:
        """
        Process a display-only image.

        Returns:
            {
              "reading":            str | None,   e.g. "00523.40"
              "reading_confidence": float,         0.0 – 1.0
              "decimal_detected":   bool,
              "decimal_digits":     int,           digits after the decimal
              "unit":               str | None,    e.g. "kWh"
              "unit_confidence":    float,
              "all_candidates":     list[dict],    all scored reading candidates
              "unit_candidates":    list[dict],    all detected unit strings
              "raw_detections":     list[dict],    every OCR box found
              "error":              str | None,
            }
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read: {image_path}")

            # ── Reading: multi-variant on the full input image ────────
            reading_result = self._extract_reading(img)

            # ── Unit: scan all OCR boxes from the original color image ─
            raw_dets   = self._run_ocr_full(img)
            unit_result = self._detect_unit(raw_dets)

            # If unit not found via OCR, try extracting it from the
            # reading candidate text itself (e.g. "00523kWh")
            if unit_result["value"] is None:
                unit_result = self._detect_unit_in_reading_strings(
                    reading_result.get("raw_candidates_text", [])
                )

            reading_val = reading_result["value"]
            decimal_detected = "." in (reading_val or "")
            decimal_digits   = (
                len(reading_val.split(".")[1]) if decimal_detected else 0
            )

            return {
                "reading":            reading_val,
                "reading_confidence": reading_result["confidence"],
                "decimal_detected":   decimal_detected,
                "decimal_digits":     decimal_digits,
                "unit":               unit_result["value"],
                "unit_confidence":    unit_result["confidence"],
                "all_candidates":     reading_result.get("candidates", []),
                "unit_candidates":    unit_result.get("candidates", []),
                "raw_detections":     raw_dets,
                "error":              None,
            }

        except Exception as exc:
            logger.exception("DisplayOCR.extract failed: %s", exc)
            return {
                "reading": None, "reading_confidence": 0.0,
                "decimal_detected": False, "decimal_digits": 0,
                "unit": None, "unit_confidence": 0.0,
                "all_candidates": [], "unit_candidates": [],
                "raw_detections": [], "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    # Reading extraction                                                   #
    # ------------------------------------------------------------------ #

    def _extract_reading(self, img: np.ndarray) -> dict:
        """
        Apply 5-variant preprocessing then run tuned EasyOCR on each.
        Aggregate, score, deduplicate and return the best reading.
        """
        variants = MeterOCR._preprocess_lcd_variants(img)
        all_candidates: list[dict] = []
        all_raw_texts:  list[str]  = []

        for variant_img, variant_name in variants:
            dets = self._run_ocr_display(variant_img)
            for d in dets:
                all_raw_texts.append(d["text"])
                val = MeterOCR._parse_7seg_text(d["text"])
                if val:
                    score = MeterOCR._score_reading(val, d["confidence"])
                    all_candidates.append({
                        "value":      val,
                        "confidence": d["confidence"],
                        "score":      score,
                        "variant":    variant_name,
                    })

        if not all_candidates:
            return {"value": None, "confidence": 0.0, "candidates": [],
                    "raw_candidates_text": all_raw_texts}

        all_candidates.sort(key=lambda c: -c["score"])
        best = all_candidates[0]
        n_agree = sum(1 for c in all_candidates if c["value"] == best["value"])
        boosted = round(min(1.0, best["confidence"] + 0.05 * (n_agree - 1)), 4)

        return {
            "value":               best["value"],
            "confidence":          boosted,
            "candidates":          all_candidates[:12],
            "raw_candidates_text": all_raw_texts,
        }

    # ------------------------------------------------------------------ #
    # Unit detection                                                       #
    # ------------------------------------------------------------------ #

    def _detect_unit(self, detections: list[dict]) -> dict:
        """Scan all OCR detections for power unit strings."""
        candidates: list[dict] = []
        for det in detections:
            text = det["text"]
            for pattern, canonical in _UNIT_MAP:
                if re.search(re.escape(pattern), text, re.IGNORECASE):
                    candidates.append({
                        "value":      canonical,
                        "raw_text":   text,
                        "confidence": det["confidence"],
                    })
                    break

        if not candidates:
            return {"value": None, "confidence": 0.0, "candidates": []}

        best = max(candidates, key=lambda c: c["confidence"])
        return {"value": best["value"], "confidence": best["confidence"],
                "candidates": candidates}

    @staticmethod
    def _detect_unit_in_reading_strings(raw_texts: list[str]) -> dict:
        """
        Fallback: look for unit text embedded in reading strings,
        e.g. "00523kWh" → unit = "kWh".
        """
        for text in raw_texts:
            for pattern, canonical in _UNIT_MAP:
                if re.search(re.escape(pattern), text, re.IGNORECASE):
                    return {"value": canonical, "confidence": 0.5, "candidates": [
                        {"value": canonical, "raw_text": text, "confidence": 0.5}
                    ]}
        return {"value": None, "confidence": 0.0, "candidates": []}

    # ------------------------------------------------------------------ #
    # OCR runners                                                          #
    # ------------------------------------------------------------------ #

    def _run_ocr_display(self, img: np.ndarray) -> list[dict]:
        """
        Tuned EasyOCR pass for preprocessed display images.
        Low link_threshold (0.2) preserves the decimal dot.
        """
        common = dict(
            detail=1, paragraph=False,
            contrast_ths=0.05, adjust_contrast=0.65,
            text_threshold=0.5, low_text=0.25,
            link_threshold=0.2,
            canvas_size=3200, mag_ratio=2.0,
        )
        raw = self.reader.readtext(img, **common)
        seen: set[str] = set()
        detections: list[dict] = []
        for bbox_pts, text, conf in raw:
            t = text.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            detections.append({
                "text": t, "confidence": round(float(conf), 4),
                "center": (sum(xs) / 4, sum(ys) / 4),
            })
        return detections

    def _run_ocr_full(self, img: np.ndarray) -> list[dict]:
        """
        Standard EasyOCR pass on the original color image.
        Used to find the unit label text.
        """
        raw = self.reader.readtext(
            img, detail=1, paragraph=False,
            contrast_ths=0.1, adjust_contrast=0.5,
            text_threshold=0.5, low_text=0.3,
            link_threshold=0.35,
            canvas_size=2560, mag_ratio=1.5,
        )
        detections: list[dict] = []
        for bbox_pts, text, conf in raw:
            t = text.strip()
            if not t:
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            detections.append({
                "text": t, "confidence": round(float(conf), 4),
                "bbox": bbox_pts,
                "center": (sum(xs) / 4, sum(ys) / 4),
            })
        return detections


# ---------------------------------------------------------------------------
# Batch processing + display
# ---------------------------------------------------------------------------

def process_images(image_paths: list[str], use_gpu: bool = False) -> list[dict]:
    engine = MeterOCR(use_gpu=use_gpu)
    results = []
    for path in image_paths:
        try:
            r = engine.extract(path)
            _print_result(r)
            results.append(r)
        except Exception as exc:
            logger.error("Failed to process %s: %s", path, exc)
            results.append({"image": path, "error": str(exc)})
    return results


def _print_result(result: dict) -> None:
    W = 65
    print("\n" + "=" * W)
    print(f"  Image : {result.get('image', 'N/A')}")
    print("=" * W)

    if "error" in result:
        print(f"  ERROR : {result['error']}")
        return

    label_map = {
        "meter_serial_number": "Meter Serial Number",
        "serial_reading":      "Serial (Reading / LCD)",
        "kWh":                 "kWh",
        "kVAh":                "kVAh",
        "MD_kW":               "MD kW",
        "Demand_kVA":          "Demand kVA",
    }

    print(f"  {'Field':<26}  {'Value':<18}  {'Confidence'}")
    print("  " + "-" * (W - 2))
    for key, label in label_map.items():
        data = result.get("fields", {}).get(key, {})
        value  = data.get("value") or "NOT FOUND"
        conf   = data.get("confidence", 0.0)
        conf_s = f"{conf:.1%}" if conf > 0 else "—"
        print(f"  {label:<26}  {str(value):<18}  {conf_s}")

    raws = result.get("raw_detections", [])
    print(f"\n  Raw OCR detections ({len(raws)}):")
    for d in raws:
        print(f"    [{d['confidence']:.3f}]  {d['text']!r}")
    print("=" * W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_gpu = "--gpu" in sys.argv

    paths = args if args else (
        glob.glob(str(_HERE / "*.jpg"))
        + glob.glob(str(_HERE / "*.jpeg"))
        + glob.glob(str(_HERE / "*.png"))
    )

    if not paths:
        print("Usage: python meter_ocr.py <image1.jpg> [image2.jpg …] [--gpu]")
        sys.exit(1)

    results = process_images(paths, use_gpu=use_gpu)

    out_path = _HERE / "meter_ocr_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
