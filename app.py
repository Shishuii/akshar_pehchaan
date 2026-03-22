"""
Electric Meter OCR - Gradio UI
================================
VLM pipeline for extracting structured data from electric meter images.
Saves per-image extraction folders with field crops + results.json.

Usage:
    python app.py
    python app.py --share      # public Gradio link
    python app.py --port 7861
"""

import argparse
import json
import logging
import tempfile
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lazy-loaded pipeline instances ────────────────────────────────────────────
_vlm_pipeline    = None
_easyocr_reader  = None

MAX_IMAGE_SIZE   = 1024   # resize before VLM inference for speed
EXTRACTIONS_ROOT = Path("./extractions")

# Fields to attempt cropping (ordered by visual priority)
CROP_FIELDS = [
    ("display_reading", "display_reading"),
    ("serial_number",   "serial_number"),
    ("manufacturer",    "manufacturer"),
    ("display_unit",    "display_unit"),
    ("meter_type",      "meter_type"),
    ("voltage_rating",  "voltage_rating"),
    ("current_rating",  "current_rating"),
    ("md_kw",           "md_kw"),
    ("demand_kva",      "demand_kva"),
    ("kvah_reading",    "kvah_reading"),
]


def _get_vlm(model_key: str):
    global _vlm_pipeline
    from vlm_pipeline import VLMPipeline
    if _vlm_pipeline is None or _vlm_pipeline.model_key != model_key:
        _vlm_pipeline = VLMPipeline(model_key=model_key)
        _vlm_pipeline.load()
    return _vlm_pipeline


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # type: ignore[import-untyped]
        logger.info("Loading EasyOCR reader for bbox detection...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


# ── Image helpers ──────────────────────────────────────────────────────────────

def _resize_if_large(pil_img: Image.Image, max_side: int = MAX_IMAGE_SIZE) -> Image.Image:
    w, h = pil_img.size
    if w <= max_side and h <= max_side:
        return pil_img
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info("Resizing %dx%d → %dx%d for VLM inference", w, h, new_w, new_h)
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def _filepath_to_vlm_tmp(filepath: str) -> str:
    """Load image from Gradio filepath, resize, save to temp JPEG for VLM."""
    pil = Image.open(filepath).convert("RGB")
    pil = _resize_if_large(pil)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil.save(tmp.name, quality=90)
        return tmp.name


# ── Crop saving ────────────────────────────────────────────────────────────────

def _bbox_xyxy(bbox) -> tuple[int, int, int, int]:
    """Convert EasyOCR 4-point bbox → (x1, y1, x2, y2)."""
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _similarity(a: str, b: str) -> float:
    """Fuzzy string similarity: substring bonus + SequenceMatcher ratio."""
    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.85
    return SequenceMatcher(None, a, b).ratio()


def _best_match(value: str, detections: list) -> Optional[tuple]:
    """Return the EasyOCR detection (bbox, text, conf) that best matches value."""
    best_score = 0.0
    best_det   = None
    for bbox, text, conf in detections:
        score = _similarity(value, text)
        if score > best_score:
            best_score = score
            best_det   = (bbox, text, conf)
    return best_det if best_score >= 0.35 else None


def _group_numeric_bbox(detections: list) -> Optional[tuple]:
    """
    For display_reading: merge all numeric/digit detections into one bounding box.
    Returns merged (x1, y1, x2, y2) or None.
    """
    import re
    numeric_dets = [
        (bbox, text, conf)
        for bbox, text, conf in detections
        if re.search(r"\d", text)
    ]
    if not numeric_dets:
        return None
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []
    for bbox, _, _ in numeric_dets:
        x1, y1, x2, y2 = _bbox_xyxy(bbox)
        all_x1.append(x1); all_y1.append(y1)
        all_x2.append(x2); all_y2.append(y2)
    return min(all_x1), min(all_y1), max(all_x2), max(all_y2)


def detect_seven_segment_display(image_path: str) -> Optional[np.ndarray]:
    """
    Detect and return a corrected crop of the seven-segment / LCD display panel.

    Pipeline
    --------
    1. Preprocess — CLAHE contrast enhancement on grayscale.
    2. Candidate generation from FOUR independent masks / edge maps:
         a. Canny edges at multiple thresholds → 4-sided polygon contours
            (perspective-corrected via getPerspectiveTransform when 4 pts found)
         b. Segment colour mask (HSV) → dilated blob contours
         c. Dark-background bezel mask → dark rectangular blobs
         d. High local-contrast map (Laplacian) → edge-rich rectangles
    3. Score every candidate rectangle with five independent signals:
         • Aspect ratio — ideal 3:1 – 7:1 (display is always wide)
         • Edge density — Laplacian variance inside; digits create many edges
         • Dark interior — mean brightness < 80 ≈ black display background
         • Segment colour presence — lit-segment hues inside the region
         • Vertical position bias — display is usually in the upper 65 % of frame
    4. Return the highest-scoring candidate, deskewed with perspective transform
       (4-point quad) or minAreaRect rotation (axis-aligned bbox fallback).
    """
    import cv2  # type: ignore[import-untyped]

    bgr = cv2.imread(image_path)
    if bgr is None:
        return None

    img_h, img_w = bgr.shape[:2]
    img_area     = img_h * img_w

    # ── Preprocessing ─────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Laplacian variance map (used both for scoring and as a candidate source)
    lap     = cv2.Laplacian(enhanced, cv2.CV_64F)
    lap_abs = np.abs(lap).astype(np.float32)

    # ── Segment colour mask ────────────────────────────────────────────────────
    seg_color_ranges = [
        ((35,  50,  50), (90,  255, 255)),   # green
        (( 8,  80,  80), (38,  255, 255)),   # amber / yellow
        (( 0,  80,  80), ( 8,  255, 255)),   # red (low)
        ((165, 80,  80), (180, 255, 255)),   # red (high)
        ((100, 60,  60), (145, 255, 255)),   # blue
        ((  0,  0, 200), (180,  50, 255)),   # white / silver
        (( 75, 50,  50), (100, 255, 255)),   # teal / cyan
        (( 20, 60,  60), ( 35, 255, 255)),   # orange
    ]
    colour_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for lo, hi in seg_color_ranges:
        colour_mask = cv2.bitwise_or(
            colour_mask, cv2.inRange(hsv, np.array(lo), np.array(hi))
        )

    # ── Collect candidates from all sources ────────────────────────────────────
    all_candidates: list[dict] = []

    def _add_from_mask(mask: np.ndarray, source: str,
                       min_frac=0.006, max_frac=0.55,
                       min_asp=2.0, max_asp=12.0) -> None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            frac = area / img_area
            if not (min_frac <= frac <= max_frac):
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            rot    = cv2.minAreaRect(cnt)
            rw, rh = rot[1]
            if rw == 0 or rh == 0:
                continue
            asp = max(rw, rh) / min(rw, rh)
            if not (min_asp <= asp <= max_asp):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            all_candidates.append({
                "approx": approx if len(approx) == 4 else None,
                "rot":    rot,
                "bbox":   (x, y, bw, bh),
                "source": source,
                "score":  0.0,
            })

    # Source A — Canny at multiple thresholds → rectangular outlines
    for t_lo, t_hi in [(20, 60), (40, 120), (60, 180), (80, 240)]:
        edges = cv2.Canny(enhanced, t_lo, t_hi)
        closed_e = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)), iterations=2,
        )
        _add_from_mask(closed_e, f"canny_{t_lo}",
                       min_frac=0.005, max_frac=0.6, min_asp=1.8, max_asp=14.0)

    # Source B — Segment colour blobs
    ck      = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 11))
    col_dil = cv2.morphologyEx(
        cv2.dilate(colour_mask, ck, iterations=4),
        cv2.MORPH_CLOSE, ck, iterations=3,
    )
    _add_from_mask(col_dil, "colour", min_frac=0.004, max_frac=0.55)

    # Source C — Dark bezel (display background is almost black)
    dark_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    dk      = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 18))
    dark_cl = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, dk, iterations=3)
    _add_from_mask(dark_cl, "dark_bezel", min_frac=0.008, max_frac=0.50)

    # Source D — High edge-density regions (Laplacian magnitude > threshold)
    _, edge_map = cv2.threshold(lap_abs, 20, 255, cv2.THRESH_BINARY)
    edge_map    = edge_map.astype(np.uint8)
    ek          = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 12))
    edge_cl     = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, ek, iterations=3)
    _add_from_mask(edge_cl, "edge_density", min_frac=0.005, max_frac=0.55)

    if not all_candidates:
        logger.warning("No display candidates found in %s", image_path)
        return None

    # ── Score every candidate ──────────────────────────────────────────────────
    for c in all_candidates:
        x, y, bw, bh = c["bbox"]
        x1 = max(0, x);          y1 = max(0, y)
        x2 = min(img_w, x + bw); y2 = min(img_h, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue

        roi_gray   = gray[y1:y2, x1:x2].astype(np.float32)
        roi_colour = colour_mask[y1:y2, x1:x2]
        roi_lap    = lap_abs[y1:y2, x1:x2]
        roi_h, roi_w = roi_gray.shape

        # 1. Aspect ratio — peak at 4.5:1, falls off either side
        rot     = c["rot"]
        rw, rh  = rot[1]
        asp     = max(rw, rh) / max(min(rw, rh), 1)
        asp_score = 1.0 - min(abs(asp - 4.5) / 4.5, 1.0)

        # 2. Edge density — Laplacian variance (digits = lots of edges)
        lap_var    = float(np.var(roi_lap)) if roi_lap.size > 0 else 0.0
        edge_score = min(lap_var / 1500.0, 1.0)

        # 3. Dark interior — display background is near-black
        mean_bright = float(np.mean(roi_gray))
        dark_score  = max(0.0, 1.0 - mean_bright / 100.0)

        # 4. Segment colour presence
        col_frac  = float(np.count_nonzero(roi_colour)) / max(roi_h * roi_w, 1)
        col_score = min(col_frac * 5.0, 1.0)

        # 5. Vertical position — display usually in upper 65 % of meter
        cy_frac   = (y + bh / 2) / img_h
        pos_score = 1.0 - max(0.0, cy_frac - 0.65) / 0.35

        c["score"] = (
            1.5 * asp_score +
            2.0 * edge_score +
            1.5 * dark_score +
            1.0 * col_score +
            0.5 * pos_score
        ) / 6.5

    best = max(all_candidates, key=lambda c: c["score"])
    logger.info(
        "Best display candidate: source=%s score=%.3f bbox=%s",
        best["source"], best["score"], best["bbox"],
    )

    # ── Crop / deskew ──────────────────────────────────────────────────────────
    PAD = 16

    # Prefer perspective transform when a clean 4-point polygon is available
    approx = best.get("approx")
    if approx is not None and len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ], dtype=np.float32)
        out_w = int(np.linalg.norm(ordered[1] - ordered[0]) + 2 * PAD)
        out_h = int(np.linalg.norm(ordered[3] - ordered[0]) + 2 * PAD)
        if out_w > 10 and out_h > 10:
            dst = np.array([
                [PAD, PAD], [out_w - PAD, PAD],
                [out_w - PAD, out_h - PAD], [PAD, out_h - PAD],
            ], dtype=np.float32)
            M    = cv2.getPerspectiveTransform(ordered, dst)
            warp = cv2.warpPerspective(bgr, M, (out_w, out_h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
            return cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)

    # Fallback — minAreaRect with safe angle normalisation
    rot      = best["rot"]
    angle    = rot[2]
    center_f = rot[0]
    rw, rh   = rot[1]

    if rw < rh:
        angle += 90.0
        rw, rh = rh, rw
    if angle > 45.0:
        angle -= 90.0

    M        = cv2.getRotationMatrix2D(center_f, angle, 1.0)
    straight = cv2.warpAffine(bgr, M, (img_w, img_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    cx, cy  = int(center_f[0]), int(center_f[1])
    x1 = max(0, cx - int(rw / 2) - PAD)
    y1 = max(0, cy - int(rh / 2) - PAD)
    x2 = min(img_w, cx + int(rw / 2) + PAD)
    y2 = min(img_h, cy + int(rh / 2) + PAD)

    crop_bgr = straight[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def save_field_crops(
    original_filepath: str,
    vlm_fields: dict,
    base_dir: Path = EXTRACTIONS_ROOT,
) -> tuple[Path, list[tuple[np.ndarray, str]]]:
    """
    1. Run EasyOCR on the original (full-res) image to get text bboxes.
    2. Match each VLM field value to the best-matching bbox.
    3. Crop & save each match as <field>.jpg inside a named folder.
    4. Save results.json with all VLM fields.

    Returns (folder_path, gallery_list) where gallery_list is
    [(np.ndarray_crop, caption), ...] for gr.Gallery.
    """
    stem   = Path(original_filepath).stem or "meter"
    folder = base_dir / stem
    folder.mkdir(parents=True, exist_ok=True)

    # Full-res image for cropping
    pil_orig = Image.open(original_filepath).convert("RGB")
    img_arr  = np.array(pil_orig)
    h, w     = img_arr.shape[:2]
    PAD      = 14  # pixels of padding around each crop

    # Run EasyOCR
    reader     = _get_easyocr_reader()
    detections = reader.readtext(original_filepath)   # [[bbox, text, conf], ...]
    logger.info("EasyOCR found %d text regions for crop extraction", len(detections))

    gallery: list[tuple[np.ndarray, str]] = []

    # ── Seven-segment display detection (first, shown at top of gallery) ───────
    display_crop = detect_seven_segment_display(original_filepath)
    if display_crop is not None:
        disp_path = folder / "display_region.jpg"
        Image.fromarray(display_crop).save(str(disp_path), quality=95)
        logger.info("Saved display region crop: %s", disp_path)
        gallery.append((display_crop, "Display Region"))
    else:
        logger.warning("No display region detected; skipping display_region.jpg")

    for fkey, fname in CROP_FIELDS:
        value = vlm_fields.get(fkey)
        if not value:
            continue

        val_str = str(value).strip()

        # Special handling for display reading: group all numeric detections
        if fkey == "display_reading":
            merged = _group_numeric_bbox(detections)
            if merged:
                x1, y1, x2, y2 = merged
            else:
                det = _best_match(val_str, detections)
                if det is None:
                    continue
                x1, y1, x2, y2 = _bbox_xyxy(det[0])
        else:
            det = _best_match(val_str, detections)
            if det is None:
                continue
            x1, y1, x2, y2 = _bbox_xyxy(det[0])

        # Pad and clamp
        x1c = max(0, x1 - PAD)
        y1c = max(0, y1 - PAD)
        x2c = min(w, x2 + PAD)
        y2c = min(h, y2 + PAD)

        crop     = img_arr[y1c:y2c, x1c:x2c]
        out_path = folder / f"{fname}.jpg"
        Image.fromarray(crop).save(str(out_path), quality=95)
        logger.info("Saved crop: %s", out_path)

        gallery.append((crop, fname.replace("_", " ").title()))

    # Save results.json (exclude confidence keys — keep clean)
    results_clean = {
        k: v for k, v in vlm_fields.items()
        if not k.endswith("_confidence")
    }
    results_clean["_model_used"]   = vlm_fields.get("_model", "")
    results_clean["_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    json_path = folder / "results.json"
    json_path.write_text(json.dumps(results_clean, indent=2, default=str))
    logger.info("Saved results JSON: %s", json_path)

    return folder, gallery


# ── Core processing ────────────────────────────────────────────────────────────

def run_vlm(image_path: str, model_key: str) -> dict:
    try:
        vlm    = _get_vlm(model_key)
        result = vlm.extract(image_path)
        return result
    except Exception as e:
        return {"error": str(e), "success": False, "fields": {}}


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_val(val) -> str:
    return str(val) if val is not None else "—"


def format_final_output(result: dict, model_key: str) -> str:
    if result.get("error") and not result.get("success"):
        return f"**ERROR:** {result.get('error')}"

    fields = result.get("fields", {})
    raw    = result.get("raw_response", "")

    def row(label, fkey):
        return f"| **{label}** | `{_fmt_val(fields.get(fkey))}` |"

    lines = [
        f"### Final Output — {model_key}\n",
        "| Field | Value |",
        "|-------|-------|",
        row("Manufacturer",    "manufacturer"),
        row("Serial Number",   "serial_number"),
        row("Display Reading", "display_reading"),
        row("Display Unit",    "display_unit"),
        row("Power Unit",      "power_unit"),
        row("kVAh Reading",    "kvah_reading"),
        row("MD kW",           "md_kw"),
        row("Demand kVA",      "demand_kva"),
        row("Meter Type",      "meter_type"),
        row("Voltage",         "voltage_rating"),
        row("Current Rating",  "current_rating"),
        f"| **Decimal Pos** | `{_fmt_val(fields.get('decimal_point_position'))}` |",
        f"| **Digit Count** | `{_fmt_val(fields.get('digit_count'))}` |",
    ]

    notes = fields.get("notes")
    if notes:
        lines.append(f"\n**Notes:** {notes}")

    if raw:
        lines.append(
            f"\n<details><summary>Raw VLM response</summary>\n\n"
            f"```\n{raw[:1500]}\n```\n</details>"
        )

    return "\n".join(lines)


# ── Main processing function ───────────────────────────────────────────────────

def process_meter_image(
    filepath: Optional[str],
    vlm_model_key: str,
) -> tuple[str, list, str]:
    """
    Returns (vlm_markdown, gallery_list, save_status_markdown).
    """
    empty = (
        "*No image provided. Upload a meter photo to begin.*",
        [],
        "",
    )
    if filepath is None:
        return empty

    # VLM inference on resized copy
    vlm_tmp = _filepath_to_vlm_tmp(filepath)
    logger.info("Running VLM (%s)...", vlm_model_key)
    result = run_vlm(vlm_tmp, vlm_model_key)

    vlm_md = format_final_output(result, vlm_model_key)

    # Crop extraction from original (full-res) image
    fields = result.get("fields", {})
    gallery: list = []
    save_status   = ""

    if fields:
        try:
            folder, gallery = save_field_crops(filepath, fields)
            abs_folder      = folder.resolve()
            save_status     = (
                f"**Saved to** `{abs_folder}`\n\n"
                + "\n".join(
                    f"- `{p.name}`"
                    for p in sorted(folder.iterdir())
                )
            )
        except Exception as exc:
            logger.exception("Crop extraction failed: %s", exc)
            save_status = f"**Crop save failed:** {exc}"

    return vlm_md, gallery, save_status


# ── Gradio UI ──────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ── */
body, .gradio-container { background: #0f1117 !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0d2137 50%, #1a1035 100%);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 16px;
    padding: 28px 36px 20px;
    margin-bottom: 18px;
    text-align: center;
    box-shadow: 0 4px 32px rgba(0,0,0,0.5);
}
.hero h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #63b3ed, #9f7aea, #63b3ed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px !important;
}
.hero .subtitle {
    color: #a0aec0;
    font-size: 0.95rem;
    margin: 0 0 14px;
}
.hero .badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 4px;
}
.badge {
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    color: #63b3ed;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.team-tag {
    display: inline-block;
    background: linear-gradient(90deg, #9f7aea33, #63b3ed33);
    border: 1px solid rgba(159,122,234,0.4);
    border-radius: 20px;
    padding: 4px 18px;
    font-size: 0.82rem;
    color: #c4b5fd;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 8px;
}

/* ── Cards ── */
.card {
    background: #161b2e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.35);
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #718096 !important;
    margin: 0 0 8px !important;
}

/* ── Run button glow ── */
.run-btn { box-shadow: 0 0 18px rgba(99,179,237,0.35) !important; }

/* ── Output panel ── */
.output-panel {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.88rem !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #4a5568;
    font-size: 0.75rem;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
"""


def build_ui() -> gr.Blocks:
    from vlm_pipeline import SUPPORTED_MODELS

    with gr.Blocks(
        title="Akshar Pehchaan — Electric Meter OCR",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    ) as demo:

        # ── Hero ──────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="hero">
          <h1>Akshar Pehchaan</h1>
          <p class="subtitle">Vision-Language Model pipeline for intelligent electric meter reading</p>
          <div class="badges">
            <span class="badge">Manufacturer</span>
            <span class="badge">Serial Number</span>
            <span class="badge">Display Reading</span>
            <span class="badge">Power Unit</span>
            <span class="badge">kVAh / MD kW</span>
            <span class="badge">Meter Type</span>
          </div>
          <div class="team-tag">Team: Akshar_Pehchaan</div>
        </div>
        """)

        # ── Main row ──────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left: input panel ──────────────────────────────────────────
            with gr.Column(scale=4, min_width=300):
                with gr.Group(elem_classes="card"):
                    gr.Markdown("**METER IMAGE**", elem_classes="section-label")
                    image_input = gr.Image(
                        label="",
                        type="filepath",
                        height=320,
                        sources=["upload", "clipboard"],
                        show_label=False,
                    )

                with gr.Group(elem_classes="card"):
                    gr.Markdown("**VLM MODEL**", elem_classes="section-label")
                    vlm_choice = gr.Dropdown(
                        label="",
                        choices=list(SUPPORTED_MODELS.keys()),
                        value="Qwen2-VL-2B",
                        show_label=False,
                        info="MiniCPM-V-2_6-int4 = best accuracy · SmolVLM-256M = lightest",
                    )

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                    run_btn   = gr.Button(
                        "Run OCR", variant="primary", scale=3,
                        elem_classes="run-btn",
                    )

                gr.HTML("""
                <div style="color:#718096;font-size:0.75rem;margin-top:10px;line-height:1.6;">
                  First run loads the model (~30–60 s).<br>
                  Images &gt;1024 px are auto-resized for speed.<br>
                  Crops are extracted from the full-resolution original.
                </div>
                """)

            # ── Right: results panel ───────────────────────────────────────
            with gr.Column(scale=8):

                # Top half: extracted fields table
                with gr.Group(elem_classes="card"):
                    gr.Markdown("**EXTRACTED FIELDS**", elem_classes="section-label")
                    final_output = gr.Markdown(
                        "*Upload a meter image and click **Run OCR**.*",
                        elem_classes="output-panel",
                    )

                gr.HTML("<div style='height:14px'></div>")

                # Bottom half: detected region crops
                with gr.Group(elem_classes="card"):
                    gr.Markdown("**DETECTED REGION CROPS**", elem_classes="section-label")
                    crop_gallery = gr.Gallery(
                        label="",
                        show_label=False,
                        columns=4,
                        rows=2,
                        height=280,
                        object_fit="cover",
                        preview=True,
                    )
                    save_status = gr.Markdown(
                        "*Cropped regions will appear here after running OCR.*",
                        elem_classes="output-panel",
                    )

        # ── Example images ─────────────────────────────────────────────────────
        example_paths = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
        if example_paths:
            with gr.Group(elem_classes="card"):
                gr.Markdown("**EXAMPLE IMAGES**", elem_classes="section-label")
                gr.Examples(
                    examples=[[str(p)] for p in example_paths[:6]],
                    inputs=image_input,
                    label="",
                )

        # ── Model info ─────────────────────────────────────────────────────────
        with gr.Accordion("Model Specifications", open=False):
            model_rows = "\n".join(
                f"| {k} | {v.get('vram_fp16','—')} | {v.get('vram_int4','—')} "
                f"| {v.get('docvqa','—')} | {v.get('ocrbench','—')} "
                f"| {v.get('description','').split('.')[0]} |"
                for k, v in SUPPORTED_MODELS.items()
            )
            gr.Markdown(f"""
| Model | fp16 VRAM | int4 VRAM | DocVQA | OCRBench | Notes |
|-------|-----------|-----------|--------|----------|-------|
{model_rows}
""")

        # ── Footer ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer">
          Built by <strong style="color:#c4b5fd;">Team Akshar_Pehchaan</strong>
          &nbsp;·&nbsp; VLM-powered electric meter OCR
        </div>
        """)

        # ── Events ─────────────────────────────────────────────────────────────
        _blank = "*Upload a meter image and click **Run OCR**.*"
        run_btn.click(
            fn=process_meter_image,
            inputs=[image_input, vlm_choice],
            outputs=[final_output, crop_gallery, save_status],
        )
        clear_btn.click(
            fn=lambda: (None, _blank, [], ""),
            outputs=[image_input, final_output, crop_gallery, save_status],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Electric Meter OCR UI")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
