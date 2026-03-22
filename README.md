# Akshar Pehchaan — Electric Meter OCR

> Vision-Language Model pipeline for intelligent, automated electric meter reading.
> Built by **Team Akshar_Pehchaan** for the Instinct Hackathon.

---

## What it does

Upload a photo of an electric utility meter and the system extracts:

| Field | Example |
|-------|---------|
| Manufacturer | Genus, Landis+Gyr, Elster … |
| Serial Number | GE7422324 |
| Display Reading | 00523.40 |
| Display Unit | kWh / kVAh |
| Meter Type | Single phase / Three phase |
| Voltage Rating | 230 V, 415 V |
| Current Rating | 5–30 A |
| MD kW / Demand kVA | — |
| kVAh Reading | — |

Results are saved as structured JSON **plus individual image crops** for each detected field, in a per-image folder under `./extractions/`.

---

## Architecture

```
app.py                  ← Gradio web UI + orchestration
├── vlm_pipeline.py     ← 11 Vision-Language Models (Qwen2-VL, MiniCPM, Florence-2 …)
├── meter_ocr.py        ← EasyOCR pipeline (labels, serial, kWh/kVAh)
├── trocr_pipeline.py   ← TrOCR specialized digit reader (7-segment displays)
└── download_models.py  ← Batch model downloader
```

**Display detection** uses a 4-source fusion (Canny edges · colour HSV mask · dark-background bezel · Laplacian edge density) with 5-factor scoring (aspect ratio · edge density · dark interior · segment colour presence · vertical position), followed by perspective deskewing.

---

## Supported VLM Models

Ranked by OCR accuracy on RTX 2080 Ti (10.76 GB VRAM):

| # | Model | fp16 VRAM | int4 VRAM | DocVQA | OCRBench |
|---|-------|-----------|-----------|--------|----------|
| 1 | MiniCPM-V-2_6-int4 | 16 GB | **~7 GB** | ~92 % | ~89 % |
| 2 | Qwen2.5-VL-3B | ~7 GB | ~3.5 GB | ~85 % | ~81 % |
| 3 | InternVL2-4B | ~10 GB | ~2.5 GB | ~91 % | ~82 % |
| 4 | InternVL2-2B | ~5 GB | ~1.5 GB | 94.1 % | ~75 % |
| 5 | PaliGemma2-3B-448 | ~8 GB | ~3 GB | ~80 % | ~78 % |
| **6** | **Qwen2-VL-2B** *(default)* | **~4.5 GB** | **~2.5 GB** | **~80 %** | **~72 %** |
| 7 | Florence-2-large | ~1.5 GB | < 1 GB | — | — |
| 8 | Florence-2-base | ~0.7 GB | < 0.7 GB | — | — |
| 9 | Moondream2 | ~3.6 GB | ~2 GB | 79.3 % | 61.2 % |
| 10 | SmolVLM-500M | ~1 GB | < 1 GB | ~67 % | ~55 % |
| 11 | SmolVLM-256M | ~0.5 GB | < 0.5 GB | 58.3 % | ~50 % |

All models are loaded lazily on first use and cached locally in `./models/`.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/<your-org>/akshar-pehchaan.git
cd akshar-pehchaan
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **CUDA users** — for faster inference install the optional extras:
> ```bash
> pip install bitsandbytes>=0.41.0   # 4-bit quantisation (~50 % memory saving)
> pip install flash-attn              # FlashAttention-2 (compile from source)
> ```

### 4. Download models

```bash
# Download just the default model (Qwen2-VL-2B, ~4.5 GB)
python download_models.py --model Qwen2-VL-2B

# Download the best accuracy model (MiniCPM-V-2_6-int4, ~7 GB)
python download_models.py --model MiniCPM-V-2_6-int4

# Download lightweight models only
python download_models.py --skip-large

# Download all 11 VLMs + TrOCR (30+ GB total)
python download_models.py
```

### 5. Run the web UI

```bash
python app.py                   # http://localhost:7860
python app.py --share           # Public Gradio link
python app.py --port 7861       # Custom port
python app.py --host 0.0.0.0    # Bind to all interfaces
```

Open your browser at `http://localhost:7860`, upload a meter photo, pick a model, and click **Run OCR**.

---

## Output

Every run creates an extraction folder:

```
extractions/
└── meter_photo/
    ├── results.json          ← All extracted fields + metadata
    ├── display_region.jpg    ← Detected & deskewed 7-segment display
    ├── display_reading.jpg   ← Crop of the reading digits
    ├── serial_number.jpg     ← Crop of the serial number label
    ├── manufacturer.jpg      ← Crop of the manufacturer name
    ├── display_unit.jpg
    ├── meter_type.jpg
    ├── voltage_rating.jpg
    └── current_rating.jpg
```

`results.json` example:

```json
{
  "manufacturer": "Genus",
  "serial_number": "GE7422324",
  "display_reading": "00025.40",
  "display_unit": "kWh",
  "power_unit": "kWh",
  "meter_type": "single phase",
  "voltage_rating": "230V",
  "current_rating": "5-30A",
  "decimal_point_position": 5,
  "digit_count": 8,
  "notes": "Clear image, high confidence on all fields.",
  "_model_used": "Qwen2-VL-2B",
  "_extracted_at": "2026-03-22T14:30:00"
}
```

---

## Project Structure

```
akshar-pehchaan/
├── app.py                  # Gradio UI, display detection, crop saving
├── vlm_pipeline.py         # VLM inference (11 models, prompt engineering)
├── meter_ocr.py            # EasyOCR extraction pipeline
├── trocr_pipeline.py       # TrOCR 7-segment digit reader
├── download_models.py      # Model downloader
├── requirements.txt        # Python dependencies
├── demo_images/            # Sample meter images
├── extractions/            # Auto-created: per-image output folders
├── models/                 # Auto-created: HuggingFace model cache
└── colab_demo/
    └── meter_ocr_colab.ipynb   # Google Colab demo notebook
```

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| VRAM | 0 GB (CPU) | 6–10 GB (CUDA) |
| Storage | 5 GB | 40 GB (all models) |
| OS | Linux / macOS / Windows | Ubuntu 22.04 |

CPU-only mode works but is slow (~1–3 min per image). GPU with 6+ GB VRAM is recommended for a responsive demo.

---

## Google Colab

A ready-to-run Colab notebook is available in [`colab_demo/meter_ocr_colab.ipynb`](colab_demo/meter_ocr_colab.ipynb). It installs dependencies, downloads a lightweight model, and runs inference on a sample image — no local setup required.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Switch to a smaller model (SmolVLM-500M) or enable int4 via bitsandbytes |
| `easyocr` import error | `pip install easyocr` — it downloads its own weights on first run |
| First run is slow | Models are downloaded and cached on first use; subsequent runs are fast |
| Display crop is wrong | The scoring pipeline tries 4 strategies; try a higher-res or less-blurry photo |
| `trust_remote_code` warning | Expected for MiniCPM and InternVL — these models require it |

---

## Team

**Akshar_Pehchaan** — Instinct Hackathon

---

## License

This project is released for hackathon and research purposes.
