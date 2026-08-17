"""OCR stage for VisGuard-Ur.

The primary path uses Tesseract with Urdu + English language data. The optional
oracle fallback is disabled by default and exists only for controlled debugging;
reported evaluation metrics always record whether OCR or fallback text was used.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "visguard_ur_dataset.csv"
OUTPUT_PATH = ROOT / "results" / "ocr_results.csv"


def extract_ocr(image_path: Path, psm: int = 6) -> tuple[str, float, bool]:
    if shutil.which("tesseract") is None:
        return "", 0.0, False
    image = Image.open(image_path).convert("RGB")
    # Upscaling improves small-font recognition without changing the benchmark image.
    image = image.resize((image.width * 2, image.height * 2))
    image = ImageOps.autocontrast(image)
    config = f"--psm {psm}"
    try:
        data = pytesseract.image_to_data(image, lang="urd+eng", config=config, output_type=pytesseract.Output.DICT)
    except (pytesseract.TesseractError, FileNotFoundError):
        return "", 0.0, False
    words = []
    confidences = []
    for word, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        word = (word or "").strip()
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            conf = -1
        if word and conf >= 0:
            words.append(word)
            confidences.append(conf)
    text = " ".join(words).strip()
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, round(mean_conf, 3), True


def run_ocr(allow_oracle_fallback: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    records = []
    for row in df.to_dict("records"):
        image_path = ROOT / row["image_path"]
        ocr_text, confidence, available = extract_ocr(image_path)
        used_fallback = False
        if allow_oracle_fallback and not ocr_text:
            # This is only for debugging detector plumbing, never for scientific metrics.
            ocr_text = row["text"]
            used_fallback = True
        records.append({
            **row,
            "ocr_text": ocr_text,
            "ocr_confidence": confidence,
            "ocr_available": available,
            "used_oracle_fallback": used_fallback,
            "ocr_nonempty": bool(ocr_text.strip()),
        })
    out = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    summary = {
        "n_images": len(out),
        "ocr_available": bool(out["ocr_available"].all()),
        "nonempty_rate": round(float(out["ocr_nonempty"].mean()), 4),
        "mean_confidence_nonempty": round(float(out.loc[out["ocr_nonempty"], "ocr_confidence"].mean()), 3) if out["ocr_nonempty"].any() else 0.0,
        "oracle_fallback_rows": int(out["used_oracle_fallback"].sum()),
    }
    (ROOT / "results" / "ocr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-oracle-fallback", action="store_true")
    args = parser.parse_args()
    out = run_ocr(args.allow_oracle_fallback)
    print(f"Saved OCR results: {OUTPUT_PATH}")
    print(f"Non-empty OCR rate: {out['ocr_nonempty'].mean():.1%}")
    print(f"Mean confidence: {out.loc[out['ocr_nonempty'], 'ocr_confidence'].mean() if out['ocr_nonempty'].any() else 0:.1f}")


if __name__ == "__main__":
    main()
