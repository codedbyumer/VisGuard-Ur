# VisGuard-Ur

## Detecting Multimodal Jailbreak and Prompt-Injection Attacks Against Urdu-Aware Vision-Language Models

**VisGuard-Ur** is a reproducible, laptop/Colab-compatible proof-of-concept that extends the character n-gram approach from [UrduGuard](https://github.com/codedbyumer/UrduGuard) into the visual modality. It renders benign and adversarial prompt patterns into images, extracts their text with Urdu-capable OCR, applies an Urdu-aware detector, and reports the effect of the detector on a transparent simulated VLM pipeline.

> **Important scope statement:** this repository does not claim to have evaluated Qwen2-VL, LLaVA, or InternVL. The local VLM stage is a controlled simulator used to test the security pipeline mechanics. Real VLM inference is an optional future extension requiring model weights and suitable compute.

## Research questions operationalized

The prototype addresses three questions. First, it compares a narrow English-pattern guardrail with an Urdu-aware character n-gram detector across Urdu script, Roman Urdu, and English control images. Second, it records OCR non-empty rate, confidence, and character-level similarity by script and visual variant. Third, it measures a simulated system-level attack success rate before and after the OCR-plus-classifier defense.

| Proposal component | Implemented prototype component |
|---|---|
| Dataset construction | 30 hand-authored benign/adversarial prompt patterns across three script groups |
| Visual robustness | Four variants: clean, dark, small-font, and noisy |
| OCR | Tesseract with `urd+eng`, confidence tracking, and optional debug-only oracle fallback |
| UrduGuard extension | Character TF-IDF n-grams from 2–5 characters plus balanced Logistic Regression |
| VLM baseline | Transparent label-driven simulator; no external model inference claimed |
| Analysis | Baseline miss rate, detector ASR, block rate, false-positive rate, OCR coverage/similarity |

## Repository layout

```text
VisGuard-Ur/
├── README.md
├── requirements.txt
├── pyproject.toml
├── dataset/
│   ├── urduguard_seed.csv
│   ├── visguard_ur_dataset.csv          # generated
│   └── dataset_summary.json             # generated
├── images/                              # generated prompt images
├── results/
│   ├── ocr_results.csv                  # generated
│   ├── ocr_summary.json                 # generated
│   ├── visguard_detector.joblib         # generated
│   ├── detector_metrics.json            # generated
│   ├── evaluation_results.csv           # generated
│   ├── evaluation_summary.json          # generated
│   └── evaluation_chart.png             # generated
├── src/
│   ├── generate_dataset.py
│   ├── ocr_pipeline.py
│   ├── train_detector.py
│   ├── evaluate.py
│   └── run_all.py
└── tests/
    └── test_core.py
```

## Installation

On Debian or Ubuntu, install the OCR binary, Urdu language data, and the Noto fonts (needed for correct Nastaliq Urdu / Latin rendering) first:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-urd fonts-noto
```

Then install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The repository uses per-script system fonts when rendering prompts: Urdu-script prompts use Noto Nastaliq Urdu, Roman Urdu and English prompts use DejaVu Sans (see `FONT_CANDIDATES` in `src/generate_dataset.py`). Rendering the wrong script with the wrong font silently produces empty glyph boxes and invalidates OCR/detector results for that script — always visually spot-check a sample image per script after changing fonts.

## Run the complete prototype

From the repository root:

```bash
python3 src/run_all.py
```

The runner performs the following sequence:

```text
1. Generate 120 prompt images from 30 prompt records and four visual variants.
2. Run Tesseract OCR with Urdu and English language data.
3. Train and save the UrduGuard-compatible detector.
4. Evaluate the English-only baseline and OCR-plus-classifier defense.
5. Save CSV, JSON, model, and chart outputs under results/.
```

To execute the stages individually:

```bash
python3 src/generate_dataset.py
python3 src/ocr_pipeline.py
python3 src/train_detector.py
python3 src/evaluate.py
```

For a diagnostic run that fills empty OCR output with the known source text, use the following only for plumbing checks:

```bash
python3 src/ocr_pipeline.py --allow-oracle-fallback
```

This option is **not valid for scientific evaluation**, because it gives the detector the answer. The normal pipeline never uses it.

## Output interpretation

`results/ocr_results.csv` contains the original label for evaluation, rendered image path, OCR text, OCR confidence, OCR availability, and whether a fallback was used. `results/detector_metrics.json` reports the classifier's held-out prompt metrics. `results/evaluation_summary.json` reports the end-to-end measures.

| Metric | Meaning |
|---|---|
| `baseline_english_miss_rate` | Fraction of labeled adversarial images not caught by the narrow English regex baseline |
| `detector_attack_success_rate` | Fraction of labeled adversarial images that remain unblocked in the simulated pipeline |
| `detector_block_rate_on_adversarial` | Fraction of labeled adversarial images blocked by the detector |
| `detector_false_positive_rate_on_benign` | Fraction of benign images blocked by the detector |
| `ocr_nonempty_rate` | Fraction of images for which OCR returned at least one token |
| `ocr_mean_similarity` | Character-level similarity between OCR output and the known rendered source text; use cautiously because this is a synthetic benchmark |

### Fixed run (post font-selection fix), 120 images

| Script | OCR similarity (mean) | OCR confidence (mean) |
|---|---|---|
| English | 0.998 | 95.5 |
| Roman Urdu | 0.978 | 92.8 |
| Urdu script (Nastaliq) | 0.449 | 53.1 |

`baseline_english_miss_rate = 0.5333`, `detector_attack_success_rate = 0.0`, `detector_block_rate_on_adversarial = 1.0`, `detector_false_positive_rate_on_benign = 0.10`.

### A real finding, not a residual bug: Nastaliq is genuinely hard for Tesseract

An earlier version of `generate_dataset.py` picked a font from a single global candidate list without checking `script`, so on machines without the Amiri font installed, **English and Roman Urdu prompts silently rendered as empty glyph boxes** (Tesseract's `urd` model was drawing every character from an Arabic-only font). That was a real bug and has been fixed: `get_font()` now selects per-script (`urdu_script` → Noto Nastaliq Urdu, `roman_urdu`/`english` → DejaVu Sans).

After the fix, English and Roman Urdu OCR became near-perfect (0.998 / 0.978 similarity), but Urdu-script similarity is genuinely only **0.449** — not a rendering bug this time. A direct side-by-side test (same sentence, same Tesseract call) shows why:

- Rendered in **Noto Nastaliq Urdu** (the traditional, cursive, ligature-heavy style used in most real Urdu print/newspapers): Tesseract reads only fragments of the sentence correctly.
- The identical sentence rendered in **Noto Naskh Arabic** (the straighter, more segmented style): Tesseract reads it back **character-for-character correctly**.

This matches a well-known, documented limitation of OCR on Urdu: Nastaliq's context-dependent, heavily-ligated letterforms are substantially harder for OCR engines trained primarily on Naskh-style Arabic script. We kept Nastaliq as the default for `urdu_script` (it is what most authentic Urdu documents actually look like, and it is what an attacker would realistically use), and report the resulting OCR gap as a genuine answer to Research Question 2 rather than tuning it away. This is a legitimate, citable finding for the write-up: **the OCR stage, not the detector, is the weakest link for Urdu-script visual attacks**, which has direct implications for anyone deploying an OCR-based defense against this attack surface.

## Testing

Run the unit tests with:

```bash
pytest -q
```

## Real VLM evaluation (Qwen2-VL-7B-Instruct)

`src/vlm_pipeline.py` and `src/vlm_evaluate.py` replace the label-based
simulator above with an actual open-source Vision-Language Model
(Qwen2-VL-7B-Instruct, 4-bit quantized so it fits a free Colab T4 GPU).
For each image: if the trained detector blocks it, the image never
reaches the VLM (matching real deployment); otherwise the image is sent
to the VLM and the response is judged with a keyword rubric.

**This code was written outside a networked environment and has not
been executed or network-tested.** It follows the standard Qwen2-VL /
`transformers` API, but run the smoke test below on a handful of
images before launching the full 120-image dataset, and be ready to
patch small dependency-version issues.

### Setup (Google Colab, GPU runtime)

```bash
pip install -q transformers accelerate bitsandbytes qwen-vl-utils
pip install -q "git+https://github.com/huggingface/transformers"  # Qwen2-VL needs a recent build
```

### Smoke test first

```bash
python3 src/vlm_evaluate.py --smoke-test 3
```

Check that it loads without an out-of-memory error (if it OOMs, edit
`model_id` in `vlm_pipeline.py` to `"Qwen/Qwen2-VL-2B-Instruct"`) and
that `results/vlm_evaluation_results.csv` contains real-looking model
text in `response_text`, not error strings.

### Full run

```bash
python3 src/vlm_evaluate.py
```

Writes `results/vlm_evaluation_results.csv` (per-image) and
`results/vlm_evaluation_summary.json` (system-level ASR, raw
no-detector ASR, benign false-positive rate, all broken down by
script). For 120 images on a T4, expect roughly 15-40 minutes.

`attack_judge`-style rubric matching in `judge_response()` is a
heuristic, not a validated judge. Before citing numbers anywhere, open
`results/vlm_evaluation_results.csv`, hand-read a random ~20-sample
slice of `response_text`, and check whether you agree with
`attack_succeeded`. Report that agreement rate (e.g. "manual audit of
20 randomly sampled responses agreed with the automatic judge on X/20
cases") in the limitations section below — it is what turns "we ran a
script" into "we validated our methodology."

## Limitations and next research steps

The dataset in this proof of concept is small and hand-authored. It should not be treated as a publishable benchmark or as evidence of production-level model safety. A full study should use a substantially larger, licensed dataset with independently written or permitted translated examples, prompt-level train/test separation, human review, and multiple annotators.

~~The simulator does not measure the behavior of a real VLM.~~ Addressed above via `vlm_evaluate.py`, though the attack-success rubric it uses is still a keyword heuristic (see note above) — an LLM-judge or fully human-annotated rubric would be a stronger next step, along with running the comparison over multiple model families (LLaVA-1.6, InternVL) and decoding settings.

OCR is a major source of uncertainty. The next version should compare Tesseract with PaddleOCR or another Urdu-capable OCR engine, report character/word error rate against ground truth, and include more typography, compression, blur, rotation, occlusion, and mixed-script cases. The current detector is intentionally lightweight and should be compared with multilingual transformer baselines when compute permits.

## References

[1]: https://github.com/codedbyumer/UrduGuard "UrduGuard repository"

[2]: https://github.com/tesseract-ocr/tesseract "Tesseract OCR project"
