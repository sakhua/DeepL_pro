# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MediCheck: a CPU-only Streamlit demo for a Deep Learning course project (NHÓM 03, ICTU). It reads a Vietnamese prescription image, checks pill photos against the drugs on that prescription, and flags drug–drug interactions. Models were trained elsewhere (Kaggle notebooks 01–11, not in this folder). This folder only does inference. The UI text, comments, README and dict keys (`thuoc_1`, `muc_do`, `c_don_that`…) are in Vietnamese. Keep new user-facing strings in Vietnamese.

## Commands

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt            # pulls CPU torch via --extra-index-url
pip install --no-deps vietocr==0.3.13      # must be --no-deps (it pins old libs like imgaug)
python setup_models.py [--results DIR] [--force]   # copies weights from ../Results/results_*.zip into models/
python -m streamlit run app.py             # http://localhost:8501; use python -m so the Demo venv's streamlit runs, not another project's
```

There are no tests, linter or build step. To check a change, run the app on the bundled samples in `data/samples/don_*`. Each sample's `meta.json` holds the ground truth, which the app displays next to its predictions.

- `MEDICHECK_MODELS` env var overrides the models directory (default `models/`).
- The first run needs network access to fetch the `vinai/phobert-base` slow tokenizer (`vocab.txt`, `bpe.codes`). If you put those two files in `models/phobert_tokenizer/`, the app runs offline.
- `_tam/` is scratch space and is not used by the app.

## Architecture

There are three pipelines. `app.py` wires them into three tabs, and each pipeline lives in its own module under `medicheck/`:

1. **Prescription** (`pres.py`, `PrescriptionReader.read`): docTR DBNet `db_resnet50` finds text lines (M2). VietOCR reads each crop in batches of 16 (M3). A PhoBERT field classifier labels each line as drugname/quantity/usage/diagnose/date/other (M4). Fuzzy matching against `drug_dict.json` turns drug names into VAIPE class IDs (M5, `text_utils.DrugIndex`). The output `c_don_cand` is the union of candidate class IDs. One drug name can map to several pill classes.
2. **Pill** (`pill.py`, `PillMatcher.analyze`): a single-class YOLO11 model finds the pills (M6). A ConvNeXt-T + ArcFace encoder turns each square crop into an L2-normalized 512-d embedding (M7). Each embedding gets a cosine score against per-class prototypes in `gallery.npy`, restricted to the prescription's candidate classes (M8). The status is set by the threshold τ: `khớp` (match), `tin cậy thấp` (low confidence, inside [τ, τ+margin)), or `ngoài đơn` (not on the prescription). Class `107` in labels means a pill that is not on the prescription.
3. **DDI** (`ddi.py`): `DrugInfo` maps class ID → name, active ingredient and DDInter names, using `data/drug_info.csv`. `DDIChecker` looks up every cross-drug pair of ingredients in `data/ddinter_vaipe.csv` (342 pairs). The `Unknown` level is shown apart from the other levels.

### Things that span multiple files

- **Preprocessing must match the training notebooks exactly**, and the docstrings say which notebook each step mirrors. Examples: the `norm_drug`/`clean_text` rules (notebook 11), `square_crop` margin and crop size from `config.json` (notebooks 01/06), the 1280 px YOLO long side, the reading-order row bucketing (`order_lines`, 12 px), and the PhoBERT input format. That input format is a pyvi-segmented line paired with the *previous* line in reading order, plus 5 positional features `[x1/W, y1/H, x2/W, y2/H, rank/n]`. If you change any of these, the demo results stop matching the reported metrics.
- PhoBERT's `RobertaConfig` is hard-coded in `pres.py`. The fine-tuned weights are all in `phobert_out/field_cls.pt`, so only the tokenizer is downloaded.
- **Threshold handling**: before rendering, `app.py` analyzes *every* pill image with `analyze(..., tau=0.0, low_margin=0.0)`. It then applies the sidebar τ/margin itself in `judge()`, so moving the sliders doesn't re-run the models. The τ that `PillMatcher` loads from `metrics_match.json` is only a default for direct callers. If C_đơn is empty, `PillMatcher` compares against all classes, and the app labels every pill `chưa đối chiếu` instead of reporting a match.
- **Streamlit state**: `st.session_state.pres` is cached per `pres_key` (the sample ID, or an md5 of the whole uploaded file). Changing the prescription clears `pill_res`. `pill_res` is keyed by `(pres_key, "<name>:<md5>", tuple(cand))`. `seen_cls` holds the pill classes confirmed in the current images under the current `cand`. It drives the summary verdict, the drug cards' "đã thấy viên" chips, and tab ③, which narrows each drug's candidate classes to those confirmed classes (or to `c_don_that` when the ground-truth checkbox is on).
- **Layout**: the summary verdict and KPI row are computed before `st.tabs`, so all model work happens up front. Custom HTML/CSS lives in the `CSS` constant. Escape any OCR or user text with `esc()` before it goes into `show_html`. The theme is forced light in `.streamlit/config.toml`, and the hard-coded card colors rely on that.
- `torch.set_grad_enabled(False)` in the constructors only affects the thread that loaded the models, and Streamlit runs each rerun on a different thread. Inference entry points (`PrescriptionReader.read`, `PillMatcher.analyze`) are therefore wrapped in `@torch.inference_mode()`. Any new inference entry point needs the same wrapper.
- Images go through `ImageOps.exif_transpose` both when loaded in `app.py` and in `PrescriptionReader.read`, so phone photos arrive upright.
- `missing_models()` in `app.py` lists the weight files the app requires, and `setup_models.FILES` lists where each comes from in the Kaggle zips. If you add a model artifact, update both.
- Streamlit ≥ 1.50 is assumed (`width="stretch"` on `st.image`/`st.dataframe`).
