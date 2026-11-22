# OCR research tools

[Project](../README.md) · [Repository structure](../docs/repository-structure.md) · [Methodology and results](../docs/experiment-results.md) · [Reproduction guide](../docs/research-guide.md)

I keep experiments separate from the desktop interface in `manga_app/`. Run the commands below from the repository root, with the relevant research environment activated.

## Setup and baseline example

```powershell
python -m pip install -r research/requirements.txt
python -m research.prepare_dataset
python -m research.run_mangaocr_batch --limit 3
python -m research.evaluate_results results_mangaocr.csv --output report.csv
```

Baseline dependencies include application dependencies. Tesseract needs its executable and Japanese language data installed separately. MangaLMM requires a separate CUDA/PyTorch/Transformers environment; the baseline requirements do not fully specify it.

## Tool index

| Module (`python -m research.<name>`) | Purpose |
| --- | --- |
| `prepare_dataset` | Generate text crops and reference CSV from Hamlet XML |
| `prepare_chatgpt_50` | Select a fixed crop sample for the ChatGPT comparison |
| `prepare_official_mangaocr_test` | Prepare the official full-page evaluation sample |
| `run_mangaocr_batch` | Recognize crops with Manga OCR |
| `run_paddle_batch` | Recognize crops with PaddleOCR |
| `run_tesseract_batch` | Recognize crops with Tesseract |
| `run_mangallmm_fullpages` | Run MangaLMM on full pages |
| `run_mangallmm_tiled` | Run tiled MangaLMM inference |
| `run_official_mangaocr_mangallmm` | Run MangaLMM on the official sample |
| `evaluate_results` | Calculate baseline CER, Exact Match, and runtime |
| `compare_ocr_test50` | Compare four systems on a shared crop sample |
| `evaluate_mangallmm_batch` | Evaluate MangaLMM page predictions |
| `evaluate_mangaocr_officialstyle50` | Evaluate official-sample detections and transcriptions |
| `evaluate_mangaocr_scaled50` | Evaluate after correcting resized-image coordinates |

`batch_runner.py` contains the shared baseline implementation. `benchmark_paths.py` defines default and external data locations. Moving these scripts does not change where data and outputs are stored; see the [reproduction guide](../docs/research-guide.md#run-a-baseline-experiment).
