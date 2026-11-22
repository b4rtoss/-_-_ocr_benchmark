# Research overview and reproduction

[Back to the project](../README.md)

## Research

To assess Japanese manga OCR approaches, I separately compare Tesseract, PaddleOCR, Manga OCR, ChatGPT, and MangaLMM. The research covers dataset preparation, shared evaluation metrics, and error analysis, informing future improvements to the application.

The [research report](experiment-results.md) contains the full methodology and results. The [verification report](verification.md) covers runtime checks of the application.

### Research implementation

- Generate text-region images from XML annotations and save reference transcriptions to CSV.
- Run batch OCR using Tesseract, PaddleOCR, and Manga OCR.
- Run MangaLMM on full pages and tiles.
- Match predictions to references, including a separate comparison with ChatGPT on 50 examples.
- Calculate recognition quality, recorded processing time, and empty predictions.
- Export results for error analysis.

### My hardware

I run the experiments locally on this laptop:

| Component | Specification |
| --- | --- |
| Platform | ARDOR GAMING, NP5x_60_7x_RNx1 |
| Operating system | Windows 10 Home, x64, build 19045 |
| Processor | Intel Core i5-12450H, 8 cores and 12 threads |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Dedicated VRAM | 6 GB |

VRAM limits the MangaLMM configuration: I use 4-bit quantization and a reduced input-image pixel budget for local inference.

### OCR systems and selection rationale

I compare systems with different specializations and tasks:

| System | Description | Why I include it |
| --- | --- | --- |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | General-purpose OCR engine | Baseline for comparison with manga-specific recognition |
| [PaddleOCR](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) | Text detection and recognition pipeline | Ready-to-use neural OCR comparison and, in the application, text and coordinates from a whole page |
| [Manga OCR](https://github.com/kha-white/manga-ocr) | Japanese text recognizer focused on manga | Test the benefit of domain specialization on prepared crops |
| ChatGPT | Service providing general-purpose multimodal models | Transcribe the same 50 crops and examine the influence of language context |
| [MangaLMM](https://github.com/manga109/MangaLMM) | Manga-specific multimodal model | Separate experiment on text detection, coordinates, and recognition on full pages |

I currently use PaddleOCR in Manga Reader because one pipeline returns text, coordinates, and confidence. Manga OCR performed best among the three baseline systems on 3116 crops, but it needs a separate detector for full-page processing. I plan to evaluate that combination during further development.

Detailed descriptions and limitations are in the [research report](experiment-results.md#ocr-systems-and-selection-rationale).

### Comparison methodology

I separate the 3116-crop baseline, the four-system comparison on the same 50 crops with seed 42, and MangaLMM evaluation on 50 full images from the official MangaOCR test split.

For crops, I apply NFKC normalization and remove whitespace from references and predictions, calculate corpus-level CER and Exact Match, and check file correspondence and missing results.

For MangaLMM, I scale boxes back to the original image and match regions one to one at IoU ≥ 0.5. I calculate detection metrics and end-to-end scores incorporating text similarity. The scaled evaluator removes whitespace from text but does not apply NFKC.

Formulas, sample selection, empty-answer handling, matching, and timing boundaries are specified in the [comparison methodology](experiment-results.md#comparison-methodology).

### Quality metrics

| Metric | Meaning |
| --- | --- |
| Character Error Rate (CER) | Total Levenshtein distance divided by total reference characters; lower is better |
| Exact Match | Fraction of examples matching the reference completely after normalization |
| Empty predictions | Number of empty recognized strings |
| Average inference time | Average recorded call duration; model loading is measured separately from individual recognition calls |

I compare systems on the same sample with the same normalization. Timing depends on hardware and the runner. Failed baseline calls remain in CER and Exact Match, but `evaluate_results.py` averages latency only across successful calls. Models are initialized before calls are timed.

### Research results

On 3116 Hamlet crops, Manga OCR achieved **9.13% CER** and **70.64% Exact Match**. Tesseract achieved 97.74% and 2.54%, and PaddleOCR 81.45% and 4.91%, respectively.

On the separate fixed sample of 50 crops:

| System | CER ↓ | Exact Match ↑ |
| --- | ---: | ---: |
| Tesseract | 97.92% | 4% |
| PaddleOCR | 80.76% | 12% |
| Manga OCR | 7.45% | 76% |
| ChatGPT | 6.24% | 80% |

ChatGPT matched 40 examples exactly, compared with 38 for Manga OCR. I treat this difference as preliminary. Recalculation requires the original ChatGPT responses and the corresponding run settings to be added to the repository.

MangaLMM is evaluated separately on full pages. After coordinate scaling was corrected, the 50-image official MangaOCR test sample yielded **3.87% Detection Hmean** and **0.86% End-to-end Hmean** with NED similarity. The local run used the RTX 4050, 4-bit quantization, and a reduced pixel budget. These scores assess localization and recognition together and are separate from crop CER.

### Main research files

| File | Purpose |
| --- | --- |
| [prepare_dataset.py](../prepare_dataset.py) | Generate crops and `ground_truth.csv` from `Hamlet/Hamlet.xml` |
| [run_mangaocr_batch.py](../run_mangaocr_batch.py) | Batch Manga OCR with per-example results |
| [run_tesseract_batch.py](../run_tesseract_batch.py) | Batch Tesseract OCR |
| [run_paddle_batch.py](../run_paddle_batch.py) | Batch PaddleOCR |
| [evaluate_results.py](../evaluate_results.py) | Baseline CER, Exact Match, and timing evaluation |
| [prepare_chatgpt_50.py](../prepare_chatgpt_50.py) | Prepare a sample for the ChatGPT comparison |
| [compare_ocr_test50.py](../compare_ocr_test50.py) | Compare four systems on the same sample |
| [run_mangallmm_fullpages.py](../run_mangallmm_fullpages.py) | MangaLMM full-page inference |
| [run_mangallmm_tiled.py](../run_mangallmm_tiled.py) | MangaLMM tiled inference |

### Run a baseline experiment

The following starts with Manga OCR; the other engines require their own setup.

1. Clone the repository and open its directory.
2. Create an environment and install benchmark dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-benchmark.txt
   ```

3. Ensure that the images and `Hamlet.xml` are in `Hamlet`. Follow the rightsholders' terms when using the source data.
4. Run `python prepare_dataset.py`.
5. By default, data and results use the repository directory. Set `OCR_BENCHMARK_DIR` or pass `--dataset-dir`, `--ground-truth`, and `--output` to use another location.
6. Run `python run_mangaocr_batch.py`.

Results contain `file`, `ground_truth`, `prediction`, `time_sec`, and `status`. The first run may need model downloads. Add `--limit 3` for a smoke test or `--resume` to continue. Resume keeps existing rows, including errors; without it, the results file is recreated. Failed samples produce a nonzero exit code and retain details in `status`.

```powershell
python run_paddle_batch.py --limit 3
python run_tesseract_batch.py --limit 3 --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
python evaluate_results.py results_mangaocr.csv --output report.csv
python prepare_chatgpt_50.py --size 50 --seed 42
python -m unittest discover -s tests -v
```

Install the Tesseract executable and `jpn` language data separately. The runner uses PATH unless `--tesseract-cmd` is provided. PaddleOCR defaults to CPU; `--device gpu:0` requires a compatible PaddlePaddle GPU/CUDA/cuDNN installation.

`evaluate_results.py` accepts one or more CSV files. Without arguments, it expects all three baseline files. With multiple files, it verifies that examples and normalized references match. Empty CSV files and duplicate sample filenames are errors.

MangaLMM scripts also use the repository's Hamlet pages by default. For a complete external Manga109 dataset, set `MANGA109_ROOT` to a directory containing `images/` and `annotations/`. Preparing the official sample across multiple books requires that complete dataset. MangaLMM needs a separate environment with CUDA, PyTorch, Transformers, Accelerate, bitsandbytes, qwen-vl-utils, and datasets; these are separate from the standard application installation.

## My contribution

In this project, I:

- Automate preparation of 3116 Hamlet crops and reference CSV data from XML annotations.
- Compare Tesseract, PaddleOCR, and Manga OCR on the same data using CER, Exact Match, empty predictions, and recorded runtime.
- Compare five systems qualitatively on one image and examine omissions, distorted text, and duplicate regions.
- Build a fixed 50-crop comparison with ChatGPT and analyze cases where language context changes the original text.
- Adapt MangaLMM to a 6 GB RTX 4050 using 4-bit quantization and a reduced input pixel budget.
- Investigate 2 × 2 tiled inference, full-page coordinate conversion, result merging, and duplicate removal.
- Correct coordinate matching after Qwen image resizing using `image_grid_thw`; the corrected evaluation finds 37 spatial matches instead of zero.
- Evaluate detection and end-to-end OCR on 50 images from the official MangaOCR test split.
- Integrate OCR and local Ollama into Manga Reader with region previews and result export.

I use Codex to help verify and improve the code. This work includes path and error-handling fixes, command-line options, incremental result saving, and 14 automated tests. Research results and application checks are documented separately.

