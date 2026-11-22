# Application verification

I use Codex to help check the project on Windows 10 Home x64, build 19045, with Python 3.12, an Intel Core i5-12450H (8 cores, 12 threads), 16 GB RAM, and an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB dedicated VRAM. This report records runtime and error-handling checks. Research metrics are reported separately in the [experiment results](experiment-results.md).

## Fixes

- Removed absolute paths from the previous working environment. Shared paths are defined in `benchmark_paths.py`, with `OCR_BENCHMARK_DIR` and `MANGA109_ROOT` overrides.
- The ChatGPT sample preparation accepts the dataset's `text` column and the compatible `ground_truth` column. Sample size and seed are command-line options.
- The three baseline runners share a CSV format and command-line interface. PaddleOCR defaults to CPU. Models are initialized before individual calls are timed. Missing images and recognition failures remain in the results; resume validates references and prevents duplicate rows.
- Evaluation reports failures, excludes failed calls from average latency, rejects empty or duplicate samples, and checks that compared CSV files contain the same examples and reference text.
- Manga Reader preserves successful OCR if subsequent Ollama analysis fails. Fixed package imports, progress reset, corrupt-image preview handling, and clearing data during processing.
- Pinned direct dependencies and added GitHub Actions tests for Windows and Linux.
- Translated the interface, documentation, and model instructions into English. The launcher is `Launch.cmd` and the setup guide is `START.md`. OCR still returns the original source text.

## Original runtime checks

- Installed the application and baseline benchmark dependencies in a new Python 3.12 environment; `pip check` passed.
- Ran 13 workflow tests covering crop and sample preparation, resume with errors, CER/NFKC, matching sample sets, report export, CLI help, page order, and OCR/Ollama failures.
- Parsed all Python scripts for syntax errors.
- Checked Tkinter window creation, image addition and preview, JSON/TXT export, and clearing records.
- Generated 11 real crops from `Hamlet/008.jpg` and its XML annotations. The first three were processed by all three OCR systems without call errors. PaddleOCR resume after interruption was also checked.
- Called `Analyzer.analyze` on a real crop: three OCR regions, nonempty Ollama analysis, no warnings, approximately 50 seconds.
- Sent successful text and page-image requests to Ollama using `qwen3-vl:4b-instruct`. These confirm API operation, not the factual accuracy of the generated description.

Diagnostic results on the same three crops:

| System | Samples | Call errors | CER | Average call time |
| --- | ---: | ---: | ---: | ---: |
| Manga OCR | 3 | 0 | 0.0588 | 0.77 s |
| Tesseract | 3 | 0 | 1.0000 | 0.50 s |
| PaddleOCR CPU | 3 | 0 | 0.2549 | 30.69 s |

Three samples are insufficient to assess model quality. Timing includes the selected engine's call-time preprocessing, but not initial model loading; internal initialization can still affect individual calls. These measurements belong to the original technical check and are not new English-interface benchmark results.

## English localization checks

- All 14 tests passed, including the regression check for English analysis instructions while preserving source-language OCR text.
- Checked English interface labels, settings, source-language mapping, JSON/TXT export, and clearing records through a Tkinter integration smoke test.
- Sent a real image-analysis request to local Ollama with the updated English prompt. I reused previously verified OCR regions from the same crop, so this checks English generation without claiming a new PaddleOCR benchmark. The model returned English analysis and dialogue translations without warnings; accuracy still requires review.
- `pip check` passed. Research-script changes outside the application are English comment translations only.

## Verification limits

For maintenance checks, I use automated tests and short smoke runs. The full benchmark, quantized MangaLMM inference, complete PaddleOCR page processing, and separate English/Russian OCR models have not been rerun as part of that original check. MangaLMM scripts were checked for syntax and path consistency. Earlier experiments on 3116 crops, a 50-crop comparison, and 50 MangaLMM pages are documented in the [research report](experiment-results.md).

A separate GPU environment produced `WinError 127` when loading cuDNN. That environment was not changed; the standard application uses CPU. PaddlePaddle 3.3.1 with oneDNN produced `ConvertPirAttribute2RuntimeAttribute` on this Windows system, so oneDNN is disabled on Windows. OCR then ran, but one crop took more than a minute. Full pages can take substantially longer. Cancellation ends the queue after the current call.

## Run the checks

The current suite includes the original workflow checks and an English-output request regression test. It does not require OCR model downloads:

```powershell
python -m unittest discover -s tests -v
```
