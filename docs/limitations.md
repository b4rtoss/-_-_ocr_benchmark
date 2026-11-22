# Limitations and known issues

[Back to the project](../README.md)

### Recognition and analysis quality

- The main OCR comparison uses crops from one manga, Hamlet. The ChatGPT comparison uses 50 crops. I do not generalize these results to every manga, language, or scan type.
- Vertical lettering, handwriting, stylized sound effects, and complex panel layouts may be recognized incorrectly. The application's approximate region order does not recover panel structure.
- PaddleOCR handles both detection and recognition in the current application. Full-page detection quality still needs separate evaluation; integrating Manga OCR remains planned work.
- The local MangaLMM configuration produced low end-to-end scores. Six of 50 pages had no parsed detections. Reduced resolution and 4-bit quantization differ from the original research conditions.
- Ollama may omit details, repeat itself, or add unsupported interpretations. Review translations and analysis. OCR confidence is not measured transcription or translation accuracy.

### Performance and compatibility

- The standard application uses CPU PaddleOCR. During verification, one small crop took over a minute; full pages can take substantially longer.
- On the tested Windows configuration, PaddlePaddle 3.3.1 with oneDNN raised `ConvertPirAttribute2RuntimeAttribute`. The code disables oneDNN on Windows as a workaround, affecting available acceleration.
- A separate GPU environment raised `WinError 127` while loading `cudnn_cnn64_9.dll`. That compatibility issue is unresolved. The standard CPU installation does not require cuDNN; GPU inference requires matching libraries.
- First-time OCR setup requires model downloads. Scene analysis requires a running local Ollama service and the selected model.

### Current application limits

- **Stop** ends the queue after the current page and does not interrupt an active OCR call or Ollama request. Analysis requests use a 900-second HTTP timeout.
- Sessions can be saved and reopened manually, but autosave and automatic crash recovery are not implemented. Images are referenced rather than embedded. Running analysis again processes all added pages again, including pages with saved results.
- Image files are supported. PDF/CBZ import, automatic source-text removal, and translated-page typesetting are not implemented.

### Experiment reproducibility

- Original ChatGPT responses are not published, and the exact model version and run date were not recorded. Library versions and settings are incomplete for some research runs.
- Official MangaOCR sample selection depends on available local scans. Seed 42 fixes the random order, not sample representativeness.
- Crop and scaled MangaLMM evaluators normalize text differently. The 50-crop comparison script excludes missing rows, so `samples = 50` and `missing = 0` must be checked before comparison.
- Some MangaLMM parameters remain in research scripts, and that environment lacks a separately pinned, verified dependency set. The current baseline runner performs no separate warm-up or repeated timing series.
- Maintenance verification used automated tests and short smoke runs. The complete benchmark and MangaLMM were not rerun; I keep historical research results separate from current-version checks.

Coordinate matching for MangaLMM evaluation and old absolute paths have been corrected. Successful OCR survives a later Ollama failure. See the [experiment results](experiment-results.md), [verification report](verification.md), and [setup guide](../START.md).
