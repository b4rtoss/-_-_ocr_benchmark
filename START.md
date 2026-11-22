# Manga Reader setup guide

Manga Reader is a local desktop application for manga page images. It detects text regions, recognizes the original text, and can use Ollama to translate dialogue into English and describe scenes, characters, and page events. The application does not require XML annotations or reference transcriptions.

## Install and launch

1. Install Python 3.12 and run `Install.cmd` from the repository folder.
2. For dialogue translation and scene analysis, install [Ollama for Windows](https://ollama.com/download/windows) and start it.
3. Download the default model:

   ```powershell
   ollama pull qwen3-vl:4b-instruct
   ```

4. Open `Launch.cmd`. Add individual images with **Add pages** or choose a chapter folder with **Open folder**.
5. Select the source language and click **Analyze**.
6. Select a page on the left to view its image, OCR regions, and results. Click **Export** for JSON with coordinates or a plain text file. Use **Save session** to continue later.

The interface, status messages, and generated analysis are in English. OCR preserves text in its original language. The default Ollama model is the `instruct` variant; you can change it in **Settings**. Model downloads require an internet connection and sufficient disk space and memory.

## OCR-only mode

In **Settings**, clear **Scenes, characters and story (Ollama)**. OCR-only mode does not require Ollama. Choose **Japanese**, **English**, or **Russian** as the source language.

PaddleOCR runs on the CPU and does not require CUDA. The current page OCR pipeline uses PaddleOCR for both detection and recognition. Manga OCR is used in the research scripts to read text crops; it is not currently part of the desktop pipeline.

## Page order and context

Pages are sorted naturally by filename: 1, 2, 10. Add pages from one chapter at a time. **Analyze** processes all added pages, not just the currently selected preview. For a new chapter, open **Settings** and click **Clear**.

When story analysis is enabled, each page receives a shortened context from earlier pages in the current run. This is not persistent memory of an entire book. Images and analysis requests are sent to the local Ollama service at `127.0.0.1:11434`. PaddleOCR downloads models on its first run and reuses a local cache afterwards.

## Processing and saving

The window remains responsive while pages are processed sequentially in a background thread. **Stop** ends the queue after the current page. It does not interrupt an OCR call or Ollama request already in progress; analysis requests use a 900-second HTTP timeout.

**Save session** writes a `.manga-session.json` file containing the ordered page list, completed OCR and analysis results (including warnings and errors), selected page, source language, Ollama setting, and model name. **Open session** restores these without running OCR or contacting Ollama. JSON/TXT exports are separate and cannot be opened as sessions.

Images are not embedded. Keep them in place, or move the session and its image folders together while preserving their relative layout. Images on another drive use absolute paths. If an image is missing, its saved text remains available, but the preview and reanalysis need the original file.

Saving is manual. You can save partial results while processing; subsequent pages require another save. Before closing, clearing, or opening another session, the application offers to save unsaved changes. A canceled or failed save keeps the current session open. A failed write preserves the previous session file. There is no automatic crash recovery, and **Analyze** processes all added pages again rather than resuming only unfinished pages. Stop processing and wait for the current page before closing the application.

If an Ollama analysis request fails after successful OCR, the recognized text remains available with a warning. If OCR fails in story mode, the application can try image analysis directly and records the OCR failure as a warning. Missing Ollama or a missing model is reported during the initial check. A model response that reaches its length limit is marked with a warning.

## Known limitations

- Speed depends on hardware, page size, and model initialization. CPU OCR can take more than a minute for an individual crop; full pages can take substantially longer.
- Vertical lettering, handwriting, sound effects, unusual fonts, and complex layouts may be recognized incorrectly. The displayed reading order is approximate.
- OCR confidence is not a measured accuracy score for a transcription, translation, or scene analysis. Review the results.
- Vision models may omit details, repeat themselves, or invent interpretations. The analysis is not guaranteed understanding of any manga.
- PDF and CBZ import, automatic text removal, and typesetting translated pages are not implemented.

## Tested installation and troubleshooting

After updating the project, run `Install.cmd` again to update `.venv` dependencies. The tested Windows/Python 3.12 CPU configuration uses PaddlePaddle 3.3.1, PaddleOCR 3.7.0, PaddleX 3.7.2, and Pillow 13.0.0.

On the tested Windows system, oneDNN caused `ConvertPirAttribute2RuntimeAttribute` errors during PP-OCRv5 inference. The application disables oneDNN on Windows as a workaround. This affects available acceleration.

A separate GPU environment produced `WinError 127` while loading `cudnn_cnn64_9.dll`. This indicates a library compatibility problem in that environment. The standard application installation uses a separate CPU build of PaddlePaddle and does not require cuDNN.

Run the tests without downloading OCR models:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Details are available in the [verification report](docs/verification.md).

## References

- [PaddleOCR API](https://www.paddleocr.ai/latest/en/quick_start.html)
- [Qwen3-VL models for Ollama](https://ollama.com/library/qwen3-vl)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
