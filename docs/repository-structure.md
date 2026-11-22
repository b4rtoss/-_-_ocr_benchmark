# Repository structure

[Back to the project](../README.md)

I keep the desktop application and research tools in separate Python packages:

```text
manga-reader/
├── manga_app/                 # Desktop application
│   ├── app.py                 # Tkinter interface and background processing
│   ├── engine.py              # Page OCR and optional Ollama analysis
│   ├── session.py             # Session saving and restoration
│   └── requirements.txt       # Application dependencies
├── research/                  # Dataset preparation, experiments, evaluation
│   ├── benchmark_paths.py     # Shared dataset and output locations
│   ├── batch_runner.py        # Shared crop OCR runner
│   ├── prepare_*.py           # Dataset and evaluation sample preparation
│   ├── run_*.py               # OCR and MangaLMM inference
│   ├── evaluate_*.py          # Recognition and detection metrics
│   ├── compare_ocr_test50.py   # Four-system crop comparison
│   └── requirements.txt       # Baseline research dependencies
├── tests/                     # Automated workflow and session checks
├── docs/                      # Application and research documentation
│   └── media/                 # README demonstration
├── Hamlet/                    # Existing research images and XML annotations
├── .github/workflows/         # Windows and Linux test automation
├── Install.cmd                # Windows application setup
├── Launch.cmd                 # Windows application launcher
├── START.md                   # Application setup and usage
└── README.md                  # Project introduction
```

## Application

The application code and its dependencies live in [`manga_app/`](../manga_app/). From the repository root, use `Install.cmd` and `Launch.cmd`, or launch an installed environment with:

```powershell
python -m manga_app.app
```

See the [setup guide](../START.md) and [application architecture](application.md).

## Research

All experimental Python scripts, shared research helpers, and the baseline dependency file live in [`research/`](../research/). Run research modules from the repository root:

```powershell
python -m pip install -r research/requirements.txt
python -m research.prepare_dataset
python -m research.run_mangaocr_batch --limit 3
python -m research.evaluate_results results_mangaocr.csv --output report.csv
```

The former root-level commands such as `python run_mangaocr_batch.py` are replaced by `python -m research.run_mangaocr_batch`. Module invocation keeps package imports consistent, including the PaddleOCR runner's reuse of the application OCR engine.

Moving the code does not move experiment data. By default, images remain under the repository's `Hamlet/`, and generated crops, reference CSVs, and results retain their previous root-level locations. `OCR_BENCHMARK_DIR` and `MANGA109_ROOT` continue to support external datasets. Generated artifacts and local sessions are ignored by Git.

See the [research tool index](../research/README.md), [reproduction guide](research-guide.md), and [methodology and results](experiment-results.md). MangaLMM still requires its separate GPU environment.

## Tests

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Tests need Pillow and Tkinter, but not downloaded OCR models or a running Ollama service. Linux CI runs Tkinter checks inside a virtual display.
