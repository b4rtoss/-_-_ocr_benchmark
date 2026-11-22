from benchmark_paths import BASE_DIR
import argparse
import math
from pathlib import Path
import csv
import os
import unicodedata



FILES = {
    "Tesseract": os.path.join(
        BASE_DIR,
        "results_tesseract.csv"
    ),

    "PaddleOCR": os.path.join(
        BASE_DIR,
        "results_paddleocr.csv"
    ),

    "Manga OCR": os.path.join(
        BASE_DIR,
        "results_mangaocr.csv"
    ),
}


# =========================================================
# JAPANESE TEXT NORMALIZATION
# =========================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text)

    # For example, normalize Unicode variants
    # to a consistent representation.
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Ignore spaces and line breaks when evaluating OCR.
    text = "".join(
        text.split()
    )

    return text


# =========================================================
# LEVENSHTEIN DISTANCE
# =========================================================

def edit_distance(a, b):

    if len(a) < len(b):
        a, b = b, a

    previous = list(
        range(len(b) + 1)
    )

    for i, ca in enumerate(a, start=1):

        current = [i]

        for j, cb in enumerate(b, start=1):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1

            substitution = (
                previous[j - 1]
                + (ca != cb)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


# =========================================================
# EVALUATE ONE MODEL
# =========================================================

def evaluate(csv_path):

    total_edits = 0
    total_reference_chars = 0

    exact_matches = 0
    total_samples = 0

    empty_predictions = 0
    failed_samples = 0

    times = []
    seen = set()


    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required = {'file', 'ground_truth', 'prediction', 'time_sec'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{csv_path}: missing required CSV columns')
        for row in reader:
            if not row['file'] or row['file'] in seen:
                raise ValueError(f'{csv_path}: empty or duplicate file name')
            seen.add(row['file'])

            reference = normalize(
                row.get(
                    "ground_truth",
                    ""
                )
            )

            prediction = normalize(
                row.get(
                    "prediction",
                    ""
                )
            )

            # Use the status column when present.
            status = row.get(
                "status",
                "ok"
            )

            if status != "ok":
                failed_samples += 1


            if prediction == "":
                empty_predictions += 1


            distance = edit_distance(
                reference,
                prediction
            )

            total_edits += distance

            total_reference_chars += len(
                reference
            )

            if reference == prediction:
                exact_matches += 1


            # Failed calls remain in quality metrics, but cannot make latency look faster.
            if status == 'ok':
                try:
                    duration = float(row['time_sec'])
                    if math.isfinite(duration) and duration >= 0:
                        times.append(duration)
                except (ValueError, TypeError):
                    pass

            total_samples += 1


    if not total_samples or not total_reference_chars:
        raise ValueError(f"{csv_path}: no samples or reference characters to evaluate")

    # Corpus-level CER
    if total_reference_chars > 0:

        cer = (
            total_edits
            / total_reference_chars
        )

    else:
        cer = 0


    exact_match = (
        exact_matches
        / total_samples
        if total_samples
        else 0
    )


    avg_time = (
        sum(times) / len(times)
        if times
        else 0
    )


    return {
        "samples": total_samples,
        "CER": cer,
        "Exact Match": exact_match,
        "empty": empty_predictions,
        "failed": failed_samples,
        "avg_time": avg_time,
    }


# =========================================================
# RUN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate OCR CSV files on the same sample.')
    parser.add_argument('csv_files', nargs='*', type=Path, help='Defaults to all three baseline result CSVs.')
    parser.add_argument('--output', type=Path, help='Save metrics to a CSV report.')
    args = parser.parse_args()
    files = [(p.stem, p) for p in args.csv_files] if args.csv_files else list(FILES.items())
    try:
        references = None
        reports = []
        for name, path in files:
            report = evaluate(path)
            with open(path, encoding='utf-8-sig', newline='') as stream:
                current = {r['file']: normalize(r['ground_truth']) for r in csv.DictReader(stream)}
            if references is not None and current != references:
                raise ValueError('Result files contain different samples or ground truth; compare the same sample.')
            references = current
            reports.append({'model': name, **report})
        print(f"{'MODEL':24} {'SAMPLES':>8} {'CER':>10} {'EXACT':>10} {'EMPTY':>7} {'FAILED':>7} {'TIME':>10}")
        for r in reports:
            print(f"{r['model']:24} {r['samples']:8} {r['CER']:10.4f} {r['Exact Match']:10.2%} {r['empty']:7} {r['failed']:7} {r['avg_time']:9.4f}s")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
                writer.writeheader()
                writer.writerows(reports)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
