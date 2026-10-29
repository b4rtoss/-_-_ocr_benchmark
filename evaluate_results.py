import csv
import os
import unicodedata


BASE_DIR = r"C:\mangalmm_test"

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
# НОРМАЛИЗАЦИЯ ЯПОНСКОГО ТЕКСТА
# =========================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text)

    # Например, приводит разные Unicode-варианты
    # символов к единому виду
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Для OCR не считаем пробелы/переносы строк ошибкой
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
# ОЦЕНКА ОДНОЙ МОДЕЛИ
# =========================================================

def evaluate(csv_path):

    total_edits = 0
    total_reference_chars = 0

    exact_matches = 0
    total_samples = 0

    empty_predictions = 0
    failed_samples = 0

    times = []


    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

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

            # Если есть колонка status
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


            try:
                times.append(
                    float(
                        row.get(
                            "time_sec",
                            0
                        )
                    )
                )
            except:
                pass


            total_samples += 1


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

print()
print("=" * 74)

print(
    f"{'MODEL':15}"
    f"{'SAMPLES':>10}"
    f"{'CER':>12}"
    f"{'EXACT':>12}"
    f"{'EMPTY':>10}"
    f"{'TIME':>12}"
)

print("=" * 74)


for model_name, csv_path in FILES.items():

    result = evaluate(csv_path)

    print(
        f"{model_name:15}"
        f"{result['samples']:>10}"
        f"{result['CER']:>12.4f}"
        f"{result['Exact Match'] * 100:>11.2f}%"
        f"{result['empty']:>10}"
        f"{result['avg_time']:>11.4f}s"
    )


print("=" * 74)

print()
print("CER: lower is better")
print("Exact Match: higher is better")
print()