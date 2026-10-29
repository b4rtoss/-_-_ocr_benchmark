import os
import csv
import time

from PIL import Image
from manga_ocr import MangaOcr


DATASET_DIR = r"C:\mangalmm_test\dataset"
GROUND_TRUTH_CSV = r"C:\mangalmm_test\ground_truth.csv"
OUTPUT_CSV = r"C:\mangalmm_test\results_mangaocr.csv"


# =========================================================
# ЗАГРУЖАЕМ MANGA OCR ОДИН РАЗ
# =========================================================

print("Loading Manga OCR...")

mocr = MangaOcr()

print("Manga OCR loaded.")
print()


# =========================================================
# ЧИТАЕМ GROUND TRUTH
# =========================================================

with open(
    GROUND_TRUTH_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    gt_rows = list(csv.DictReader(f))


total = len(gt_rows)

print("Total samples:", total)
print()


# =========================================================
# СОЗДАЁМ CSV
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "ground_truth",
            "prediction",
            "time_sec",
            "status"
        ]
    )

    writer.writeheader()


# =========================================================
# OCR
# =========================================================

for i, row in enumerate(gt_rows, start=1):

    file_name = row["file"]
    ground_truth = row["text"]

    image_path = os.path.join(
        DATASET_DIR,
        file_name
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        start = time.perf_counter()

        prediction = mocr(image)

        elapsed = time.perf_counter() - start

        prediction = str(prediction).strip()

        status = "ok"


    except Exception as e:

        prediction = ""
        elapsed = 0
        status = "error:" + type(e).__name__


    # =====================================================
    # СОХРАНЯЕМ РЕЗУЛЬТАТ СРАЗУ
    # =====================================================

    with open(
        OUTPUT_CSV,
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "ground_truth",
                "prediction",
                "time_sec",
                "status"
            ]
        )

        writer.writerow({
            "file": file_name,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "time_sec": f"{elapsed:.4f}",
            "status": status
        })


    if i % 50 == 0 or i == total:

        print(
            f"[{i}/{total}] done"
        )


print()
print("====================================")
print("MANGA OCR BATCH DONE")
print("====================================")

print(
    "Saved to:",
    OUTPUT_CSV
)

print(
    "Samples processed:",
    total
)