import os
import csv
import time
import pytesseract
from PIL import Image


DATASET_DIR = r"C:\mangalmm_test\dataset"
GROUND_TRUTH_CSV = r"C:\mangalmm_test\ground_truth.csv"
OUTPUT_CSV = r"C:\mangalmm_test\results_tesseract.csv"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def load_ground_truth(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    gt_rows = load_ground_truth(GROUND_TRUTH_CSV)

    results = []

    total = len(gt_rows)

    for i, row in enumerate(gt_rows, start=1):
        file_name = row["file"]
        gt_text = row["text"]

        image_path = os.path.join(DATASET_DIR, file_name)

        if not os.path.exists(image_path):
            print(f"[{i}/{total}] MISSING: {file_name}")
            continue

        image = Image.open(image_path).convert("RGB")

        start = time.perf_counter()

        pred_text = pytesseract.image_to_string(
            image,
            lang="jpn",
            config="--psm 6"
        )

        elapsed = time.perf_counter() - start

        pred_text = pred_text.strip().replace("\n", " ")

        results.append({
            "file": file_name,
            "ground_truth": gt_text,
            "prediction": pred_text,
            "time_sec": f"{elapsed:.4f}"
        })

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] done")

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "ground_truth", "prediction", "time_sec"]
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("====================================")
    print("TESSERACT BATCH DONE")
    print("====================================")
    print("Saved to:", OUTPUT_CSV)
    print("Samples processed:", len(results))


if __name__ == "__main__":
    main()