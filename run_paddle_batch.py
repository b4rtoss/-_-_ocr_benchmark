import os
import csv
import time

# ВАЖНО ДЛЯ WINDOWS:
# заставляем PaddleX использовать путь без кириллицы
PADDLE_CACHE = r"C:\paddlex_cache"
os.makedirs(PADDLE_CACHE, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = PADDLE_CACHE

import paddle
from paddleocr import PaddleOCR


DATASET_DIR = r"C:\mangalmm_test\dataset"
GROUND_TRUTH_CSV = r"C:\mangalmm_test\ground_truth.csv"
OUTPUT_CSV = r"C:\mangalmm_test\results_paddleocr.csv"


# =========================================================
# ПРОВЕРКА GPU
# =========================================================

print("Paddle version:", paddle.__version__)
print("CUDA available:", paddle.device.is_compiled_with_cuda())
print("Device:", paddle.device.get_device())
print()


# =========================================================
# ЗАГРУЖАЕМ PADDLEOCR ОДИН РАЗ
# =========================================================

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="japan",
    ocr_version="PP-OCRv5",

    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,

    device="gpu",
)

print("PaddleOCR loaded.")
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
# ЕСЛИ CSV УЖЕ ЕСТЬ — МОЖЕМ ПРОДОЛЖИТЬ
# =========================================================

processed = set()

if os.path.exists(OUTPUT_CSV):

    with open(
        OUTPUT_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            processed.add(row["file"])

    print("Already processed:", len(processed))
    print()

else:

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

    if file_name in processed:
        continue

    image_path = os.path.join(
        DATASET_DIR,
        file_name
    )

    prediction = ""
    elapsed = 0.0
    status = "ok"

    try:

        start = time.perf_counter()

        results = list(
            ocr.predict(image_path)
        )

        elapsed = time.perf_counter() - start

        texts = []

        for res in results:

            data = res.json["res"]

            rec_texts = data.get(
                "rec_texts",
                []
            )

            texts.extend(rec_texts)

        # Для кропа ожидаем один текстовый фрагмент.
        # Если Paddle вернул несколько, объединяем без пробелов.
        prediction = "".join(texts).strip()

    except Exception as e:

        prediction = ""
        status = "error:" + type(e).__name__


    # =====================================================
    # СРАЗУ СОХРАНЯЕМ РЕЗУЛЬТАТ
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
print("PADDLE OCR BATCH DONE")
print("====================================")

print(
    "Saved to:",
    OUTPUT_CSV
)

print(
    "Samples processed:",
    total
)