import csv
import random
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\mangalmm_test")

# Папка со всеми 3116 кропами
DATASET_DIR = BASE / "dataset"

# CSV с Ground Truth
GROUND_TRUTH_CSV = BASE / "ground_truth.csv"

# Папка для выбранных 50 изображений
OUTPUT_DIR = BASE / "chatgpt_test50"

# CSV для выбранных 50 изображений
OUTPUT_CSV = BASE / "chatgpt_test50.csv"


# ============================================================
# СОЗДАЁМ ПАПКУ
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ФИКСИРОВАННЫЙ RANDOM SEED
#
# Благодаря seed=42 каждый запуск выбирает
# одни и те же 50 примеров.
# ============================================================

random.seed(42)


# ============================================================
# ЧИТАЕМ GROUND TRUTH CSV
# ============================================================

with open(
    GROUND_TRUTH_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    rows = list(
        csv.DictReader(f)
    )


print(
    "Total crops:",
    len(rows)
)


# ============================================================
# ПРОВЕРЯЕМ НАЗВАНИЯ СТОЛБЦОВ
# ============================================================

if not rows:
    raise ValueError(
        "ground_truth.csv пустой"
    )


print(
    "CSV columns:",
    list(rows[0].keys())
)


if "file" not in rows[0]:
    raise KeyError(
        "В CSV нет столбца 'file'"
    )


if "ground_truth" not in rows[0]:
    raise KeyError(
        "В CSV нет столбца 'ground_truth'"
    )


if len(rows) < 50:
    raise ValueError(
        "В ground_truth.csv меньше 50 примеров"
    )


# ============================================================
# ВЫБИРАЕМ 50 СЛУЧАЙНЫХ ПРИМЕРОВ
# ============================================================

sample = random.sample(
    rows,
    50
)


# ============================================================
# КОПИРУЕМ ВЫБРАННЫЕ КРОПЫ
# ============================================================

output_rows = []


for number, row in enumerate(
    sample,
    start=1
):

    original_filename = str(
        row["file"]
    ).strip()


    ground_truth = str(
        row["ground_truth"]
    ).strip()


    # --------------------------------------------------------
    # ИСХОДНЫЙ ФАЙЛ
    # --------------------------------------------------------

    source_path = (
        DATASET_DIR
        / original_filename
    )


    if not source_path.exists():

        raise FileNotFoundError(
            f"Не найден файл:\n{source_path}"
        )


    # --------------------------------------------------------
    # НОВОЕ ИМЯ
    #
    # 001.jpg
    # 002.jpg
    # ...
    # 050.jpg
    # --------------------------------------------------------

    extension = (
        source_path.suffix
        if source_path.suffix
        else ".jpg"
    )


    new_filename = (
        f"{number:03d}"
        f"{extension}"
    )


    destination_path = (
        OUTPUT_DIR
        / new_filename
    )


    # --------------------------------------------------------
    # КОПИРУЕМ ФАЙЛ
    # --------------------------------------------------------

    shutil.copy2(
        source_path,
        destination_path
    )


    # --------------------------------------------------------
    # СОХРАНЯЕМ ИНФОРМАЦИЮ ДЛЯ CSV
    # --------------------------------------------------------

    output_rows.append({

        "number": number,

        "original_file": (
            original_filename
        ),

        "test_file": (
            new_filename
        ),

        "ground_truth": (
            ground_truth
        ),

        "chatgpt_prediction": "",
    })


    print(
        f"{number:03d}/050  "
        f"{original_filename} "
        f"-> {new_filename}"
    )


# ============================================================
# СОХРАНЯЕМ CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "number",
            "original_file",
            "test_file",
            "ground_truth",
            "chatgpt_prediction",
        ]
    )


    writer.writeheader()

    writer.writerows(
        output_rows
    )


# ============================================================
# ПРОВЕРКА
# ============================================================

created_images = list(
    OUTPUT_DIR.glob("*")
)


# ============================================================
# ГОТОВО
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    "Total source crops:",
    len(rows)
)

print(
    "Selected samples:",
    len(output_rows)
)

print(
    "Files in output folder:",
    len(created_images)
)

print()
print(
    "Images saved to:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "CSV saved to:"
)

print(
    OUTPUT_CSV
)