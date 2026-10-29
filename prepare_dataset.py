import os
import csv
import xml.etree.ElementTree as ET
from PIL import Image
# =========================================================
# ПУТИ
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "Hamlet"
)

XML_PATH = os.path.join(
    IMAGE_DIR,
    "Hamlet.xml"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "ground_truth.csv"
)

# Небольшой отступ вокруг текста
PADDING = 5

# =========================================================
# СОЗДАЁМ ПАПКУ
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# ЧИТАЕМ XML
# =========================================================

tree = ET.parse(XML_PATH)
root = tree.getroot()

rows = []

crop_number = 1


# =========================================================
# ОБХОДИМ СТРАНИЦЫ
# =========================================================

for page in root.findall(".//page"):

    page_index = int(page.attrib["index"])

    # Например page index=10 -> 010.jpg
    base_name = f"{page_index:03d}"

    # Ищем изображение
    image_path = None

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

        candidate = os.path.join(
            IMAGE_DIR,
            base_name + ext
        )

        if os.path.exists(candidate):
            image_path = candidate
            break


    if image_path is None:
        print(
            f"WARNING: image for page {page_index} not found"
        )
        continue


    image = Image.open(image_path).convert("RGB")

    width, height = image.size


    # =====================================================
    # ВСЕ TEXT BOX НА ЭТОЙ СТРАНИЦЕ
    # =====================================================

    for text_element in page.findall("text"):

        text = (text_element.text or "").strip()

        if not text:
            continue


        xmin = int(text_element.attrib["xmin"])
        ymin = int(text_element.attrib["ymin"])
        xmax = int(text_element.attrib["xmax"])
        ymax = int(text_element.attrib["ymax"])


        # Добавляем небольшой padding
        left = max(0, xmin - PADDING)
        top = max(0, ymin - PADDING)

        right = min(width, xmax + PADDING)
        bottom = min(height, ymax + PADDING)


        crop = image.crop(
            (left, top, right, bottom)
        )


        crop_name = f"{crop_number:05d}.jpg"

        crop_path = os.path.join(
            OUTPUT_DIR,
            crop_name
        )

        crop.save(
            crop_path,
            quality=95
        )


        rows.append({
            "file": crop_name,
            "text": text,

            "page": page_index,

            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,

            "text_id": text_element.attrib.get(
                "id",
                ""
            )
        })


        crop_number += 1


# =========================================================
# СОХРАНЯЕМ GROUND TRUTH
# =========================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "text",
            "page",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "text_id"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# =========================================================
# ИТОГ
# =========================================================

print()
print("======================================")
print("DATASET CREATED")
print("======================================")

print(
    "Number of text crops:",
    len(rows)
)

print(
    "Images:",
    OUTPUT_DIR
)

print(
    "Ground truth:",
    CSV_PATH
)