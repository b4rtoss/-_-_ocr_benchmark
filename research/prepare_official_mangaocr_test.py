from .benchmark_paths import BASE_DIR, MANGA109_ROOT
import json
import random
import shutil
from pathlib import Path

from datasets import load_dataset


# =========================================================
# SETTINGS
# =========================================================

SAMPLE_SIZE = 50
RANDOM_SEED = 42

# Previously downloaded Manga109 dataset
MANGA_ROOT = MANGA109_ROOT

if MANGA_ROOT is None:
    raise RuntimeError("Set MANGA109_ROOT to the Manga109 dataset directory (images/ and annotations/).")
IMAGE_ROOT = MANGA_ROOT / "images"

OUTPUT_DIR = BASE_DIR / 'official_mangaocr_test50'

IMAGE_OUTPUT = OUTPUT_DIR / "images"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST = OUTPUT_DIR / "manifest.jsonl"


# =========================================================
# DOWNLOAD OFFICIAL METADATA
# =========================================================

print("Loading official MangaOCR TEST split...")

dataset = load_dataset(
    "hal-utokyo/MangaOCR",
    split="test"
)

print("Official test examples:", len(dataset))


# =========================================================
# FUNCTIONS
# =========================================================

def get_user_prompt(messages):

    for message in messages:

        if message.get("role") != "user":
            continue

        for content in message.get(
            "content",
            []
        ):

            if (
                content.get("type") == "text"
                and content.get("text")
            ):

                return content["text"]

    return ""


def get_ground_truth(messages):

    for message in messages:

        if message.get("role") != "assistant":
            continue

        texts = []

        for content in message.get(
            "content",
            []
        ):

            text = content.get("text")

            if text:
                texts.append(text)

        return "\n".join(texts)

    return ""


def locate_local_image(example_id):

    # example:
    # ARMS_000012
    #
    # book = ARMS
    # page = 12

    if "_" not in example_id:
        return None

    book, page_string = (
        example_id.rsplit("_", 1)
    )

    try:
        page_number = int(page_string)

    except ValueError:
        return None


    # Official MangaLMM code:
    # page index -> zfill(3)
    page_file = (
        f"{page_number:03d}.jpg"
    )


    possible = [
        IMAGE_ROOT
        / book
        / page_file,

        IMAGE_ROOT
        / book
        / page_file.upper(),
    ]


    for path in possible:

        if path.exists():
            return path


    # Also handle PNG/JPEG files.
    for ext in [
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    ]:

        path = (
            IMAGE_ROOT
            / book
            / (
                f"{page_number:03d}"
                + ext
            )
        )

        if path.exists():
            return path


    return None


# =========================================================
# RANDOM REPRODUCIBLE SAMPLE
# =========================================================

random.seed(RANDOM_SEED)

indices = list(
    range(len(dataset))
)

random.shuffle(indices)


selected = []

missing = []


# Traverse the randomized order until
# 50 available local images have been found.

for index in indices:

    row = dataset[index]

    example_id = row["id"]

    image_path = locate_local_image(
        example_id
    )


    if image_path is None:

        missing.append(
            example_id
        )

        continue


    selected.append(
        (
            index,
            row,
            image_path
        )
    )


    if len(selected) >= SAMPLE_SIZE:
        break


# =========================================================
# VALIDATION
# =========================================================

print()
print("Found local images:", len(selected))
print("Missing encountered:", len(missing))


if len(selected) < SAMPLE_SIZE:

    print()
    print(
        "WARNING: could not find "
        f"{SAMPLE_SIZE} matching images."
    )

    print(
        "Check Manga109 path/version."
    )


# =========================================================
# SAVE
# =========================================================

with open(
    MANIFEST,
    "w",
    encoding="utf-8"
) as manifest_file:


    for number, (
        original_index,
        row,
        image_path
    ) in enumerate(
        selected,
        start=1
    ):

        example_id = row["id"]

        messages = row["messages"]


        prompt = get_user_prompt(
            messages
        )

        ground_truth = (
            get_ground_truth(
                messages
            )
        )


        new_filename = (
            f"{number:03d}_"
            f"{example_id}.jpg"
        )


        destination = (
            IMAGE_OUTPUT
            / new_filename
        )


        shutil.copy2(
            image_path,
            destination
        )


        record = {
            "sample_number": number,

            "official_test_index": (
                original_index
            ),

            "id": example_id,

            "image": str(
                destination
            ),

            "source_image": str(
                image_path
            ),

            "prompt": prompt,

            "ground_truth": (
                ground_truth
            ),
        }


        manifest_file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


        print(
            f"[{number:02d}/{SAMPLE_SIZE}] "
            f"{example_id}"
        )


# =========================================================
# DONE
# =========================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    "Samples:",
    len(selected)
)

print(
    "Images:",
    IMAGE_OUTPUT
)

print(
    "Manifest:",
    MANIFEST
)