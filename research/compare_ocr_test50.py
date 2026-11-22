from .benchmark_paths import BASE_DIR
import csv
import re
import unicodedata
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = BASE_DIR

CHATGPT_CSV = BASE / "chatgpt_test50.csv"

TESSERACT_CSV = BASE / "results_tesseract.csv"
PADDLE_CSV = BASE / "results_paddleocr.csv"
MANGAOCR_CSV = BASE / "results_mangaocr.csv"

SUMMARY_CSV = BASE / "ocr_comparison_test50.csv"
DETAILS_CSV = BASE / "ocr_comparison_test50_details.csv"


# ============================================================
# TEXT NORMALIZATION
#
# Apply the same normalization to all four systems:
# - Unicode NFKC
# - remove spaces and line breaks
#
# Do not translate, correct Japanese, or substitute punctuation.
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = re.sub(
        r"[\s\u3000]+",
        "",
        text
    )

    return text


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein(a, b):

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    previous = list(
        range(len(b) + 1)
    )

    for i, ca in enumerate(
        a,
        start=1
    ):

        current = [i]

        for j, cb in enumerate(
            b,
            start=1
        ):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (0 if ca == cb else 1)
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


# ============================================================
# READ CSV
# ============================================================

def read_csv(path):

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# FILE KEY
#
# Handle both full paths
# and bare filenames.
# ============================================================

def file_key(value):

    if value is None:
        return ""

    value = str(value).strip()

    value = value.replace(
        "\\",
        "/"
    )

    return value.split("/")[-1].lower()


# ============================================================
# LOAD DATA
# ============================================================

chat_rows = read_csv(
    CHATGPT_CSV
)

tesseract_rows = read_csv(
    TESSERACT_CSV
)

paddle_rows = read_csv(
    PADDLE_CSV
)

manga_rows = read_csv(
    MANGAOCR_CSV
)


print("=" * 70)
print("OCR COMPARISON — SAME 50 CROPS")
print("=" * 70)

print()
print(
    "ChatGPT test rows:",
    len(chat_rows)
)

print(
    "Tesseract result rows:",
    len(tesseract_rows)
)

print(
    "PaddleOCR result rows:",
    len(paddle_rows)
)

print(
    "Manga OCR result rows:",
    len(manga_rows)
)


# ============================================================
# CREATE LOOKUP TABLES
# ============================================================

def make_lookup(rows):

    lookup = {}

    for row in rows:

        key = file_key(
            row.get("file", "")
        )

        if key:
            lookup[key] = row

    return lookup


tesseract_lookup = make_lookup(
    tesseract_rows
)

paddle_lookup = make_lookup(
    paddle_rows
)

manga_lookup = make_lookup(
    manga_rows
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Tesseract": {
        "lookup": tesseract_lookup,
    },

    "PaddleOCR": {
        "lookup": paddle_lookup,
    },

    "Manga OCR": {
        "lookup": manga_lookup,
    },

    "ChatGPT": {
        "lookup": None,
    },
}


# ============================================================
# STATISTICS STORAGE
# ============================================================

stats = {}

for model_name in models:

    stats[model_name] = {
        "samples": 0,
        "missing": 0,
        "empty": 0,
        "exact": 0,
        "edit_distance": 0,
        "gt_chars": 0,
        "pred_chars": 0,
        "times": [],
    }


detail_rows = []


# ============================================================
# PROCESS EXACT SAME 50 CROPS
# ============================================================

for chat_row in chat_rows:

    number = chat_row.get(
        "number",
        ""
    )

    original_file = chat_row.get(
        "original_file",
        ""
    )

    test_file = chat_row.get(
        "test_file",
        ""
    )

    key = file_key(
        original_file
    )

    gt_raw = chat_row.get(
        "ground_truth",
        ""
    )

    gt = normalize_text(
        gt_raw
    )


    predictions = {}


    # --------------------------------------------------------
    # CHATGPT
    # --------------------------------------------------------

    predictions["ChatGPT"] = {
        "prediction": chat_row.get(
            "chatgpt_prediction",
            ""
        ),

        "time_sec": "",
        "found": True,
    }


    # --------------------------------------------------------
    # OTHER OCR SYSTEMS
    # --------------------------------------------------------

    for model_name in [
        "Tesseract",
        "PaddleOCR",
        "Manga OCR",
    ]:

        lookup = models[
            model_name
        ]["lookup"]

        result_row = lookup.get(
            key
        )

        if result_row is None:

            predictions[
                model_name
            ] = {
                "prediction": "",
                "time_sec": "",
                "found": False,
            }

        else:

            predictions[
                model_name
            ] = {
                "prediction": result_row.get(
                    "prediction",
                    ""
                ),

                "time_sec": result_row.get(
                    "time_sec",
                    ""
                ),

                "found": True,
            }


    # --------------------------------------------------------
    # METRICS FOR EACH MODEL
    # --------------------------------------------------------

    for model_name, info in predictions.items():

        s = stats[
            model_name
        ]

        if not info["found"]:

            s["missing"] += 1

            continue


        pred_raw = info[
            "prediction"
        ]

        pred = normalize_text(
            pred_raw
        )


        distance = levenshtein(
            gt,
            pred
        )

        exact = (
            gt == pred
        )


        s["samples"] += 1

        s["gt_chars"] += len(
            gt
        )

        s["pred_chars"] += len(
            pred
        )

        s["edit_distance"] += (
            distance
        )


        if exact:
            s["exact"] += 1


        if pred == "":
            s["empty"] += 1


        time_value = info.get(
            "time_sec",
            ""
        )

        try:

            if str(
                time_value
            ).strip():

                s["times"].append(
                    float(
                        time_value
                    )
                )

        except Exception:
            pass


        sample_cer = (
            distance / len(gt)
            if len(gt) > 0
            else (
                0.0
                if len(pred) == 0
                else 1.0
            )
        )


        detail_rows.append({
            "number": number,
            "original_file": original_file,
            "test_file": test_file,
            "model": model_name,
            "ground_truth": gt_raw,
            "prediction": pred_raw,
            "normalized_gt": gt,
            "normalized_prediction": pred,
            "edit_distance": distance,
            "gt_chars": len(gt),
            "sample_CER": sample_cer,
            "exact_match": int(
                exact
            ),
        })


# ============================================================
# CHECK THAT OLD OCR RESULTS MATCH ALL 50 FILES
# ============================================================

print()
print("-" * 70)
print("MATCHING CHECK")
print("-" * 70)

for model_name in [
    "Tesseract",
    "PaddleOCR",
    "Manga OCR",
]:

    missing = stats[
        model_name
    ]["missing"]

    found = (
        len(chat_rows)
        - missing
    )

    print(
        f"{model_name:12s}: "
        f"found {found}/50, "
        f"missing {missing}"
    )


# ============================================================
# CALCULATE FINAL METRICS
# ============================================================

summary_rows = []


print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)


for model_name in [
    "Tesseract",
    "PaddleOCR",
    "Manga OCR",
    "ChatGPT",
]:

    s = stats[
        model_name
    ]


    # Micro CER:
    #
    # total edit operations
    # ---------------------
    # total GT characters
    #
    # This is the primary CER.
    # --------------------------------------------------------

    cer = (
        s["edit_distance"]
        / s["gt_chars"]
        if s["gt_chars"] > 0
        else 0.0
    )


    exact_rate = (
        s["exact"]
        / s["samples"]
        if s["samples"] > 0
        else 0.0
    )


    avg_time = (
        sum(
            s["times"]
        )
        / len(
            s["times"]
        )
        if s["times"]
        else None
    )


    summary_rows.append({
        "model": model_name,
        "samples": s["samples"],
        "missing": s["missing"],
        "empty_predictions": s["empty"],
        "gt_characters": s["gt_chars"],
        "edit_distance": s["edit_distance"],
        "CER": cer,
        "CER_percent": cer * 100,
        "exact_matches": s["exact"],
        "exact_match_percent": (
            exact_rate * 100
        ),
        "avg_time_sec": (
            ""
            if avg_time is None
            else avg_time
        ),
    })


    print()
    print(model_name)
    print("-" * len(model_name))

    print(
        f"Samples:       "
        f"{s['samples']}"
    )

    print(
        f"Missing:       "
        f"{s['missing']}"
    )

    print(
        f"Empty:         "
        f"{s['empty']}"
    )

    print(
        f"GT characters: "
        f"{s['gt_chars']}"
    )

    print(
        f"Edit distance: "
        f"{s['edit_distance']}"
    )

    print(
        f"CER:           "
        f"{cer:.4f} "
        f"({cer * 100:.2f}%)"
    )

    print(
        f"Exact Match:   "
        f"{s['exact']}/{s['samples']} "
        f"({exact_rate * 100:.2f}%)"
    )


    if avg_time is not None:

        print(
            f"Avg time:      "
            f"{avg_time:.4f} sec/image"
        )

    else:

        print(
            "Avg time:      N/A"
        )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

with open(
    SUMMARY_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    fieldnames = [
        "model",
        "samples",
        "missing",
        "empty_predictions",
        "gt_characters",
        "edit_distance",
        "CER",
        "CER_percent",
        "exact_matches",
        "exact_match_percent",
        "avg_time_sec",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ============================================================
# SAVE DETAILS CSV
# ============================================================

with open(
    DETAILS_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    fieldnames = [
        "number",
        "original_file",
        "test_file",
        "model",
        "ground_truth",
        "prediction",
        "normalized_gt",
        "normalized_prediction",
        "edit_distance",
        "gt_chars",
        "sample_CER",
        "exact_match",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        detail_rows
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print(
    "Summary saved:"
)

print(
    SUMMARY_CSV
)

print()
print(
    "Detailed results saved:"
)

print(
    DETAILS_CSV
)