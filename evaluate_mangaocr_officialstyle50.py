import csv
import json
import re
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\mangalmm_test\official_mangaocr_test50"
)

MANIFEST = BASE / "manifest.jsonl"

RESULTS = (
    BASE
    / "mangallmm_results"
    / "results.jsonl"
)

OUT_TXT = (
    BASE
    / "mangallmm_results"
    / "officialstyle_summary_50.txt"
)

OUT_CSV = (
    BASE
    / "mangallmm_results"
    / "officialstyle_per_sample_50.csv"
)


# ============================================================
# OFFICIAL EVALUATION SETTING
# ============================================================

IOU_THRESHOLD = 0.5


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


# ============================================================
# TEXT NORMALIZATION
#
# Повторяем нормализацию из официального eval_OCR.ipynb
# ============================================================

def normalize_repeated_symbols(text):

    text = re.sub(
        r'([~\～\〜\-ー]+)',
        lambda m: m.group(1)[0],
        text
    )

    text = re.sub(
        r'[~～〜]',
        '~',
        text
    )

    text = re.sub(
        r'[-ー]',
        '-',
        text
    )

    return text


def normalize_punctuation(text):

    conversion_map = {
        "！": "!",
        "？": "?",
        "…": "...",
    }

    pattern = "|".join(
        map(
            re.escape,
            conversion_map.keys()
        )
    )

    text = re.sub(
        pattern,
        lambda m: conversion_map[
            m.group()
        ],
        text
    )

    text = re.sub(
        r'[・･.]',
        '・',
        text
    )

    return text


def normalize_text(text):

    text = str(
        text or ""
    )

    # whitespace / Japanese full-width space
    text = re.sub(
        r"[\s\u3000]+",
        "",
        text
    )

    text = normalize_repeated_symbols(
        text
    )

    text = normalize_punctuation(
        text
    )

    return text


# ============================================================
# LEVENSHTEIN
# ============================================================

def edit_distance(a, b):

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)


    previous = list(
        range(
            len(b) + 1
        )
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
                current[j - 1]
                + 1
            )

            deletion = (
                previous[j]
                + 1
            )

            substitution = (
                previous[j - 1]
                + (
                    0
                    if ca == cb
                    else 1
                )
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
# NORMALIZED EDIT SIMILARITY
#
# Официальный evaluator:
#
# 1 - edit_distance / max(len(pred), len(gt))
#
# ============================================================

def ned_similarity(gt, pred):

    gt = normalize_text(gt)
    pred = normalize_text(pred)

    denominator = max(
        len(gt),
        len(pred)
    )

    if denominator == 0:
        return 1.0

    distance = edit_distance(
        pred,
        gt
    )

    score = (
        1.0
        - distance / denominator
    )

    return max(
        0.0,
        min(
            1.0,
            score
        )
    )


# ============================================================
# BBOX PARSER
# ============================================================

PAIR_PATTERN = re.compile(
    r'"bbox_2d"\s*:\s*'
    r'\[\s*'
    r'(-?\d+)\s*,\s*'
    r'(-?\d+)\s*,\s*'
    r'(-?\d+)\s*,\s*'
    r'(-?\d+)\s*'
    r'\]'
    r'.*?'
    r'"text_content"\s*:\s*'
    r'"([^"]*)"',
    flags=re.DOTALL
)


def parse_raw_pairs(raw_text):

    if not raw_text:
        return []


    text = str(raw_text)

    # официальный inference отдельно
    # обрабатывает escaped quotes
    text = text.replace(
        '\\"',
        '"'
    )


    items = []


    for m in PAIR_PATTERN.finditer(
        text
    ):

        bbox = [
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
        ]

        content = m.group(5)


        items.append({
            "bbox": bbox,
            "text": content,
        })


    return items


# ============================================================
# VALID BBOX
# ============================================================

def valid_bbox(bbox):

    if (
        not isinstance(
            bbox,
            (list, tuple)
        )
        or len(bbox) != 4
    ):
        return False


    try:

        x1, y1, x2, y2 = [
            float(x)
            for x in bbox
        ]

    except Exception:
        return False


    if x2 <= x1:
        return False

    if y2 <= y1:
        return False


    return True


# ============================================================
# IOU
#
# bbox_2d в официальном inference
# преобразуется в прямоугольный quad.
# Для таких прямоугольников обычный bbox IoU эквивалентен.
# ============================================================

def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = [
        float(x)
        for x in a
    ]

    bx1, by1, bx2, by2 = [
        float(x)
        for x in b
    ]


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )


    intersection = (
        iw * ih
    )


    area_a = (
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        * (by2 - by1)
    )


    union = (
        area_a
        + area_b
        - intersection
    )


    if union <= 0:
        return 0.0


    return (
        intersection
        / union
    )


# ============================================================
# PREPARE GT
# ============================================================

def get_gt_items(
    manifest_row
):

    raw_gt = manifest_row.get(
        "ground_truth",
        ""
    )


    items = parse_raw_pairs(
        raw_gt
    )


    prepared = []


    for item in items:

        if not valid_bbox(
            item["bbox"]
        ):
            continue


        prepared.append({
            "bbox": item["bbox"],

            "text": normalize_text(
                item["text"]
            ),
        })


    return prepared


# ============================================================
# PREPARE PREDICTIONS
# ============================================================

def get_pred_items(
    result_row
):

    items = []


    # Сначала используем уже распарсенные detections
    detections = result_row.get(
        "detections",
        []
    )


    if isinstance(
        detections,
        list
    ):

        for det in detections:

            bbox = det.get(
                "bbox_2d"
            )

            text = det.get(
                "text_content",
                ""
            )

            if valid_bbox(
                bbox
            ):

                items.append({
                    "bbox": bbox,

                    "text": (
                        normalize_text(
                            text
                        )
                    ),
                })


    # Если parser во время inference ничего не нашёл,
    # пробуем ещё раз прямо по raw output.
    if not items:

        raw = result_row.get(
            "prediction_raw",
            ""
        )

        raw_items = (
            parse_raw_pairs(
                raw
            )
        )


        for item in raw_items:

            if not valid_bbox(
                item["bbox"]
            ):
                continue

            items.append({
                "bbox": item["bbox"],

                "text": (
                    normalize_text(
                        item["text"]
                    )
                ),
            })


    # ========================================================
    # OFFICIAL DEDUPLICATION
    #
    # В официальном notebook:
    # если одно и то же слово появляется >= 10 раз
    # на странице, все такие повторения удаляются.
    # ========================================================

    counts = Counter(
        item["text"]
        for item in items
    )


    filtered = [
        item
        for item in items

        if counts[
            item["text"]
        ] < 10
    ]


    return filtered


# ============================================================
# MATCH ONE SAMPLE
#
# Повторяем логику официального evaluator:
#
# - one-to-one spatial matching
# - IoU > 0.5
# - Detection score = 1
# - E2E score = NED similarity
# ============================================================

def evaluate_sample(
    gt_items,
    pred_items
):

    gt_used = [
        False
        for _ in gt_items
    ]

    pred_used = [
        False
        for _ in pred_items
    ]


    spatial_matches = 0

    ned_sum = 0.0


    matched_pairs = []


    for gi, gt in enumerate(
        gt_items
    ):

        for pi, pred in enumerate(
            pred_items
        ):

            if gt_used[gi]:
                continue

            if pred_used[pi]:
                continue


            overlap = bbox_iou(
                gt["bbox"],
                pred["bbox"]
            )


            # Официальный код использует >
            # а не >=
            if overlap > IOU_THRESHOLD:

                gt_used[gi] = True
                pred_used[pi] = True

                spatial_matches += 1


                ned = ned_similarity(
                    gt["text"],
                    pred["text"]
                )


                ned_sum += ned


                matched_pairs.append({
                    "gt_index": gi,
                    "pred_index": pi,

                    "iou": overlap,
                    "ned": ned,

                    "gt_text": gt["text"],
                    "pred_text": (
                        pred["text"]
                    ),
                })


    return {
        "gt_count": len(
            gt_items
        ),

        "pred_count": len(
            pred_items
        ),

        "spatial_matches": (
            spatial_matches
        ),

        "ned_sum": ned_sum,

        "pairs": matched_pairs,
    }


# ============================================================
# LOAD
# ============================================================

manifest_rows = load_jsonl(
    MANIFEST
)

result_rows = load_jsonl(
    RESULTS
)


manifest_map = {
    row["id"]: row
    for row in manifest_rows
}

result_map = {
    row["id"]: row
    for row in result_rows
}


ids = [
    row["id"]
    for row in manifest_rows
]


print(
    "GT samples:",
    len(manifest_map)
)

print(
    "Prediction samples:",
    len(result_map)
)

print(
    "Matched sample IDs:",
    len(
        set(manifest_map)
        & set(result_map)
    )
)


# ============================================================
# AGGREGATE
# ============================================================

TOTAL_GT = 0
TOTAL_PRED = 0

TOTAL_SPATIAL_MATCHES = 0

TOTAL_NED = 0.0


per_sample_rows = []


raw_pred_count = 0

samples_with_zero_pred = 0


for sample_id in ids:

    manifest_row = manifest_map[
        sample_id
    ]

    result_row = result_map.get(
        sample_id,
        {}
    )


    raw_detections = result_row.get(
        "detections",
        []
    )


    if isinstance(
        raw_detections,
        list
    ):

        raw_pred_count += len(
            raw_detections
        )


    gt_items = get_gt_items(
        manifest_row
    )

    pred_items = get_pred_items(
        result_row
    )


    if len(pred_items) == 0:
        samples_with_zero_pred += 1


    evaluation = (
        evaluate_sample(
            gt_items,
            pred_items
        )
    )


    gt_count = evaluation[
        "gt_count"
    ]

    pred_count = evaluation[
        "pred_count"
    ]

    spatial = evaluation[
        "spatial_matches"
    ]

    ned_sum = evaluation[
        "ned_sum"
    ]


    TOTAL_GT += gt_count

    TOTAL_PRED += pred_count

    TOTAL_SPATIAL_MATCHES += (
        spatial
    )

    TOTAL_NED += ned_sum


    # --------------------------------------------------------
    # Per-sample DET
    # --------------------------------------------------------

    det_precision = (
        spatial / pred_count
        if pred_count
        else (
            1.0
            if gt_count == 0
            else 0.0
        )
    )


    det_recall = (
        spatial / gt_count
        if gt_count
        else 1.0
    )


    det_hmean = (
        0.0
        if (
            det_precision
            + det_recall
        ) == 0

        else (
            2
            * det_precision
            * det_recall
            / (
                det_precision
                + det_recall
            )
        )
    )


    # --------------------------------------------------------
    # Per-sample E2E
    # --------------------------------------------------------

    e2e_precision = (
        ned_sum / pred_count
        if pred_count
        else (
            1.0
            if gt_count == 0
            else 0.0
        )
    )


    e2e_recall = (
        ned_sum / gt_count
        if gt_count
        else 1.0
    )


    e2e_hmean = (
        0.0
        if (
            e2e_precision
            + e2e_recall
        ) == 0

        else (
            2
            * e2e_precision
            * e2e_recall
            / (
                e2e_precision
                + e2e_recall
            )
        )
    )


    per_sample_rows.append({
        "id": sample_id,

        "gt_regions": gt_count,

        "pred_regions": pred_count,

        "spatial_matches": spatial,

        "det_precision": (
            det_precision
        ),

        "det_recall": (
            det_recall
        ),

        "det_hmean": (
            det_hmean
        ),

        "ned_sum": ned_sum,

        "e2e_precision": (
            e2e_precision
        ),

        "e2e_recall": (
            e2e_recall
        ),

        "e2e_hmean": (
            e2e_hmean
        ),
    })


# ============================================================
# GLOBAL DETECTION METRICS
# ============================================================

DET_PRECISION = (
    TOTAL_SPATIAL_MATCHES
    / TOTAL_PRED
    if TOTAL_PRED
    else 0.0
)


DET_RECALL = (
    TOTAL_SPATIAL_MATCHES
    / TOTAL_GT
    if TOTAL_GT
    else 0.0
)


DET_HMEAN = (
    0.0

    if (
        DET_PRECISION
        + DET_RECALL
    ) == 0

    else (
        2
        * DET_PRECISION
        * DET_RECALL
        / (
            DET_PRECISION
            + DET_RECALL
        )
    )
)


# ============================================================
# GLOBAL END-TO-END METRICS
# ============================================================

E2E_PRECISION = (
    TOTAL_NED
    / TOTAL_PRED
    if TOTAL_PRED
    else 0.0
)


E2E_RECALL = (
    TOTAL_NED
    / TOTAL_GT
    if TOTAL_GT
    else 0.0
)


E2E_HMEAN = (
    0.0

    if (
        E2E_PRECISION
        + E2E_RECALL
    ) == 0

    else (
        2
        * E2E_PRECISION
        * E2E_RECALL
        / (
            E2E_PRECISION
            + E2E_RECALL
        )
    )
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 70)

print(
    "MANGALMM OFFICIAL-STYLE MangaOCR TEST-50"
)

print("=" * 70)

print(
    "IoU threshold:",
    IOU_THRESHOLD
)

print()

print(
    "Samples:",
    len(ids)
)

print(
    "Samples with zero predictions:",
    samples_with_zero_pred
)

print()

print(
    "GT regions:",
    TOTAL_GT
)

print(
    "Predicted regions after dedup:",
    TOTAL_PRED
)

print(
    "Spatial matches:",
    TOTAL_SPATIAL_MATCHES
)

print()

print(
    "DET_PRH:"
)

print(
    f"Precision = "
    f"{DET_PRECISION * 100:.2f}%"
)

print(
    f"Recall    = "
    f"{DET_RECALL * 100:.2f}%"
)

print(
    f"Hmean     = "
    f"{DET_HMEAN * 100:.2f}%"
)

print()

print(
    "E2E_PRH (with NED):"
)

print(
    f"Precision = "
    f"{E2E_PRECISION * 100:.2f}%"
)

print(
    f"Recall    = "
    f"{E2E_RECALL * 100:.2f}%"
)

print(
    f"Hmean     = "
    f"{E2E_HMEAN * 100:.2f}%"
)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "gt_regions",
            "pred_regions",
            "spatial_matches",

            "det_precision",
            "det_recall",
            "det_hmean",

            "ned_sum",

            "e2e_precision",
            "e2e_recall",
            "e2e_hmean",
        ]
    )

    writer.writeheader()

    writer.writerows(
        per_sample_rows
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MangaLMM official-style MangaOCR test-50\n"
    )

    f.write(
        "=" * 60
        + "\n"
    )

    f.write(
        f"IoU threshold: "
        f"{IOU_THRESHOLD}\n"
    )

    f.write(
        f"Samples: "
        f"{len(ids)}\n"
    )

    f.write(
        f"Samples with zero predictions: "
        f"{samples_with_zero_pred}\n\n"
    )


    f.write(
        f"GT regions: "
        f"{TOTAL_GT}\n"
    )

    f.write(
        f"Pred regions: "
        f"{TOTAL_PRED}\n"
    )

    f.write(
        f"Spatial matches: "
        f"{TOTAL_SPATIAL_MATCHES}\n\n"
    )


    f.write(
        "DET_PRH\n"
    )

    f.write(
        f"Precision: "
        f"{DET_PRECISION * 100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{DET_RECALL * 100:.2f}%\n"
    )

    f.write(
        f"Hmean: "
        f"{DET_HMEAN * 100:.2f}%\n\n"
    )


    f.write(
        "E2E_PRH (NED)\n"
    )

    f.write(
        f"Precision: "
        f"{E2E_PRECISION * 100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{E2E_RECALL * 100:.2f}%\n"
    )

    f.write(
        f"Hmean: "
        f"{E2E_HMEAN * 100:.2f}%\n"
    )


print()
print("=" * 70)

print("DONE")

print(
    "Summary:",
    OUT_TXT
)

print(
    "Per-sample CSV:",
    OUT_CSV
)