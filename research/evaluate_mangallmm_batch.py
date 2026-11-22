from .benchmark_paths import BASE_DIR, XML_PATH
import csv
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

# XML_PATH is supplied by benchmark_paths.

RESULT_DIR = BASE_DIR / 'mangallmm_tiled_batch'

PAGE_CSV = BASE_DIR / 'mangallmm_page_metrics.csv'

SUMMARY_TXT = BASE_DIR / 'mangallmm_summary.txt'


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC",
        str(text or "")
    )

    text = re.sub(
        r"\s+",
        "",
        text
    )

    return text


# =========================================================
# LEVENSHTEIN
# =========================================================

def edit_distance(a, b):

    previous = list(
        range(len(b) + 1)
    )

    for i, ca in enumerate(a, 1):

        current = [i]

        for j, cb in enumerate(b, 1):

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1]
                    + (ca != cb)
                )
            )

        previous = current

    return previous[-1]


# =========================================================
# IOU
# =========================================================

def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# =========================================================
# LOAD XML GROUND TRUTH
# =========================================================

tree = ET.parse(XML_PATH)
root = tree.getroot()

GT_BY_PAGE = {}

for page in root.findall(".//page"):

    page_index = int(
        page.attrib["index"]
    )

    regions = []

    for item in page.findall("text"):

        text = (
            item.text or ""
        ).strip()

        if not text:
            continue

        regions.append({
            "text": text,

            "bbox": [
                int(item.attrib["xmin"]),
                int(item.attrib["ymin"]),
                int(item.attrib["xmax"]),
                int(item.attrib["ymax"]),
            ]
        })

    GT_BY_PAGE[
        page_index
    ] = regions


# =========================================================
# MATCH ONE PAGE
# =========================================================

def evaluate_page(
    gt,
    pred,
    threshold
):

    candidates = []

    for gi, g in enumerate(gt):

        for pi, p in enumerate(pred):

            score = iou(
                g["bbox"],
                p["bbox"]
            )

            if score >= threshold:

                candidates.append(
                    (
                        score,
                        gi,
                        pi
                    )
                )


    # Highest IoU first
    candidates.sort(
        reverse=True
    )


    used_gt = set()
    used_pred = set()

    matches = []


    for score, gi, pi in candidates:

        if gi in used_gt:
            continue

        if pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

        matches.append(
            (
                gi,
                pi,
                score
            )
        )


    tp = len(matches)

    fp = len(pred) - tp
    fn = len(gt) - tp


    # -----------------------------------------
    # OCR QUALITY ON MATCHED REGIONS
    # -----------------------------------------

    edits = 0
    chars = 0
    exact = 0


    for gi, pi, score in matches:

        ref = normalize_text(
            gt[gi]["text"]
        )

        hyp = normalize_text(
            pred[pi]["text"]
        )

        edits += edit_distance(
            ref,
            hyp
        )

        chars += len(ref)

        if ref == hyp:
            exact += 1


    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,

        "edits": edits,
        "chars": chars,
        "exact": exact,
        "matches": len(matches),
    }


# =========================================================
# FIND COMPLETED PAGES
# =========================================================

result_files = sorted(
    RESULT_DIR.glob(
        "*_result.json"
    )
)


if not result_files:

    raise RuntimeError(
        "No completed result JSON files found."
    )


print(
    "Completed pages found:",
    len(result_files)
)

print()


# =========================================================
# RUN BOTH IOU THRESHOLDS
# =========================================================

thresholds = [
    0.30,
    0.50
]


all_summary = {}

page_rows = []


for threshold in thresholds:

    TOTAL_GT = 0
    TOTAL_PRED = 0

    TOTAL_TP = 0
    TOTAL_FP = 0
    TOTAL_FN = 0

    TOTAL_EDITS = 0
    TOTAL_CHARS = 0

    TOTAL_EXACT = 0
    TOTAL_MATCHES = 0

    TOTAL_TIME = 0.0

    pages_count = 0


    print(
        "=" * 65
    )

    print(
        "IOU THRESHOLD:",
        threshold
    )

    print(
        "=" * 65
    )


    for result_file in result_files:

        with open(
            result_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        page_number = int(
            data["page"]
        )


        gt = GT_BY_PAGE.get(
            page_number,
            []
        )


        pred = []

        for item in data.get(
            "detections",
            []
        ):

            pred.append({
                "text": item.get(
                    "text_content",
                    ""
                ),

                "bbox": item[
                    "bbox_page"
                ],
            })


        result = evaluate_page(
            gt,
            pred,
            threshold
        )


        TOTAL_GT += len(gt)
        TOTAL_PRED += len(pred)

        TOTAL_TP += result["tp"]
        TOTAL_FP += result["fp"]
        TOTAL_FN += result["fn"]

        TOTAL_EDITS += result[
            "edits"
        ]

        TOTAL_CHARS += result[
            "chars"
        ]

        TOTAL_EXACT += result[
            "exact"
        ]

        TOTAL_MATCHES += result[
            "matches"
        ]


        page_time = float(
            data.get(
                "time_sec",
                0
            )
        )

        TOTAL_TIME += page_time

        pages_count += 1


        page_precision = (
            result["tp"]
            / (
                result["tp"]
                + result["fp"]
            )
            if (
                result["tp"]
                + result["fp"]
            )
            else 0
        )


        page_recall = (
            result["tp"]
            / (
                result["tp"]
                + result["fn"]
            )
            if (
                result["tp"]
                + result["fn"]
            )
            else 0
        )


        page_f1 = (
            2
            * page_precision
            * page_recall
            / (
                page_precision
                + page_recall
            )
            if (
                page_precision
                + page_recall
            )
            else 0
        )


        page_cer = (
            result["edits"]
            / result["chars"]
            if result["chars"]
            else 0
        )


        page_rows.append({
            "iou_threshold": threshold,
            "page": page_number,

            "gt_regions": len(gt),
            "pred_regions": len(pred),

            "tp": result["tp"],
            "fp": result["fp"],
            "fn": result["fn"],

            "precision": page_precision,
            "recall": page_recall,
            "f1": page_f1,

            "cer_matched": page_cer,

            "exact_matched": (
                result["exact"]
                / result["matches"]
                if result["matches"]
                else 0
            ),

            "time_sec": page_time,
        })


        print(
            f"Page {page_number:03d} | "
            f"GT {len(gt):3d} | "
            f"PRED {len(pred):3d} | "
            f"TP {result['tp']:3d} | "
            f"F1 {page_f1:.3f} | "
            f"CER {page_cer:.3f}"
        )


    # =====================================================
    # MICRO-AVERAGED METRICS
    # =====================================================

    precision = (
        TOTAL_TP
        / (
            TOTAL_TP
            + TOTAL_FP
        )
        if (
            TOTAL_TP
            + TOTAL_FP
        )
        else 0
    )


    recall = (
        TOTAL_TP
        / (
            TOTAL_TP
            + TOTAL_FN
        )
        if (
            TOTAL_TP
            + TOTAL_FN
        )
        else 0
    )


    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0
    )


    cer = (
        TOTAL_EDITS
        / TOTAL_CHARS
        if TOTAL_CHARS
        else 0
    )


    exact_matched = (
        TOTAL_EXACT
        / TOTAL_MATCHES
        if TOTAL_MATCHES
        else 0
    )


    end_exact_recall = (
        TOTAL_EXACT
        / TOTAL_GT
        if TOTAL_GT
        else 0
    )


    avg_time = (
        TOTAL_TIME
        / pages_count
        if pages_count
        else 0
    )


    summary = {
        "pages": pages_count,

        "gt": TOTAL_GT,
        "pred": TOTAL_PRED,

        "tp": TOTAL_TP,
        "fp": TOTAL_FP,
        "fn": TOTAL_FN,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "cer": cer,

        "exact_matched": (
            exact_matched
        ),

        "end_exact_recall": (
            end_exact_recall
        ),

        "avg_time": avg_time,
        "total_time": TOTAL_TIME,
    }


    all_summary[
        threshold
    ] = summary


    print()
    print(
        "--- AGGREGATE ---"
    )

    print(
        "Pages:",
        pages_count
    )

    print(
        "GT regions:",
        TOTAL_GT
    )

    print(
        "Predicted regions:",
        TOTAL_PRED
    )

    print()

    print(
        "TP:",
        TOTAL_TP
    )

    print(
        "FP:",
        TOTAL_FP
    )

    print(
        "FN:",
        TOTAL_FN
    )

    print()

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print()

    print(
        f"CER on matched regions: "
        f"{cer:.4f}"
    )

    print(
        f"Exact Match on matched: "
        f"{exact_matched * 100:.2f}%"
    )

    print(
        f"End-to-end exact recall: "
        f"{end_exact_recall * 100:.2f}%"
    )

    print()

    print(
        f"Average page time: "
        f"{avg_time:.2f} sec"
    )

    print()


# =========================================================
# SAVE PER-PAGE CSV
# =========================================================

with open(
    PAGE_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "iou_threshold",
            "page",
            "gt_regions",
            "pred_regions",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "cer_matched",
            "exact_matched",
            "time_sec",
        ]
    )

    writer.writeheader()
    writer.writerows(
        page_rows
    )


# =========================================================
# SAVE SUMMARY TXT
# =========================================================

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as f:

    for threshold in thresholds:

        s = all_summary[
            threshold
        ]

        f.write(
            "=" * 60
            + "\n"
        )

        f.write(
            f"IoU threshold: "
            f"{threshold}\n"
        )

        f.write(
            "=" * 60
            + "\n"
        )

        f.write(
            f"Pages: {s['pages']}\n"
        )

        f.write(
            f"GT regions: {s['gt']}\n"
        )

        f.write(
            f"Predicted regions: "
            f"{s['pred']}\n"
        )

        f.write(
            f"TP: {s['tp']}\n"
        )

        f.write(
            f"FP: {s['fp']}\n"
        )

        f.write(
            f"FN: {s['fn']}\n\n"
        )

        f.write(
            f"Precision: "
            f"{s['precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{s['recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{s['f1']:.4f}\n\n"
        )

        f.write(
            f"CER matched: "
            f"{s['cer']:.4f}\n"
        )

        f.write(
            f"Exact Match matched: "
            f"{s['exact_matched'] * 100:.2f}%\n"
        )

        f.write(
            f"End-to-end exact recall: "
            f"{s['end_exact_recall'] * 100:.2f}%\n\n"
        )

        f.write(
            f"Average page time: "
            f"{s['avg_time']:.2f} sec\n\n"
        )


print()
print("=" * 65)
print("DONE")
print("=" * 65)

print(
    "Per-page metrics:",
    PAGE_CSV
)

print(
    "Summary:",
    SUMMARY_TXT
)