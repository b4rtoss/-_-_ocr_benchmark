from benchmark_paths import BASE_DIR
import json
import re
import math
from pathlib import Path

from PIL import Image
from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info


# ============================================================
# PATHS
# ============================================================

BASE = BASE_DIR / 'official_mangaocr_test50'

MANIFEST = BASE / "manifest.jsonl"

RESULTS = (
    BASE
    / "mangallmm_results"
    / "results.jsonl"
)

OUT = (
    BASE
    / "mangallmm_results"
    / "scaled_eval_50.txt"
)


# ============================================================
# USE THE SAME SETTINGS
# AS DURING INFERENCE
# ============================================================

PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

IOU_THRESHOLD = 0.50


print("Loading processor...")

processor = Qwen2_5_VLProcessor.from_pretrained(
    PROCESSOR_ID,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

print("Processor loaded.")


# ============================================================
# JSONL
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

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


# ============================================================
# PARSE GT
# ============================================================

PATTERN = re.compile(
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
    re.DOTALL
)


def parse_gt(raw):

    if not raw:
        return []

    raw = raw.replace(
        '\\"',
        '"'
    )

    items = []

    for m in PATTERN.finditer(raw):

        items.append({
            "bbox": [
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
            ],

            "text": m.group(5),
        })

    return items


# ============================================================
# IoU
# ============================================================

def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    inter = iw * ih

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# TEXT
# ============================================================

def normalize_text(text):

    text = str(
        text or ""
    )

    return re.sub(
        r"[\s\u3000]+",
        "",
        text
    )


def edit_distance(a, b):

    previous = list(
        range(
            len(b) + 1
        )
    )

    for i, ca in enumerate(
        a,
        1
    ):

        current = [i]

        for j, cb in enumerate(
            b,
            1
        ):

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


def ned(gt, pred):

    gt = normalize_text(gt)
    pred = normalize_text(pred)

    denominator = max(
        len(gt),
        len(pred)
    )

    if denominator == 0:
        return 1.0

    return max(
        0.0,
        1.0
        - (
            edit_distance(
                gt,
                pred
            )
            / denominator
        )
    )


# ============================================================
# RECOVER THE ACTUAL QWEN INPUT SIZE
# ============================================================

def get_processed_size(image):

    messages = [
        {
            "role": "user",

            "content": [
                {
                    "type": "image",
                    "image": image,
                }
            ],
        }
    ]

    image_inputs, _ = (
        process_vision_info(
            messages
        )
    )

    processed = (
        processor.image_processor(
            images=image_inputs,
            return_tensors="pt"
        )
    )

    grid = processed[
        "image_grid_thw"
    ][0]

    # Qwen2.5-VL patch = 14 px
    patch_size = int(
        processor.image_processor.patch_size
    )

    resized_h = (
        int(grid[1])
        * patch_size
    )

    resized_w = (
        int(grid[2])
        * patch_size
    )

    return (
        resized_w,
        resized_h
    )


# ============================================================
# SCALE PREDICTIONS
# ============================================================

def prepare_predictions(
    result_row,
    original_w,
    original_h,
    resized_w,
    resized_h
):

    scale_x = (
        original_w
        / resized_w
    )

    scale_y = (
        original_h
        / resized_h
    )

    output = []

    for det in result_row.get(
        "detections",
        []
    ):

        bbox = det.get(
            "bbox_2d"
        )

        if (
            not bbox
            or len(bbox) != 4
        ):
            continue

        x1, y1, x2, y2 = bbox

        scaled_bbox = [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ]

        output.append({
            "bbox": scaled_bbox,

            "text": det.get(
                "text_content",
                ""
            ),
        })

    return output


# ============================================================
# GREEDY MATCHING
# ============================================================

def match_regions(
    gt,
    pred
):

    candidates = []

    for gi, g in enumerate(gt):

        for pi, p in enumerate(pred):

            score = iou(
                g["bbox"],
                p["bbox"]
            )

            if score >= IOU_THRESHOLD:

                candidates.append(
                    (
                        score,
                        gi,
                        pi
                    )
                )


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

    return matches


# ============================================================
# LOAD DATA
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


print()
print(
    "GT samples:",
    len(manifest_map)
)

print(
    "Prediction samples:",
    len(result_map)
)


# ============================================================
# GLOBAL COUNTERS
# ============================================================

TOTAL_GT = 0
TOTAL_PRED = 0

TOTAL_MATCH = 0

TOTAL_NED = 0.0

ZERO_PRED = 0


lines = []


# ============================================================
# LOOP
# ============================================================

for sample_number, manifest_row in enumerate(
    manifest_rows,
    start=1
):

    sample_id = manifest_row["id"]

    result_row = result_map.get(
        sample_id,
        {}
    )


    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    gt = parse_gt(
        manifest_row.get(
            "ground_truth",
            ""
        )
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_path = Path(
        manifest_row["image"]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    original_w, original_h = (
        image.size
    )


    # --------------------------------------------------------
    # ACTUAL MODEL INPUT SIZE
    # --------------------------------------------------------

    resized_w, resized_h = (
        get_processed_size(
            image
        )
    )


    # --------------------------------------------------------
    # SCALE PREDICTIONS BACK TO ORIGINAL PAGE
    # --------------------------------------------------------

    pred = prepare_predictions(
        result_row,

        original_w,
        original_h,

        resized_w,
        resized_h
    )


    if len(pred) == 0:
        ZERO_PRED += 1


    # --------------------------------------------------------
    # MATCH
    # --------------------------------------------------------

    matches = match_regions(
        gt,
        pred
    )


    ned_sum = 0.0

    for gi, pi, score in matches:

        ned_sum += ned(
            gt[gi]["text"],
            pred[pi]["text"]
        )


    TOTAL_GT += len(gt)

    TOTAL_PRED += len(pred)

    TOTAL_MATCH += len(matches)

    TOTAL_NED += ned_sum


    print(
        f"[{sample_number:02d}/50] "
        f"{sample_id} | "
        f"{original_w}x{original_h} -> "
        f"{resized_w}x{resized_h} | "
        f"GT {len(gt)} | "
        f"PRED {len(pred)} | "
        f"MATCH {len(matches)}"
    )


# ============================================================
# METRICS
# ============================================================

det_precision = (
    TOTAL_MATCH / TOTAL_PRED
    if TOTAL_PRED
    else 0
)

det_recall = (
    TOTAL_MATCH / TOTAL_GT
    if TOTAL_GT
    else 0
)

det_f1 = (
    2
    * det_precision
    * det_recall
    / (
        det_precision
        + det_recall
    )

    if (
        det_precision
        + det_recall
    )
    else 0
)


e2e_precision = (
    TOTAL_NED / TOTAL_PRED
    if TOTAL_PRED
    else 0
)

e2e_recall = (
    TOTAL_NED / TOTAL_GT
    if TOTAL_GT
    else 0
)

e2e_hmean = (
    2
    * e2e_precision
    * e2e_recall
    / (
        e2e_precision
        + e2e_recall
    )

    if (
        e2e_precision
        + e2e_recall
    )
    else 0
)


# ============================================================
# PRINT FINAL
# ============================================================

print()
print("=" * 70)

print(
    "SCALED COORDINATE EVALUATION"
)

print("=" * 70)

print(
    "Samples:",
    len(manifest_rows)
)

print(
    "Zero predictions:",
    ZERO_PRED
)

print()

print(
    "GT regions:",
    TOTAL_GT
)

print(
    "Predicted regions:",
    TOTAL_PRED
)

print(
    "Spatial matches:",
    TOTAL_MATCH
)

print()

print(
    "DETECTION:"
)

print(
    f"Precision: "
    f"{det_precision * 100:.2f}%"
)

print(
    f"Recall:    "
    f"{det_recall * 100:.2f}%"
)

print(
    f"Hmean:     "
    f"{det_f1 * 100:.2f}%"
)

print()

print(
    "END-TO-END NED:"
)

print(
    f"Precision: "
    f"{e2e_precision * 100:.2f}%"
)

print(
    f"Recall:    "
    f"{e2e_recall * 100:.2f}%"
)

print(
    f"Hmean:     "
    f"{e2e_hmean * 100:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

with open(
    OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Samples: {len(manifest_rows)}\n"
    )

    f.write(
        f"Zero predictions: {ZERO_PRED}\n\n"
    )

    f.write(
        f"GT regions: {TOTAL_GT}\n"
    )

    f.write(
        f"Pred regions: {TOTAL_PRED}\n"
    )

    f.write(
        f"Spatial matches: {TOTAL_MATCH}\n\n"
    )

    f.write(
        "DETECTION\n"
    )

    f.write(
        f"Precision: "
        f"{det_precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{det_recall * 100:.2f}%\n"
    )

    f.write(
        f"Hmean: "
        f"{det_f1 * 100:.2f}%\n\n"
    )

    f.write(
        "END-TO-END NED\n"
    )

    f.write(
        f"Precision: "
        f"{e2e_precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{e2e_recall * 100:.2f}%\n"
    )

    f.write(
        f"Hmean: "
        f"{e2e_hmean * 100:.2f}%\n"
    )


print()
print(
    "Saved:",
    OUT
)