from .benchmark_paths import BASE_DIR, IMAGE_DIR
import re
import gc
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from qwen_vl_utils import process_vision_info

from transformers import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)


# =========================================================
# SETTINGS
# =========================================================

# IMAGE_DIR is supplied by benchmark_paths.

OUTPUT_DIR = BASE_DIR / 'mangallmm_tiled_batch'

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Process 20 pages in this experiment.
PAGES = list(range(8, 28))


MODEL_ID = "hal-utokyo/MangaLMM"

PROCESSOR_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)


OCR_PROMPT = (
    "Please perform OCR on this image and output "
    "the recognized Japanese text along with its "
    "position (grounding)."
)


MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

MAX_NEW_TOKENS = 320

OVERLAP_RATIO = 0.08


# =========================================================
# CUDA
# =========================================================

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available"
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

torch.cuda.empty_cache()


# =========================================================
# 4-BIT
# =========================================================

quantization_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=(
            torch.float16
        ),

        bnb_4bit_use_double_quant=True,
    )
)


# =========================================================
# PROCESSOR
# =========================================================

print()
print("Loading processor...")

processor = (
    Qwen2_5_VLProcessor
    .from_pretrained(
        PROCESSOR_ID,

        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
)

print("Processor loaded.")


# =========================================================
# MODEL
# =========================================================

print()
print("Loading MangaLMM...")

model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,

        quantization_config=(
            quantization_config
        ),

        device_map={"": 0},

        torch_dtype=torch.float16,

        low_cpu_mem_usage=True,
    )
)

model.eval()

print()
print(
    "MangaLMM loaded successfully!"
)

print(
    "Model memory footprint:",
    round(
        model.get_memory_footprint()
        / 1024**3,
        2
    ),
    "GB"
)


# =========================================================
# FIND THE JPG
# =========================================================

def find_page_image(page_number):

    base = f"{page_number:03d}"

    for ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]:

        candidate = (
            IMAGE_DIR
            / f"{base}{ext}"
        )

        if candidate.exists():
            return candidate

    return None


# =========================================================
# PARSER
# =========================================================

def parse_output(text):

    if not text:
        return []

    clean = (
        text
        .replace("```json", "")
        .replace("```", "")
    )

    pattern = re.compile(
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

    results = []

    for m in pattern.finditer(clean):

        results.append({
            "bbox_2d": [
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
            ],

            "text_content": (
                m.group(5)
            ),
        })

    return results


# =========================================================
# RESIZE
# =========================================================

def resize_for_model(image):

    w, h = image.size

    area = w * h

    if area <= MAX_PIXELS:

        scale = 1.0

    else:

        scale = math.sqrt(
            MAX_PIXELS / area
        )


    new_w = max(
        28,
        int(w * scale) // 28 * 28
    )

    new_h = max(
        28,
        int(h * scale) // 28 * 28
    )


    while new_w * new_h > MAX_PIXELS:

        if new_w >= new_h:
            new_w -= 28

        else:
            new_h -= 28


    return image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )


# =========================================================
# IoU
# =========================================================

def bbox_iou(a, b):

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
# DEDUP
# =========================================================

def normalize_text(text):

    return re.sub(
        r"\s+",
        "",
        str(text)
    )


def deduplicate(detections):

    unique = []

    for det in detections:

        bbox = det["bbox_page"]

        text = normalize_text(
            det["text_content"]
        )


        duplicate = False


        for existing in unique:

            existing_bbox = (
                existing["bbox_page"]
            )

            existing_text = (
                normalize_text(
                    existing[
                        "text_content"
                    ]
                )
            )


            overlap = bbox_iou(
                bbox,
                existing_bbox
            )


            # Nearly identical box
            if overlap >= 0.80:

                duplicate = True
                break


            # Identical text
            # in an overlapping region
            if (
                text
                and text == existing_text
                and overlap >= 0.25
            ):

                duplicate = True
                break


        if not duplicate:

            unique.append(det)


    return unique


# =========================================================
# OCR FOR ONE TILE
# =========================================================

def run_tile(
    model_tile,
    tile_name,
    page_number,
    page_output_dir,
):

    messages = [
        {
            "role": "user",

            "content": [
                {
                    "type": "image",
                    "image": model_tile,
                },

                {
                    "type": "text",
                    "text": OCR_PROMPT,
                },
            ],
        }
    ]


    prompt = (
        processor.apply_chat_template(
            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )


    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )


    inputs = processor(
        text=[prompt],

        images=image_inputs,

        videos=video_inputs,

        padding=True,

        return_tensors="pt",
    )


    inputs = inputs.to("cuda")


    input_tokens = (
        inputs.input_ids.shape[1]
    )


    torch.cuda.reset_peak_memory_stats()


    start = time.perf_counter()


    with torch.inference_mode():

        generated_ids = (
            model.generate(
                **inputs,

                max_new_tokens=(
                    MAX_NEW_TOKENS
                ),

                do_sample=False,

                use_cache=True,

                eos_token_id=(
                    processor
                    .tokenizer
                    .eos_token_id
                ),

                pad_token_id=(
                    processor
                    .tokenizer
                    .pad_token_id
                ),
            )
        )


    elapsed = (
        time.perf_counter()
        - start
    )


    trimmed = [
        output_ids[
            len(input_ids):
        ]

        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated_ids
        )
    ]


    raw_output = (
        processor.batch_decode(
            trimmed,

            skip_special_tokens=True,

            clean_up_tokenization_spaces=False,

        )[0]
    )


    raw_file = (
        page_output_dir
        / (
            f"{page_number:03d}_"
            f"{tile_name}_raw.txt"
        )
    )


    with open(
        raw_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(raw_output)


    detections = parse_output(
        raw_output
    )


    peak_vram = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )


    del inputs
    del generated_ids
    del trimmed

    gc.collect()

    torch.cuda.empty_cache()


    return (
        detections,
        elapsed,
        peak_vram,
        input_tokens,
    )


# =========================================================
# MAIN LOOP
# =========================================================

print()
print("=" * 70)
print("START 20-PAGE TILED BATCH")
print("=" * 70)


batch_start = time.time()


for page_position, page_number in enumerate(
    PAGES,
    start=1
):

    page_result_file = (
        OUTPUT_DIR
        / f"{page_number:03d}_result.json"
    )


    # =========================================
    # RESUME
    # =========================================

    if page_result_file.exists():

        print()
        print(
            f"[{page_position}/{len(PAGES)}] "
            f"PAGE {page_number:03d} "
            f"already done -> SKIP"
        )

        continue


    print()
    print("=" * 70)

    print(
        f"[{page_position}/{len(PAGES)}] "
        f"PAGE {page_number:03d}"
    )

    print("=" * 70)


    image_path = find_page_image(
        page_number
    )


    if image_path is None:

        print("IMAGE NOT FOUND")

        continue


    page = Image.open(
        image_path
    ).convert("RGB")


    PAGE_W, PAGE_H = (
        page.size
    )


    print(
        "Page size:",
        PAGE_W,
        "x",
        PAGE_H
    )


    # =========================================
    # PAGE DIRECTORY
    # =========================================

    page_output_dir = (
        OUTPUT_DIR
        / f"{page_number:03d}"
    )

    page_output_dir.mkdir(
        exist_ok=True
    )


    # =========================================
    # 4 TILES
    # =========================================

    mid_x = PAGE_W // 2
    mid_y = PAGE_H // 2


    overlap_x = int(
        PAGE_W
        * OVERLAP_RATIO
    )

    overlap_y = int(
        PAGE_H
        * OVERLAP_RATIO
    )


    tiles = [
        {
            "name": "top_left",

            "box": (
                0,
                0,

                mid_x + overlap_x,

                mid_y + overlap_y,
            ),
        },

        {
            "name": "top_right",

            "box": (
                mid_x - overlap_x,

                0,

                PAGE_W,

                mid_y + overlap_y,
            ),
        },

        {
            "name": "bottom_left",

            "box": (
                0,

                mid_y - overlap_y,

                mid_x + overlap_x,

                PAGE_H,
            ),
        },

        {
            "name": "bottom_right",

            "box": (
                mid_x - overlap_x,

                mid_y - overlap_y,

                PAGE_W,

                PAGE_H,
            ),
        },
    ]


    all_detections = []

    tile_stats = []


    page_start = time.time()


    # =========================================
    # TILE LOOP
    # =========================================

    for tile_index, tile in enumerate(
        tiles,
        start=1
    ):

        name = tile["name"]

        x1, y1, x2, y2 = (
            tile["box"]
        )


        print()
        print(
            f"  TILE {tile_index}/4: "
            f"{name}"
        )


        original_tile = page.crop(
            (x1, y1, x2, y2)
        )


        original_w, original_h = (
            original_tile.size
        )


        model_tile = resize_for_model(
            original_tile
        )


        model_w, model_h = (
            model_tile.size
        )


        print(
            "   size:",
            original_w,
            "x",
            original_h,
            "->",
            model_w,
            "x",
            model_h
        )


        detections, elapsed, peak_vram, tokens = (
            run_tile(
                model_tile,

                name,

                page_number,

                page_output_dir,
            )
        )


        print(
            "   detections:",
            len(detections)
        )

        print(
            "   time:",
            round(elapsed, 2),
            "sec"
        )

        print(
            "   VRAM:",
            round(peak_vram, 2),
            "GB"
        )


        # =====================================
        # LOCAL -> PAGE COORDINATES
        # =====================================

        scale_x = (
            original_w
            / model_w
        )

        scale_y = (
            original_h
            / model_h
        )


        for det in detections:

            bx1, by1, bx2, by2 = (
                det["bbox_2d"]
            )


            bx1 = max(
                0,
                min(model_w, bx1)
            )

            bx2 = max(
                0,
                min(model_w, bx2)
            )

            by1 = max(
                0,
                min(model_h, by1)
            )

            by2 = max(
                0,
                min(model_h, by2)
            )


            page_bbox = [
                round(
                    x1
                    + bx1 * scale_x
                ),

                round(
                    y1
                    + by1 * scale_y
                ),

                round(
                    x1
                    + bx2 * scale_x
                ),

                round(
                    y1
                    + by2 * scale_y
                ),
            ]


            all_detections.append({
                "tile": name,

                "bbox_model": [
                    bx1,
                    by1,
                    bx2,
                    by2,
                ],

                "bbox_page": (
                    page_bbox
                ),

                "text_content": (
                    det[
                        "text_content"
                    ]
                ),
            })


        tile_stats.append({
            "tile": name,

            "detections": (
                len(detections)
            ),

            "time_sec": elapsed,

            "peak_vram_gb": (
                peak_vram
            ),

            "input_tokens": (
                tokens
            ),

            "model_size": [
                model_w,
                model_h,
            ],
        })


    # =========================================
    # DEDUP
    # =========================================

    unique_detections = (
        deduplicate(
            all_detections
        )
    )


    page_elapsed = (
        time.time()
        - page_start
    )


    print()
    print(
        "Before dedup:",
        len(all_detections)
    )

    print(
        "After dedup:",
        len(unique_detections)
    )

    print(
        "PAGE TIME:",
        round(page_elapsed, 2),
        "sec"
    )


    # =========================================
    # SAVE PAGE JSON
    # =========================================

    result = {
        "page": page_number,

        "image": str(
            image_path
        ),

        "page_size": [
            PAGE_W,
            PAGE_H,
        ],

        "time_sec": (
            page_elapsed
        ),

        "detections_before_dedup": (
            len(all_detections)
        ),

        "detections_after_dedup": (
            len(unique_detections)
        ),

        "tile_stats": (
            tile_stats
        ),

        "detections": (
            unique_detections
        ),
    }


    with open(
        page_result_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,

            ensure_ascii=False,

            indent=2
        )


    # =========================================
    # VISUALIZATION
    # =========================================

    visual = page.copy()

    draw = ImageDraw.Draw(
        visual
    )


    for index, det in enumerate(
        unique_detections,
        start=1
    ):

        bbox = det[
            "bbox_page"
        ]

        draw.rectangle(
            bbox,

            outline="red",

            width=3
        )

        draw.text(
            (
                bbox[0],

                max(
                    0,
                    bbox[1] - 15
                ),
            ),

            str(index),

            fill="red"
        )


    vis_path = (
        page_output_dir
        / f"{page_number:03d}_boxes.jpg"
    )


    visual.save(
        vis_path,
        quality=95
    )


    gc.collect()
    torch.cuda.empty_cache()


# =========================================================
# FINISH
# =========================================================

batch_elapsed = (
    time.time()
    - batch_start
)


print()
print("=" * 70)
print("20-PAGE BATCH FINISHED")
print("=" * 70)

print(
    "Output:",
    OUTPUT_DIR
)

print(
    "Total elapsed:",
    round(
        batch_elapsed / 60,
        2
    ),
    "minutes"
)