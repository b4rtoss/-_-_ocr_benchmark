import os
import re
import gc
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image

from qwen_vl_utils import process_vision_info

from transformers import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)


# =========================================================
# ПУТИ
# =========================================================

# Так не нужно вручную писать C:\Users\Соня
ROOT = (
    Path.home()
    / "Desktop"
    / "Новая папка"
    / "Manga109S_released_2026_05_21"
)

IMAGE_DIR = ROOT / "images" / "Hamlet"
XML_PATH = ROOT / "annotations" / "Hamlet.xml"

OUTPUT_DIR = Path(
    r"C:\mangalmm_test\mangallmm_pages_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_JSONL = (
    OUTPUT_DIR
    / "results_pages_5.jsonl"
)


# =========================================================
# БЕРЁМ 5 НОРМАЛЬНЫХ СТРАНИЦ МАНГИ
# =========================================================

# Не 000-004, потому что там могут быть
# обложки/служебные страницы.
PAGES = [8, 9, 10, 11, 12]


# =========================================================
# MODEL
# =========================================================

MODEL_ID = "hal-utokyo/MangaLMM"

# Процессор берём от базовой Qwen,
# как в нашей уже рабочей конфигурации.
PROCESSOR_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)


# =========================================================
# ОФИЦИАЛЬНЫЙ OCR PROMPT
# =========================================================

OCR_PROMPT = (
    "Please perform OCR on this image and output "
    "the recognized Japanese text along with its "
    "position (grounding)."
)


# =========================================================
# ПРОВЕРКА CUDA
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

        # Полная страница требует больше
        # visual tokens, чем наши кропы.
        #
        # Начинаем с 256.
        # Для RTX 4050 6GB это уже
        # довольно близко к пределу.
        min_pixels=128 * 28 * 28,

        max_pixels=256 * 28 * 28,
    )
)


print(
    "Processor loaded successfully."
)


# =========================================================
# MANGALMM
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
# ЧИТАЕМ HAMLET.XML
# =========================================================

print()
print("Reading Hamlet.xml...")


tree = ET.parse(XML_PATH)

root = tree.getroot()


GROUND_TRUTH = {}


for page in root.findall(".//page"):

    page_index = int(
        page.attrib["index"]
    )

    texts = []

    for item in page.findall("text"):

        content = (
            item.text or ""
        ).strip()

        if not content:
            continue

        texts.append({
            "text": content,

            "bbox": [
                int(item.attrib["xmin"]),
                int(item.attrib["ymin"]),
                int(item.attrib["xmax"]),
                int(item.attrib["ymax"]),
            ],

            "id": item.attrib.get(
                "id",
                ""
            ),
        })

    GROUND_TRUTH[
        page_index
    ] = texts


print(
    "XML loaded."
)


# =========================================================
# PARSER MANGALMM OUTPUT
# =========================================================

def parse_mangallmm_output(text):

    if not text:
        return []

    # Убираем некоторые служебные части
    clean = (
        text
        .replace("```json", "")
        .replace("```", "")
    )

    results = []

    # Ищем объекты вида:
    #
    # "bbox_2d": [x1,y1,x2,y2],
    # "text_content": "..."
    #
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

    for match in pattern.finditer(
        clean
    ):

        bbox = [
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
        ]

        content = match.group(5)

        results.append({
            "bbox_2d": bbox,
            "text_content": content,
        })

    return results


# =========================================================
# FIND PAGE IMAGE
# =========================================================

def find_page_image(
    page_number
):

    name = f"{page_number:03d}"

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    for extension in extensions:

        candidate = (
            IMAGE_DIR
            / f"{name}{extension}"
        )

        if candidate.exists():
            return candidate

    return None


# =========================================================
# ОЧИЩАЕМ СТАРЫЙ ТЕСТОВЫЙ JSONL
# =========================================================

if RESULT_JSONL.exists():

    RESULT_JSONL.unlink()


# =========================================================
# INFERENCE
# =========================================================

print()
print("=" * 60)
print("START PAGE OCR")
print("=" * 60)


for counter, page_number in enumerate(
    PAGES,
    start=1
):

    print()
    print(
        f"PAGE {page_number:03d} "
        f"({counter}/{len(PAGES)})"
    )


    image_path = find_page_image(
        page_number
    )


    if image_path is None:

        print(
            "IMAGE NOT FOUND"
        )

        continue


    image = Image.open(
        image_path
    ).convert("RGB")


    width, height = (
        image.size
    )


    print(
        "Original size:",
        width,
        "x",
        height
    )


    gt = GROUND_TRUTH.get(
        page_number,
        []
    )


    print(
        "GT text regions:",
        len(gt)
    )


    # =====================================================
    # MESSAGE
    # =====================================================

    messages = [
        {
            "role": "user",

            "content": [
                {
                    "type": "image",
                    "image": image,
                },

                {
                    "type": "text",
                    "text": OCR_PROMPT,
                },
            ],
        }
    ]


    prompt = (
        processor
        .apply_chat_template(
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


    # =====================================================
    # PREPROCESSING
    # =====================================================

    inputs = processor(
        text=[prompt],

        images=image_inputs,

        videos=video_inputs,

        padding=True,

        return_tensors="pt",
    )


    inputs = inputs.to(
        "cuda"
    )


    print(
        "Input tokens:",
        inputs.input_ids.shape[1]
    )


    # =====================================================
    # GENERATION
    # =====================================================

    torch.cuda.reset_peak_memory_stats()


    print(
        "Running MangaLMM..."
    )


    start = time.perf_counter()


    status = "ok"

    raw_output = ""

    predictions = []


    try:

        with torch.inference_mode():

            generated_ids = (
                model.generate(

                    **inputs,

                    # На странице текста больше,
                    # поэтому 128 уже мало.
                    max_new_tokens=768,

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


        predictions = (
            parse_mangallmm_output(
                raw_output
            )
        )


    except torch.cuda.OutOfMemoryError:

        elapsed = (
            time.perf_counter()
            - start
        )

        status = "cuda_oom"

        print(
            "CUDA OUT OF MEMORY"
        )


    except Exception as e:

        elapsed = (
            time.perf_counter()
            - start
        )

        status = (
            "error:"
            + type(e).__name__
        )

        print(
            "ERROR:",
            repr(e)
        )


    # =====================================================
    # MEMORY
    # =====================================================

    peak_memory = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )


    print(
        "Predicted regions:",
        len(predictions)
    )

    print(
        "Inference time:",
        round(elapsed, 2),
        "sec"
    )

    print(
        "Peak VRAM:",
        round(peak_memory, 2),
        "GB"
    )

    print(
        "Status:",
        status
    )


    # =====================================================
    # SAVE RAW OUTPUT
    # =====================================================

    raw_path = (
        OUTPUT_DIR
        / f"{page_number:03d}_raw.txt"
    )


    with open(
        raw_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            raw_output
        )


    # =====================================================
    # SAVE JSONL
    # =====================================================

    record = {

        "page": page_number,

        "image": str(
            image_path
        ),

        "width": width,

        "height": height,

        "status": status,

        "time_sec": elapsed,

        "peak_vram_gb": peak_memory,

        "ground_truth": gt,

        "prediction": predictions,

        "raw_output_file": str(
            raw_path
        ),
    }


    with open(
        RESULT_JSONL,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


    # =====================================================
    # CLEAN MEMORY
    # =====================================================

    del inputs

    if "generated_ids" in locals():
        del generated_ids

    if "trimmed" in locals():
        del trimmed

    gc.collect()

    torch.cuda.empty_cache()


print()
print("=" * 60)
print("5-PAGE TEST FINISHED")
print("=" * 60)

print(
    "Results:",
    RESULT_JSONL
)

print(
    "Raw outputs:",
    OUTPUT_DIR
)