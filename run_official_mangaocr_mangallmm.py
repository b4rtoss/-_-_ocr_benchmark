import json
import time
import gc
import re
from pathlib import Path

import torch
from PIL import Image

from transformers import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\mangalmm_test\official_mangaocr_test50"
)

MANIFEST = BASE / "manifest.jsonl"

OUTPUT_DIR = BASE / "mangallmm_results"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    OUTPUT_DIR / "results.jsonl"
)


# ============================================================
# PILOT
#
# Пока тестируем только 3 изображения.
# Когда убедимся, что всё работает,
# поменяем 3 на 50.
# ============================================================

LIMIT = 50


# ============================================================
# MODEL / PROCESSOR
# ============================================================

MODEL_ID = "hal-utokyo/MangaLMM"

# MangaLMM основана на Qwen2.5-VL.
# Processor берём из базовой Qwen2.5-VL-7B,
# потому что локальная версия transformers
# не может распознать image processor
# непосредственно из MangaLMM.
PROCESSOR_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)


# ============================================================
# IMAGE RESOLUTION
#
# Ограничиваем разрешение из-за RTX 4050 6 GB.
# ============================================================

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28


# ============================================================
# CUDA
# ============================================================

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


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=(
        torch.float16
    ),

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# PROCESSOR
# ============================================================

print()
print("Loading Qwen processor...")


processor = (
    Qwen2_5_VLProcessor
    .from_pretrained(
        PROCESSOR_ID,

        min_pixels=MIN_PIXELS,

        max_pixels=MAX_PIXELS,
    )
)


print(
    "Processor loaded successfully."
)


# ============================================================
# MANGALMM MODEL
# ============================================================

print()
print("Loading MangaLMM...")


model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,

        quantization_config=(
            bnb_config
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


# ============================================================
# READ MANIFEST
# ============================================================

records = []


with open(
    MANIFEST,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        records.append(
            json.loads(line)
        )


# Только первые LIMIT
records = records[:LIMIT]


print()
print(
    "Samples for this run:",
    len(records)
)


# ============================================================
# RESUME
#
# Если какой-то пример уже успешно записан,
# повторно его не запускаем.
# ============================================================

already_done = set()


if RESULT_FILE.exists():

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

                if (
                    row.get("status")
                    == "ok"
                ):

                    already_done.add(
                        row["id"]
                    )

            except Exception:
                pass


print(
    "Already completed:",
    len(already_done)
)


# ============================================================
# OUTPUT PARSER
#
# MangaLMM обычно выдаёт:
#
# [
#   {
#     "bbox_2d": [x1,y1,x2,y2],
#     "text_content": "..."
#   }
# ]
# ============================================================

def parse_output(text):

    if not text:
        return []


    clean = str(text)

    clean = (
        clean
        .replace("```json", "")
        .replace("```JSON", "")
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


    detections = []


    for match in pattern.finditer(
        clean
    ):

        detections.append({
            "bbox_2d": [
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            ],

            "text_content": (
                match.group(5)
            ),
        })


    return detections


# ============================================================
# RUN
# ============================================================

for index, record in enumerate(
    records,
    start=1
):

    example_id = record["id"]


    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if example_id in already_done:

        print(
            f"[{index}/{len(records)}] "
            f"{example_id} -> SKIP"
        )

        continue


    print()
    print("=" * 70)

    print(
        f"[{index}/{len(records)}] "
        f"{example_id}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    image_path = Path(
        record["image"]
    )

    prompt = record["prompt"]


    print(
        "Image:",
        image_path
    )

    print(
        "Prompt:",
        prompt
    )


    if not image_path.exists():

        print(
            "ERROR: image does not exist!"
        )

        continue


    image = Image.open(
        image_path
    ).convert("RGB")


    print(
        "Original image size:",
        image.size
    )


    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

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
                    "text": prompt,
                },
            ],
        }
    ]


    # --------------------------------------------------------
    # CHAT TEMPLATE
    # --------------------------------------------------------

    text_prompt = (
        processor
        .apply_chat_template(
            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )


    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )


    inputs = processor(
        text=[text_prompt],

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


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()


    start = time.perf_counter()


    status = "ok"

    raw_output = ""

    detections = []

    output_ids = None
    generated = None


    try:

        print(
            "Running MangaLMM..."
        )


        with torch.inference_mode():

            output_ids = (
                model.generate(
                    **inputs,

                    max_new_tokens=1024,

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


        # ----------------------------------------------------
        # Убираем prompt tokens
        # ----------------------------------------------------

        generated = [

            out[
                len(inp):
            ]

            for inp, out
            in zip(
                inputs.input_ids,
                output_ids
            )
        ]


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        raw_output = (
            processor.batch_decode(
                generated,

                skip_special_tokens=True,

                clean_up_tokenization_spaces=False,
            )[0]
        )


        # ----------------------------------------------------
        # PARSE
        # ----------------------------------------------------

        detections = (
            parse_output(
                raw_output
            )
        )


    except torch.cuda.OutOfMemoryError:

        elapsed = (
            time.perf_counter()
            - start
        )

        status = "cuda_oom"

        raw_output = (
            "CUDA OUT OF MEMORY"
        )

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

        raw_output = repr(e)

        print(
            "ERROR:",
            repr(e)
        )


    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    peak_vram = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )


    print()
    print(
        "Status:",
        status
    )

    print(
        "Detected regions:",
        len(detections)
    )

    print(
        "Inference time:",
        round(
            elapsed,
            2
        ),
        "sec"
    )

    print(
        "Peak VRAM:",
        round(
            peak_vram,
            2
        ),
        "GB"
    )


    # --------------------------------------------------------
    # SHOW SHORT RAW OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "RAW OUTPUT PREVIEW:"
    )

    print(
        raw_output[:1000]
    )

    print()


    # ========================================================
    # SAVE RAW OUTPUT
    # ========================================================

    raw_path = (
        OUTPUT_DIR
        / (
            f"{index:03d}_"
            f"{example_id}_raw.txt"
        )
    )


    with open(
        raw_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            raw_output
        )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    output_record = {

        "sample_number": (
            record[
                "sample_number"
            ]
        ),

        "official_test_index": (
            record[
                "official_test_index"
            ]
        ),

        "id": example_id,

        "image": str(
            image_path
        ),

        "prompt": prompt,

        "ground_truth": (
            record[
                "ground_truth"
            ]
        ),

        "prediction_raw": (
            raw_output
        ),

        "detections": (
            detections
        ),

        "detected_regions": (
            len(detections)
        ),

        "time_sec": (
            elapsed
        ),

        "peak_vram_gb": (
            peak_vram
        ),

        "status": status,
    }


    with open(
        RESULT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                output_record,

                ensure_ascii=False
            )
            + "\n"
        )


    # ========================================================
    # CLEAN MEMORY
    # ========================================================

    del inputs


    if output_ids is not None:
        del output_ids


    if generated is not None:
        del generated


    del image

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)

print(
    "PILOT FINISHED"
)

print("=" * 70)

print(
    "Results:",
    RESULT_FILE
)

print(
    "Raw outputs:",
    OUTPUT_DIR
)