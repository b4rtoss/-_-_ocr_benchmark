"""Common CSV contract and command line for the three crop OCR engines."""
import argparse
import csv
import time
from pathlib import Path

from PIL import Image
from .benchmark_paths import BASE_DIR

FIELDS = ['file', 'ground_truth', 'prediction', 'time_sec', 'status']


def read_ground_truth(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        text_column = 'text' if 'text' in columns else 'ground_truth'
        if 'file' not in columns or text_column not in columns:
            raise ValueError('Ground truth CSV needs file and text (or ground_truth) columns.')
        rows = list(reader)
    if not rows:
        raise ValueError('Ground truth CSV is empty.')
    names = [row['file'] for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError('Ground truth file names must be nonempty and unique.')
    return [{'file': row['file'], 'ground_truth': row[text_column]} for row in rows]


def run_batch(rows, dataset_dir, output, recognize, resume=False):
    output = Path(output)
    processed = {}
    if resume and output.exists() and output.stat().st_size:
        with output.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError('Cannot resume: incompatible results CSV columns.')
            previous = list(reader)
        expected = {row['file']: row['ground_truth'] for row in rows}
        for row in previous:
            if row['file'] in processed or expected.get(row['file']) != row['ground_truth']:
                raise ValueError('Cannot resume: results contain duplicates or different ground truth.')
            processed[row['file']] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    failed = sum(row['status'] != 'ok' for row in processed.values())
    with output.open('a' if processed else 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not processed:
            writer.writeheader()
        for index, row in enumerate(rows, 1):
            if row['file'] in processed:
                continue
            prediction, status = '', 'ok'
            started = None
            try:
                with Image.open(Path(dataset_dir) / row['file']) as source:
                    image = source.convert('RGB')
                started = time.perf_counter()
                prediction = str(recognize(image)).strip()
            except Exception as exc:
                status = f'error:{type(exc).__name__}: {exc}'
                failed += 1
            elapsed = time.perf_counter() - started if started is not None else 0.0
            writer.writerow({**row, 'prediction': prediction, 'time_sec': f'{elapsed:.6f}', 'status': status})
            stream.flush()
            print(f'[{index}/{len(rows)}] {row["file"]}: {status}')
    print(f'Saved: {output}; samples: {len(rows)}; failed: {failed}')
    return failed


def main(engine):
    parser = argparse.ArgumentParser(description=f'Run {engine} on text crops.')
    parser.add_argument('--dataset-dir', type=Path, default=BASE_DIR / 'dataset')
    parser.add_argument('--ground-truth', type=Path, default=BASE_DIR / 'ground_truth.csv')
    parser.add_argument('--output', type=Path, default=BASE_DIR / f'results_{engine}.csv')
    parser.add_argument('--limit', type=int, help='Process the first N crops for a smoke test.')
    parser.add_argument('--resume', action='store_true', help='Keep existing rows, including errors; omit to retry all.')
    parser.add_argument('--language', default='japan' if engine == 'paddleocr' else 'jpn')
    parser.add_argument('--device', default='cpu', help='PaddleOCR device (cpu or gpu:0).')
    parser.add_argument('--tesseract-cmd', help='Tesseract executable if absent from PATH.')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    try:
        rows = read_ground_truth(args.ground_truth)
        if args.limit:
            rows = rows[:args.limit]
        if engine == 'mangaocr':
            from manga_ocr import MangaOcr
            recognize = MangaOcr()
        elif engine == 'paddleocr':
            from manga_app.engine import Analyzer
            analyzer = Analyzer()
            recognize = lambda image: ''.join(r['text'] for r in analyzer.recognize(image, args.language))
            analyzer.device = args.device
            analyzer.initialize_ocr(args.language)
        else:
            import pytesseract
            if args.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
            pytesseract.get_tesseract_version()
            recognize = lambda image: pytesseract.image_to_string(image, lang=args.language, config='--psm 6')
        return 1 if run_batch(rows, args.dataset_dir, args.output, recognize, args.resume) else 0
    except Exception as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')
