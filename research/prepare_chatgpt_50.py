"""Prepare a reproducible sample for manual ChatGPT OCR."""
import argparse
import csv
import random
import shutil
from .benchmark_paths import BASE_DIR
from .batch_runner import read_ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error('--size must be positive')
    rows = read_ground_truth(BASE_DIR / 'ground_truth.csv')
    if len(rows) < args.size:
        parser.error(f'Ground truth contains {len(rows)} rows, fewer than {args.size}')
    sample = random.Random(args.seed).sample(rows, args.size)
    output = BASE_DIR / 'chatgpt_test50'
    sources = [BASE_DIR / 'dataset' / row['file'] for row in sample]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    result = []
    for index, (row, source) in enumerate(zip(sample, sources), 1):
        name = f'{index:03d}{source.suffix or ".jpg"}'
        shutil.copy2(source, output / name)
        result.append({'number': index, 'original_file': row['file'], 'test_file': name,
                       'ground_truth': row['ground_truth'], 'chatgpt_prediction': ''})
    csv_path = BASE_DIR / 'chatgpt_test50.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f'Saved {len(result)} samples: {csv_path}')


if __name__ == '__main__':
    main()
