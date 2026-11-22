"""Extract annotated Hamlet text crops and their reference transcription."""
import csv
import xml.etree.ElementTree as ET
from PIL import Image
from .benchmark_paths import BASE_DIR, IMAGE_DIR, XML_PATH

PADDING = 5
FIELDS = ['file', 'text', 'page', 'xmin', 'ymin', 'xmax', 'ymax', 'text_id']


def main():
    root = ET.parse(XML_PATH).getroot()
    output = BASE_DIR / 'dataset'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for page in root.findall('.//page'):
        index = int(page.attrib['index'])
        image_path = next((IMAGE_DIR / f'{index:03d}{ext}'
                           for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                           if (IMAGE_DIR / f'{index:03d}{ext}').is_file()), None)
        if image_path is None:
            print(f'WARNING: image for page {index} not found')
            continue
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        for element in page.findall('text'):
            text = (element.text or '').strip()
            if not text:
                continue
            xmin, ymin, xmax, ymax = (int(element.attrib[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax'])
            box = (max(0, xmin - PADDING), max(0, ymin - PADDING),
                   min(image.width, xmax + PADDING), min(image.height, ymax + PADDING))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'Invalid text box on page {index}: {element.attrib}')
            name = f'{len(rows) + 1:05d}.jpg'
            image.crop(box).save(output / name, quality=95)
            rows.append(dict(zip(FIELDS, [name, text, index, xmin, ymin, xmax, ymax, element.get('id', '')])))
    if not rows:
        raise ValueError('No annotated crops found. Check the XML and image directory.')
    csv_path = BASE_DIR / 'ground_truth.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Dataset created: {len(rows)} crops in {output}; ground truth: {csv_path}')


if __name__ == '__main__':
    main()
