"""Versioned session files; page images remain external to the session."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile

FORMAT = 'manga-reader-session'
VERSION = 1
LANGUAGES = ('Japanese', 'English', 'Russian')


def _validate_record(record):
    if not isinstance(record, dict):
        raise ValueError('Invalid page result.')
    for key in ('analysis', 'error'):
        if key in record and not isinstance(record[key], str):
            raise ValueError('Invalid result text.')
    warnings = record.get('warnings', [])
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        raise ValueError('Invalid result warnings.')
    regions = record.get('regions', [])
    if not isinstance(regions, list):
        raise ValueError('Invalid OCR regions.')
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    for region in regions:
        if not isinstance(region, dict) or not isinstance(region.get('text'), str):
            raise ValueError('Invalid OCR text.')
        confidence = region.get('confidence')
        if not number(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid OCR confidence.')
        polygon = region.get('polygon')
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError('Invalid OCR polygon.')
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2 or not all(number(n) for n in point):
                raise ValueError('Invalid OCR coordinates.')


def _decode(payload, directory):
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError('This is not a Manga Reader session. Exported results are not session files.')
    if type(payload.get('version')) is not int or payload['version'] != VERSION:
        raise ValueError('Unsupported session version.')
    settings = payload.get('settings')
    if not isinstance(settings, dict) or settings.get('language') not in LANGUAGES:
        raise ValueError('Invalid source language.')
    if type(settings.get('story')) is not bool or not isinstance(settings.get('model'), str):
        raise ValueError('Invalid analysis settings.')
    pages = payload.get('pages')
    if not isinstance(pages, list):
        raise ValueError('Invalid page list.')
    files, records = [], {}
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get('path'), str) or not page['path']:
            raise ValueError('Invalid image path.')
        if '\x00' in page['path']:
            raise ValueError('Invalid image path.')
        path = str((directory / page['path']).resolve())
        if path in files:
            raise ValueError('Duplicate image path.')
        files.append(path)
        record = page.get('result')
        if record is not None:
            _validate_record(record)
            record = copy.deepcopy(record)
            record['file'] = path
            records[path] = record
    selected = payload.get('selected')
    if selected is not None and (type(selected) is not int or not 0 <= selected < len(files)):
        raise ValueError('Invalid selected page.')
    return dict(files=files, records=records, selected=selected, **settings)


def load_session(path):
    path = Path(path).resolve()
    return _decode(json.loads(path.read_text(encoding='utf-8')), path.parent)


def save_session(path, files, records, language, story, model, selected):
    """Replace atomically so a failed write cannot truncate an existing session."""
    path = Path(path).resolve()
    pages = []
    for file in files:
        try:
            stored_path = os.path.relpath(Path(file).resolve(), path.parent)
        except ValueError:  # Images may be on a different Windows drive.
            stored_path = str(Path(file).resolve())
        record = copy.deepcopy(records.get(file))
        if record is not None:
            record.pop('file', None)
        pages.append({'path': stored_path, 'result': record})
    payload = {'format': FORMAT, 'version': VERSION, 'pages': pages,
               'selected': selected,
               'settings': {'language': language, 'story': story, 'model': model}}
    _decode(payload, path.parent)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
