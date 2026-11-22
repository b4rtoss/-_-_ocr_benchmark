import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from research.batch_runner import FIELDS, read_ground_truth, run_batch
from research.evaluate_results import evaluate
from manga_app.engine import Analyzer, load_image, natural_key

ROOT = Path(__file__).resolve().parents[1]


class Workflows(unittest.TestCase):
    def test_all_scripts_parse_without_optional_model_dependencies(self):
        for path in [*(ROOT / 'research').glob('*.py'), *(ROOT / 'manga_app').glob('*.py')]:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_research_default_data_path_survives_package_move(self):
        env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONIOENCODING': 'utf-8'}
        env.pop('OCR_BENCHMARK_DIR', None)
        env.pop('MANGA109_ROOT', None)
        result = subprocess.run([sys.executable, '-c',
            'from research.benchmark_paths import BASE_DIR, IMAGE_DIR; '
            'import json; print(json.dumps([str(BASE_DIR), str(IMAGE_DIR)]))'],
            cwd=self.base, env=env, capture_output=True, encoding='utf-8', check=True)
        self.assertEqual(json.loads(result.stdout), [str(ROOT), str(ROOT / 'Hamlet')])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def csv(self, name, rows, fields=FIELDS):
        path = self.base / name
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def script(self, name, *args, ok=True):
        result = subprocess.run([sys.executable, '-m', 'research.' + Path(name).stem, *map(str, args)],
                                cwd=self.base, capture_output=True, encoding='utf-8',
                                env={**os.environ, 'OCR_BENCHMARK_DIR': str(self.base),
                                     'PYTHONIOENCODING': 'utf-8', 'MANGA109_ROOT': '',
                                     'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH', '')]))})
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def test_prepare_dataset_and_chatgpt_sample(self):
        folder = self.base / 'Hamlet'
        folder.mkdir()
        Image.new('RGB', (100, 100), 'white').save(folder / '001.jpg')
        (folder / 'Hamlet.xml').write_text('<book><pages><page index="1">'
            '<text id="a" xmin="10" ymin="10" xmax="40" ymax="40">日本語</text>'
            '</page></pages></book>', encoding='utf-8')
        self.script('prepare_dataset.py')
        rows = read_ground_truth(self.base / 'ground_truth.csv')
        self.assertEqual(rows, [{'file': '00001.jpg', 'ground_truth': '日本語'}])
        with Image.open(self.base / 'dataset' / '00001.jpg') as image:
            self.assertEqual(image.size, (40, 40))
        self.script('prepare_chatgpt_50.py', '--size', 1)
        with (self.base / 'chatgpt_test50.csv').open(encoding='utf-8-sig') as stream:
            sample = list(csv.DictReader(stream))
        self.assertEqual(sample[0]['ground_truth'], '日本語')
        self.assertTrue((self.base / 'chatgpt_test50' / sample[0]['test_file']).exists())

    def test_batch_records_missing_images_and_resumes_without_duplicates(self):
        Image.new('RGB', (20, 20)).save(self.base / 'one.png')
        rows = [{'file': f, 'ground_truth': '日本'} for f in ['one.png', 'missing.png']]
        output = self.base / 'results.csv'
        self.assertEqual(run_batch(rows, self.base, output, lambda image: '日本'), 1)
        self.assertEqual(run_batch(rows, self.base, output, lambda image: self.fail(), resume=True), 1)
        report = evaluate(output)
        self.assertEqual(report['samples'], 2)
        self.assertEqual(report['failed'], 1)
        self.assertEqual(report['CER'], 0.5)

    def test_resume_rejects_other_ground_truth(self):
        output = self.csv('results.csv', [{'file': 'a', 'ground_truth': 'old', 'prediction': '',
                                          'time_sec': 0, 'status': 'ok'}])
        with self.assertRaisesRegex(ValueError, 'different ground truth'):
            run_batch([{'file': 'a', 'ground_truth': 'new'}], self.base, output, None, True)

    def test_metrics_normalize_and_exclude_failed_latency(self):
        path = self.csv('results.csv', [
            {'file': 'a', 'ground_truth': 'Ａ 日\n本', 'prediction': 'A日本', 'time_sec': '2', 'status': 'ok'},
            {'file': 'b', 'ground_truth': '語', 'prediction': '', 'time_sec': '0', 'status': 'error:bad'}])
        report = evaluate(path)
        self.assertEqual(report['CER'], 0.25)
        self.assertEqual(report['Exact Match'], 0.5)
        self.assertEqual(report['avg_time'], 2)

    def test_metrics_reject_empty_or_duplicate_results(self):
        path = self.csv('empty.csv', [])
        with self.assertRaises(ValueError):
            evaluate(path)
        row = {'file': 'a', 'ground_truth': '日', 'prediction': '日', 'time_sec': 1, 'status': 'ok'}
        path = self.csv('duplicate.csv', [row, row])
        with self.assertRaises(ValueError):
            evaluate(path)

    def test_report_rejects_different_samples(self):
        row = {'file': 'a', 'ground_truth': '日', 'prediction': '日', 'time_sec': 1, 'status': 'ok'}
        first = self.csv('one.csv', [row])
        second = self.csv('two.csv', [{**row, 'file': 'b'}])
        result = self.script('evaluate_results.py', first, second, ok=False)
        self.assertIn('different samples', result.stderr)

    def test_single_result_report_exports_metrics(self):
        row = {'file': 'a', 'ground_truth': '日', 'prediction': '日', 'time_sec': 1, 'status': 'ok'}
        first = self.csv('one.csv', [row])
        output = self.base / 'report.csv'
        self.script('evaluate_results.py', first, '--output', output)
        self.assertTrue(output.exists())

    def test_all_baseline_help_commands_without_optional_engines(self):
        for name in ['run_paddle_batch.py', 'run_mangaocr_batch.py', 'run_tesseract_batch.py']:
            self.script(name, '--help')

    def test_story_failure_preserves_ocr(self):
        path = self.base / 'page.png'
        Image.new('RGB', (40, 50), 'white').save(path)
        regions = [{'text': '日本', 'confidence': 0.9, 'polygon': [[0, 0], [20, 0], [20, 20], [0, 20]]}]
        with patch.object(Analyzer, 'recognize', return_value=regions), \
                patch('manga_app.engine.request_json', side_effect=RuntimeError('offline')):
            record = Analyzer().analyze(path)
        self.assertEqual(record['regions'], regions)
        self.assertIn('offline', record['warnings'][0])

    def test_story_can_work_when_ocr_fails(self):
        path = self.base / 'page.png'
        Image.new('RGB', (40, 50), 'white').save(path)
        with patch.object(Analyzer, 'recognize', side_effect=RuntimeError('ocr failed')), \
                patch('manga_app.engine.request_json', return_value={'message': {'content': 'Scene'}, 'done_reason': 'length'}):
            record = Analyzer().analyze(path)
        self.assertEqual(record['analysis'], 'Scene')
        self.assertEqual(len(record['warnings']), 2)

    def test_ocr_only_does_not_call_ollama(self):
        path = self.base / 'page.png'
        Image.new('RGBA', (40, 50)).save(path)
        self.assertEqual(load_image(path).mode, 'RGB')
        with patch.object(Analyzer, 'recognize', return_value=[]), \
                patch('manga_app.engine.request_json') as request:
            self.assertEqual(Analyzer().analyze(path, story=False)['regions'], [])
            request.assert_not_called()

    def test_story_requests_english_without_translating_ocr_input(self):
        path = self.base / 'page.png'
        Image.new('RGB', (40, 50), 'white').save(path)
        regions = [{'text': '日本語', 'confidence': 0.9, 'polygon': [[0, 0], [20, 0], [20, 20], [0, 20]]}]
        with patch.object(Analyzer, 'recognize', return_value=regions), \
                patch('manga_app.engine.request_json', return_value={'message': {'content': 'An English analysis.'}}) as request:
            record = Analyzer().analyze(path, context='An earlier page.')
        route, payload = request.call_args.args
        self.assertEqual(route, 'chat')
        message = payload['messages'][0]
        self.assertIn('Respond in English', message['content'])
        self.assertIn('translation into English', message['content'])
        self.assertIn('日本語', message['content'])
        self.assertIn('An earlier page.', message['content'])
        self.assertEqual(len(message['images']), 1)
        self.assertEqual(record['regions'], regions)
        self.assertEqual(record['analysis'], 'An English analysis.')

    def test_natural_page_order(self):
        self.assertEqual(sorted(['10.png', '2.png', '1.png'], key=natural_key), ['1.png', '2.png', '10.png'])


if __name__ == '__main__':
    unittest.main()
