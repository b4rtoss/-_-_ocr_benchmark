import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from manga_app.session import load_session, save_session


class SessionFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.files = [str(self.root / name) for name in ('2.png', '1.png', '3.png')]
        for file in self.files:
            Image.new('RGB', (20, 20), 'white').save(file)
        self.records = {
            self.files[0]: {'file': self.files[0], 'width': 20, 'height': 20,
                'analysis': 'English scene description.', 'warnings': ['Check this line.'],
                'regions': [{'text': '日本語', 'confidence': 0.85,
                             'polygon': [[1, 1], [10, 1], [10, 10], [1, 10]]}]},
            self.files[1]: {'file': self.files[1], 'error': 'Unreadable page.'},
        }
        self.path = self.root / 'chapter.manga-session.json'

    def save(self):
        save_session(self.path, self.files, self.records, 'Japanese', False, 'custom:model', 1)


class Sessions(SessionFixture, unittest.TestCase):
    def test_roundtrip_preserves_order_results_settings_and_selection(self):
        original = copy.deepcopy(self.records)
        self.save()
        state = load_session(self.path)
        self.assertEqual(state['files'], self.files)
        self.assertEqual(state['records'], original)
        self.assertEqual(self.records, original)
        self.assertEqual((state['language'], state['story'], state['model'], state['selected']),
                         ('Japanese', False, 'custom:model', 1))
        self.assertNotIn(self.files[2], state['records'])

    def test_relative_paths_survive_moving_session_and_images_together(self):
        self.save()
        moved = self.root / 'moved'
        moved.mkdir()
        for file in [self.path, *map(Path, self.files)]:
            shutil.copyfile(file, moved / file.name)
        state = load_session(moved / self.path.name)
        expected = [str(moved / Path(file).name) for file in self.files]
        self.assertEqual(state['files'], expected)
        self.assertEqual(state['records'][expected[0]]['file'], expected[0])

    def test_missing_images_do_not_discard_saved_text(self):
        self.save()
        Path(self.files[0]).unlink()
        self.assertEqual(load_session(self.path)['records'][self.files[0]]['regions'][0]['text'], '日本語')

    def test_failed_atomic_replace_preserves_previous_session(self):
        self.save()
        before = self.path.read_bytes()
        with patch('manga_app.session.os.replace', side_effect=OSError('Disk error')):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_invalid_sessions_are_rejected(self):
        self.save()
        good = json.loads(self.path.read_text(encoding='utf-8'))
        invalid = [[], {**good, 'version': 99}, {**good, 'selected': -1},
                   {**good, 'settings': {**good['settings'], 'story': 'false'}},
                   {**good, 'pages': [good['pages'][0], good['pages'][0]]}]
        malformed = copy.deepcopy(good)
        malformed['pages'][0]['result']['regions'][0]['polygon'] = [[float('nan'), 1]] * 4
        invalid.append(malformed)
        for payload in invalid:
            with self.subTest(payload=payload):
                self.path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_session(self.path)

    def test_empty_session_roundtrip(self):
        save_session(self.path, [], {}, 'English', True, '', None)
        self.assertEqual(load_session(self.path)['files'], [])


class SessionInterface(SessionFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from manga_app.app import App
        self.app = App()
        self.app.withdraw()
        self.addCleanup(self.app.destroy)

    def test_save_and_reopen_without_running_ocr(self):
        self.app.add(self.files)
        # Restore a deliberately non-natural order to check opening preserves it.
        self.app.files = list(self.files)
        self.app.records = copy.deepcopy(self.records)
        self.app.pages.selection_clear(0, 'end')
        self.app.pages.selection_set(1)
        self.app.mode.set(False)
        self.app.model.set('custom:model')
        with patch('manga_app.app.filedialog.asksaveasfilename', return_value=str(self.path)):
            self.assertTrue(self.app.save_session())
        self.assertFalse(self.app.modified)
        self.app.clear()
        with patch('manga_app.app.filedialog.askopenfilename', return_value=str(self.path)), \
             patch.object(self.app.analyzer, 'analyze') as analyze:
            self.app.open_session()
            analyze.assert_not_called()
        self.assertEqual(self.app.records, self.records)
        self.assertEqual(self.app.files, self.files)
        self.assertEqual(self.app.pages.curselection(), (1,))
        self.assertFalse(self.app.mode.get())
        self.assertEqual(self.app.model.get(), 'custom:model')
        self.app.pages.selection_clear(0, 'end')
        self.app.pages.selection_set(0)
        self.app.show()
        self.assertIn('日本語', self.app.text.get('1.0', 'end'))

    def test_canceled_save_keeps_session_when_clearing(self):
        self.app.add(self.files)
        with patch('manga_app.app.messagebox.askyesnocancel', return_value=True), \
             patch('manga_app.app.filedialog.asksaveasfilename', return_value=''):
            self.app.clear()
        self.assertEqual(len(self.app.files), 3)
        self.assertTrue(self.app.modified)

    def test_invalid_open_keeps_current_results(self):
        self.app.add(self.files)
        self.app.records = copy.deepcopy(self.records)
        self.path.write_text('[]', encoding='utf-8')
        with patch('manga_app.app.filedialog.askopenfilename', return_value=str(self.path)), \
             patch('manga_app.app.messagebox.showerror') as error:
            self.app.open_session()
        error.assert_called_once()
        self.assertEqual(self.app.records, self.records)

    def test_busy_open_is_blocked_but_partial_save_is_allowed(self):
        self.app.add(self.files)
        self.app.records = copy.deepcopy(self.records)
        self.app.busy = True
        with patch('manga_app.app.filedialog.askopenfilename') as dialog:
            self.app.open_session()
            dialog.assert_not_called()
        with patch('manga_app.app.filedialog.asksaveasfilename', return_value=str(self.path)):
            self.assertTrue(self.app.save_session())
        self.app.events.put(('result', (self.files[2], {'file': self.files[2], 'analysis': 'New result.'}, 3)))
        self.app.poll()
        self.assertTrue(self.app.modified)
        self.assertNotIn(self.files[2], load_session(self.path)['records'])
