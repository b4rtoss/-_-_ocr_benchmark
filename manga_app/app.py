import json
import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import ImageTk, ImageDraw
from engine import Analyzer, EXTENSIONS, natural_key, load_image, request_json

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Manga Reader — текст, сцены и сюжет')
        self.geometry('1250x820')
        self.minsize(950, 620)
        self.configure(bg='#F5F6F8')
        self.option_add('*Font', ('Segoe UI', 11))
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
            tkfont.nametofont(name).configure(family='Segoe UI', size=11)
        style = ttk.Style(self)
        style.configure('TFrame', background='white')
        style.configure('TLabel', background='white', font=('Segoe UI', 11))
        style.configure('TCheckbutton', background='white', font=('Segoe UI', 11))
        style.configure('Page.TLabel', background='#F5F6F8')
        style.configure('TPanedwindow', background='white')
        self.files, self.records = [], {}
        self.events, self.cancel = queue.Queue(), threading.Event()
        self.busy = False
        self.analyzer = Analyzer()
        self.mode = tk.BooleanVar(value=True)
        self.lang = tk.StringVar(value='Японский')
        self.model = tk.StringVar(value='qwen3-vl:4b-instruct')
        header = ttk.Frame(self, padding=16)
        header.pack(fill='x', padx=16, pady=(16, 16))
        toolbar = ttk.Frame(header)
        toolbar.pack(fill='x')
        left = ttk.Frame(toolbar)
        left.pack(side='left')
        right = ttk.Frame(toolbar)
        right.pack(side='right')
        self.controls = []
        for index, (title, command) in enumerate([
                ('Добавить страницы', self.add_files), ('Открыть папку', self.add_folder)]):
            holder, button = self.header_button(left, title, command)
            holder.pack(side='left', padx=(0, 8 if index == 0 else 0))
            self.controls.append(button)
        for index, (title, command) in enumerate([
                ('Инструкция', lambda: os.startfile(str(Path(__file__).resolve().parent.parent / 'НАЧАТЬ.md'))),
                ('Настройки', self.open_settings), ('Сохранить', self.save)]):
            holder, _ = self.header_button(right, title, command)
            holder.pack(side='left', padx=(0, 8 if index < 2 else 0))
        second_row = ttk.Frame(header)
        second_row.pack(fill='x', pady=(16, 0))
        ttk.Label(second_row, text='Язык оригинала').pack(side='left', padx=(0, 8))
        self.language_select = ttk.Combobox(second_row, textvariable=self.lang,
            values=['Японский', 'Английский', 'Русский'], state='readonly', width=14)
        self.language_select.pack(side='left', padx=(0, 8))
        holder, button = self.header_button(second_row, 'Анализировать', self.start, primary=True)
        holder.pack(side='left')
        self.controls.append(button)
        self.stop_holder, self.stop_button = self.header_button(second_row, 'Остановить', self.cancel.set)
        panes = ttk.Panedwindow(self, orient='horizontal')
        panes.pack(fill='both', expand=True, padx=16)
        self.pages = tk.Listbox(panes, width=25, exportselection=False)
        self.pages.bind('<<ListboxSelect>>', self.show)
        panes.add(self.pages, weight=1)
        self.canvas = tk.Canvas(panes, bg='#222733', highlightthickness=0)
        self.canvas.bind('<Configure>', self.show)
        panes.add(self.canvas, weight=3)
        frame = ttk.Frame(panes)
        self.text = tk.Text(frame, wrap='word', font=('Segoe UI', 11), padx=16, pady=12)
        scroll = ttk.Scrollbar(frame, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.text.pack(fill='both', expand=True)
        panes.add(frame, weight=3)
        self.status = tk.StringVar(value='Выберите страницы одной главы в порядке чтения. JPG, PNG, WebP, BMP, TIFF.')
        ttk.Label(self, textvariable=self.status, style='Page.TLabel', padding=(16, 12), wraplength=1150).pack(fill='x')
        self.progress = ttk.Progressbar(self)
        self.progress.pack(fill='x', padx=16, pady=(0, 16))
        self.after(150, self.poll)

    def header_button(self, parent, text, command, primary=False):
        background = '#705CF0' if primary else 'white'
        holder = tk.Frame(parent, bg='white', height=38,
                          width=tkfont.nametofont('TkDefaultFont').measure(text) + 32)
        holder.pack_propagate(False)
        button = tk.Button(holder, text=text, command=command, font=('Segoe UI', 11),
            bg=background, fg='white' if primary else '#252936',
            activebackground='#6250DA' if primary else '#F0F1F5',
            activeforeground='white' if primary else '#252936',
            disabledforeground='#B9B0F5' if primary else '#969BA5',
            relief='flat', borderwidth=0, highlightthickness=0, cursor='hand2', padx=0, pady=0)
        button.pack(fill='both', expand=True)
        return holder, button

    def open_settings(self):
        dialog = tk.Toplevel(self)
        dialog.title('Настройки')
        dialog.configure(bg='white')
        dialog.resizable(False, False)
        dialog.transient(self)
        content = ttk.Frame(dialog, padding=16)
        content.pack(fill='both', expand=True)
        ttk.Checkbutton(content, text='Сцены, персонажи и сюжет (Ollama)',
                        variable=self.mode).pack(anchor='w', pady=(0, 16))
        ttk.Label(content, text='Модель Ollama').pack(anchor='w', pady=(0, 8))
        model_entry = ttk.Entry(content, textvariable=self.model, width=36)
        model_entry.pack(fill='x', pady=(0, 16))
        actions = ttk.Frame(content)
        actions.pack(fill='x')
        # The existing clear action stays available without crowding the header.
        holder, clear_button = self.header_button(actions, 'Очистить', self.clear)
        holder.pack(side='left', padx=(0, 8))
        clear_button.configure(state='disabled' if self.busy else 'normal')
        holder, _ = self.header_button(actions, 'Закрыть', dialog.destroy)
        holder.pack(side='right')
        dialog.protocol('WM_DELETE_WINDOW', dialog.destroy)
        dialog.grab_set()
        model_entry.focus_set()

    def add(self, paths):
        self.files = sorted(set(self.files + [str(Path(p).resolve()) for p in paths
                            if Path(p).suffix.lower() in EXTENSIONS]), key=natural_key)
        self.pages.delete(0, 'end')
        for p in self.files:
            self.pages.insert('end', Path(p).name)
        if self.files:
            self.pages.selection_set(0)
            self.show()

    def add_files(self):
        self.add(filedialog.askopenfilenames(filetypes=[('Страницы манги', '*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff')]))

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.add([p for p in Path(folder).iterdir() if p.is_file()])

    def clear(self):
        self.files.clear()
        self.records.clear()
        self.pages.delete(0, 'end')
        self.canvas.delete('all')
        self.text.delete('1.0', 'end')

    def show(self, event=None):
        selected = self.pages.curselection()
        if not selected:
            return
        path = self.files[selected[0]]
        record = self.records.get(path, {})
        self.text.delete('1.0', 'end')
        self.text.insert('end', self.format_record(record) if record else 'Страница ещё не проанализирована.')
        try:
            image = load_image(path)
            draw = ImageDraw.Draw(image)
            for i, region in enumerate(record.get('regions', []), 1):
                poly = [tuple(p) for p in region['polygon']]
                draw.line(poly + poly[:1], fill='#00bb88', width=3)
                draw.text(poly[0], str(i), fill='red', stroke_width=1)
            image.thumbnail((max(100, self.canvas.winfo_width()-16), max(100, self.canvas.winfo_height()-16)))
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.delete('all')
            self.canvas.create_image(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, image=self.photo)
        except Exception as exc:
            self.text.insert('end', '\nНе удалось открыть изображение: ' + str(exc))

    @staticmethod
    def format_record(record):
        if record.get('error'):
            return 'Ошибка: ' + record['error']
        parts = [record.get('analysis', ''), '\nРАСПОЗНАННЫЙ ТЕКСТ (порядок приблизительный)\n']
        parts += [f"{i}. {r['text']}  [{r['confidence']:.0%}]" for i, r in enumerate(record.get('regions', []), 1)]
        if not record.get('regions'):
            parts.append('Текст не найден или OCR недоступен.')
        parts += record.get('warnings', [])
        return '\n'.join(parts)

    def start(self):
        if self.busy or not self.files:
            return
        self.busy = True
        self.stop_holder.pack(side='left', padx=(8, 0))
        self.cancel.clear()
        for button in self.controls:
            button.configure(state='disabled')
        paths = list(self.files)
        language = {'Японский': 'japan', 'Английский': 'en', 'Русский': 'ru'}[self.lang.get()]
        story, model = self.mode.get(), self.model.get().strip()
        self.progress.configure(value=0, maximum=len(paths))
        def worker():
            context = ''
            try:
                if story:
                    self.events.put(('status', 'Проверяю локальную модель Ollama…'))
                    tags = request_json('tags', timeout=5).get('models', [])
                    if model not in [m['name'] for m in tags]:
                        raise RuntimeError(f'Модель {model} не установлена. Выполните: ollama pull {model}')
                for i, path in enumerate(paths):
                    if self.cancel.is_set():
                        break
                    self.events.put(('status', f'{i+1}/{len(paths)}: {Path(path).name}. Первый запуск загружает OCR-модели; это может занять несколько минут.'))
                    try:
                        record = self.analyzer.analyze(path, language, story, model, context)
                        context = (context + '\n' + Path(path).name + '\n' + record['analysis'])[-12000:]
                    except Exception as exc:
                        record = {'file': path, 'error': str(exc)}
                    self.events.put(('result', (path, record, i+1)))
            except Exception as exc:
                self.events.put(('failure', str(exc)))
            finally:
                self.events.put(('done', None))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'status':
                    self.status.set(value)
                elif kind == 'result':
                    path, record, count = value
                    self.records[path] = record
                    self.progress.configure(value=count)
                    self.show()
                elif kind == 'failure':
                    messagebox.showerror('Не удалось запустить анализ', value)
                elif kind == 'done':
                    self.busy = False
                    self.stop_holder.pack_forget()
                    for button in self.controls:
                        button.configure(state='normal')
                    self.status.set('Остановлено.' if self.cancel.is_set() else 'Обработка завершена. Проверьте результаты и сообщения об ошибках. Анализ модели может содержать неточности.')
        except queue.Empty:
            pass
        self.after(150, self.poll)

    def save(self):
        if not self.records:
            messagebox.showinfo('Сохранение', 'Сначала проанализируйте страницы.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON — текст и координаты', '*.json'), ('Текст', '*.txt')])
        if path:
            try:
                records = [self.records[p] for p in self.files if p in self.records]
                content = json.dumps(records, ensure_ascii=False, indent=2) if path.lower().endswith('.json') else '\n\n'.join(r['file']+'\n'+self.format_record(r) for r in records)
                Path(path).write_text(content, encoding='utf-8')
            except Exception as exc:
                messagebox.showerror('Ошибка сохранения', str(exc))

if __name__ == '__main__':
    App().mainloop()
