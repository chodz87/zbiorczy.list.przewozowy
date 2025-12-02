import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional

import pdfplumber
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


SENDER_LINES = [
    'Kersia',
    'ul. Kasztanowa 4',
    '64-320 Niepruszewo',
]

FOOTER_LEFT = 'Podpis nadawcy'
FOOTER_RIGHT = 'Dane przewoźnika'

SUMMARY_TITLE = 'Zbiorczy list przewozowy'

FONT_FILE_NAME = 'DejaVuSans.ttf'
FONT_FALLBACK = 'Helvetica'
FONT_NAME: str  # ustawiane w init_font()

LOGO_FILE_NAME = 'logo_kersia.png'
LOGO_PATH: Optional[str] = None  # ustawiane w get_logo_path()


@dataclass
class OrderData:
    order_number: str
    shipment_number: Optional[str] = None
    net_weight: Optional[float] = None
    delivery_address: Optional[str] = None


def init_font() -> str:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    ttf_path = os.path.join(base_dir, FONT_FILE_NAME)
    if os.path.exists(ttf_path):
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', ttf_path))
            return 'DejaVuSans'
        except Exception:
            pass
    return FONT_FALLBACK


def get_logo_path() -> Optional[str]:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    path = os.path.join(base_dir, LOGO_FILE_NAME)
    return path if os.path.exists(path) else None


FONT_NAME = init_font()
LOGO_PATH = get_logo_path()


def parse_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    text = text.replace('\u00a0', '').replace(' ', '').replace(',', '.')
    try:
        return float(text)
    except ValueError:
        return None


def find_net_weight_summary(lines: List[str]) -> Optional[float]:
    if not any('Ilość palet' in l for l in lines):
        return None
    idx = None
    for i, line in enumerate(lines):
        if (
            'Całkowita waga netto' in line
            or 'Total Net Weight' in line
            or 'Całkowita objęto' in line
            or 'Total Volume' in line
        ):
            idx = i
    if idx is None:
        return None
    for line in lines[idx + 1: idx + 8]:
        nums = re.findall(r'[0-9.,]+', line)
        values: List[float] = []
        for n in nums:
            v = parse_float(n)
            if v is not None:
                values.append(v)
        if len(values) >= 2:
            return values[-1]
    return None


def find_shipment_number(lines: List[str]) -> Optional[str]:
    for i, line in enumerate(lines):
        low = line.lower()
        if ('numer przesy' in low) or ('nr przesy' in low):
            for j in range(i, min(i + 3, len(lines))):
                nums = re.findall(r'\d{6,}', lines[j])
                if nums:
                    return nums[-1]
    for i, line in enumerate(lines):
        if 'QUALITY CERTIFICATE' in line.upper():
            for j in range(i + 1, min(i + 5, len(lines))):
                nums = re.findall(r'\d{6,}', lines[j])
                if nums:
                    return nums[-1]
    return None


def find_delivery_address(lines: List[str]) -> Optional[str]:
    start = None
    end = None
    for i, line in enumerate(lines):
        if 'Adresat/Consignee' in line:
            start = i + 1
        if start is not None and 'Adres nabywcy' in line:
            end = i
            break
    if start is None or end is None or end <= start:
        return None
    block = [l.strip() for l in lines[start:end] if l.strip()]
    if not block:
        return None
    cleaned: List[str] = []
    for l in block:
        if 'Nadawca/Consignor' in l:
            continue
        words = [w for w in l.split() if w.upper() != 'NIEPRUSZEWO']
        if not words:
            continue
        cleaned.append(' '.join(words))
    if not cleaned:
        return None
    segments: List[List[str]] = []
    current: List[str] = []
    for l in cleaned:
        current.append(l)
        if l.strip().upper() == 'POLAND':
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    if not segments:
        return None
    def is_plant_segment(seg: List[str]) -> bool:
        txt = ' '.join(seg).upper()
        return ('64-320' in txt) or (' BUK' in txt) or ('KASZTANOWA' in txt)
    non_plant_segments = [seg for seg in segments if not is_plant_segment(seg)]
    if non_plant_segments:
        selected = non_plant_segments[0]
    else:
        if len(segments) >= 2:
            selected = segments[1]
        else:
            selected = segments[-1]
    selected = [s for s in selected if s.strip()]
    if not selected:
        return None
    return ', '.join(selected)


class PackingListExtractor:
    def __init__(self, pdf_path: str, order_numbers: List[str]) -> None:
        self.pdf_path = pdf_path
        self.order_numbers = order_numbers

    def extract(self) -> Tuple[Dict[str, OrderData], Set[int]]:
        results: Dict[str, OrderData] = {
            nr: OrderData(order_number=nr) for nr in self.order_numbers
        }
        pages_to_keep: Set[int] = set()
        with pdfplumber.open(self.pdf_path) as pdf:
            pages_text: List[str] = []
            pages_lines: List[List[str]] = []
            for page in pdf.pages:
                text = page.extract_text() or ''
                pages_text.append(text)
                pages_lines.append(text.split('\n') if text else [])
            order_pages: Dict[str, Set[int]] = {nr: set() for nr in self.order_numbers}
            for page_idx, text in enumerate(pages_text):
                for nr in self.order_numbers:
                    if nr in text:
                        order_pages[nr].add(page_idx)
                        pages_to_keep.add(page_idx)
            for nr in self.order_numbers:
                pages = sorted(order_pages[nr])
                if not pages:
                    continue
                first_page_idx = pages[0]
                lines_first = pages_lines[first_page_idx]
                data = results[nr]
                if not data.delivery_address:
                    addr = find_delivery_address(lines_first)
                    if addr:
                        data.delivery_address = addr
                if not data.shipment_number:
                    shipment = find_shipment_number(lines_first)
                    if shipment is None:
                        next_idx = first_page_idx + 1
                        if next_idx < len(pages_lines):
                            shipment = find_shipment_number(pages_lines[next_idx])
                            if shipment:
                                pages_to_keep.add(next_idx)
                    if shipment:
                        data.shipment_number = shipment
                net_weight = self._extract_net_weight(pages, pages_lines)
                data.net_weight = net_weight
                results[nr] = data
        return results, pages_to_keep

    def _extract_net_weight(
        self, page_indices: List[int], pages_lines: List[List[str]]
    ) -> Optional[float]:
        net: Optional[float] = None
        for idx in page_indices:
            val = find_net_weight_summary(pages_lines[idx])
            if val is not None:
                net = val
        if net is not None:
            return net
        for idx in page_indices:
            next_idx = idx + 1
            if next_idx < len(pages_lines):
                val = find_net_weight_summary(pages_lines[next_idx])
                if val is not None:
                    net = val
                    break
        return net


class ExcelExporter:
    @staticmethod
    def save(results: Dict[str, OrderData], order_numbers: List[str], output_path: str) -> None:
        rows: List[Dict[str, object]] = []
        total_net = 0.0
        for nr in order_numbers:
            data = results.get(nr)
            if not data:
                continue
            if data.net_weight is not None:
                try:
                    total_net += float(data.net_weight)
                except (TypeError, ValueError):
                    pass
            rows.append({
                'Numer zamówienia': data.order_number,
                'Numer przesyłki': data.shipment_number,
                'Całkowita waga netto (kg)': data.net_weight,
                'Adres dostawy': data.delivery_address,
            })
        rows.append({
            'Numer zamówienia': 'RAZEM',
            'Numer przesyłki': '',
            'Całkowita waga netto (kg)': round(total_net, 2),
            'Adres dostawy': '',
        })
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False)


class SummaryPdfBuilder:
    def __init__(self, font_name: str, logo_path: Optional[str]) -> None:
        self.font_name = font_name
        self.logo_path = logo_path

    def build(self, summary_path: str, results: Dict[str, OrderData],
              order_numbers: List[str]) -> None:
        c = canvas.Canvas(summary_path, pagesize=landscape(A4))
        width, height = landscape(A4)

        def draw_header(title_suffix: str = '') -> float:
            c.setFont(self.font_name, 16)
            title = SUMMARY_TITLE + (f' {title_suffix}' if title_suffix else '')
            tw = c.stringWidth(title, self.font_name, 16)
            c.drawString((width - tw) / 2, height - 40, title)
            c.setFont(self.font_name, 10)
            y_sender = height - 65
            for line in SENDER_LINES:
                c.drawString(40, y_sender, line)
                y_sender -= 12
            date_str = datetime.now().strftime('%Y-%m-%d')
            date_font_size = 10
            c.setFont(self.font_name, date_font_size)
            dw = c.stringWidth(date_str, self.font_name, date_font_size)
            date_y = height - 65
            c.drawString(width - 40 - dw, date_y, date_str)
            if self.logo_path:
                logo_w = 130
                logo_h = 55
                logo_x = width - 40 - logo_w
                logo_y = date_y - logo_h - 5
                c.drawImage(
                    self.logo_path,
                    logo_x,
                    logo_y,
                    width=logo_w,
                    height=logo_h,
                    preserveAspectRatio=True,
                    mask='auto',
                )
                bottom_logo = logo_y
            else:
                bottom_logo = height - 80
            bottom_sender = y_sender
            return min(bottom_sender, bottom_logo) - 15

        def draw_footer() -> None:
            c.setFont(self.font_name, 10)
            y = 40
            left_x = width * 0.25
            right_x = width * 0.60
            c.drawString(left_x, y, FOOTER_LEFT)
            c.drawString(right_x, y, FOOTER_RIGHT)
            c.setFont(self.font_name, 9)
            dots = '.................................'
            c.drawString(left_x, y - 20, dots)
            c.drawString(right_x, y - 20, dots)

        start_y = draw_header()
        base_row_height = 26
        col_x = [40, 150, 280, 380]
        col_widths = [110, 120, 90, width - 380 - 40]
        headers = [
            'Numer zamówienia',
            'Numer przesyłki',
            'Całkowita waga netto (kg)',
            'Adres dostawy',
        ]

        def split_header_texts() -> List[List[str]]:
            out: List[List[str]] = []
            for h in headers:
                if 'Całkowita waga netto' in h:
                    out.append(['Całkowita waga', 'netto (kg)'])
                else:
                    out.append([h])
            return out

        def draw_row(y: float, height_row: float,
                     texts_lines: List[List[str]],
                     bold_flags: List[bool]) -> None:
            line_h = 11
            for i in range(4):
                x = col_x[i]
                w = col_widths[i]
                c.rect(x, y - height_row, w, height_row)
                lines_raw = texts_lines[i]
                if not isinstance(lines_raw, list):
                    lines = [lines_raw]
                else:
                    lines = lines_raw
                font_size = 9 if bold_flags[i] else 8
                c.setFont(self.font_name, font_size)
                if not lines:
                    continue
                total_th = line_h * len(lines)
                ty = y - 6 - (height_row - total_th) / 2
                for line in lines:
                    if isinstance(line, (int, float)):
                        line = f"{float(line):.2f}"
                    text = '' if line is None else str(line)
                    c.drawString(x + 3, ty, text)
                    ty -= line_h

        def new_page(suffix: str = '(cd.)') -> float:
            c.showPage()
            nonlocal width, height
            width, height = landscape(A4)
            return draw_header(suffix)

        y = start_y
        header_texts = split_header_texts()
        header_bold = [True, True, True, True]
        header_h = base_row_height
        draw_row(y, header_h, header_texts, header_bold)
        y -= header_h
        total_net = 0.0
        for nr in order_numbers:
            data = results.get(nr)
            if not data:
                continue
            if data.net_weight is not None:
                try:
                    total_net += float(data.net_weight)
                except (TypeError, ValueError):
                    pass
            addr = data.delivery_address or ''
            max_chars = 80
            addr_lines: List[str] = []
            tmp = addr
            while len(tmp) > max_chars:
                cut = tmp.rfind(',', 0, max_chars)
                if cut == -1:
                    cut = max_chars
                addr_lines.append(tmp[:cut].strip())
                tmp = tmp[cut + 1:].strip()
            if tmp:
                addr_lines.append(tmp)
            if not addr_lines:
                addr_lines = ['']
            lines_in_cell = max(1, len(addr_lines))
            row_h = max(base_row_height, lines_in_cell * 11 + 4)
            if y - row_h < 70:
                draw_footer()
                y = new_page()
                draw_row(y, header_h, header_texts, header_bold)
                y -= header_h
            row_texts = [
                [data.order_number],
                [data.shipment_number or ''],
                [f"{float(data.net_weight):.2f}"] if data.net_weight is not None else [''],
                addr_lines,
            ]
            row_bold = [False, False, False, False]
            draw_row(y, row_h, row_texts, row_bold)
            y -= row_h + 2
        y -= 10
        sum_row_h = base_row_height
        if y - sum_row_h < 70:
            draw_footer()
            y = new_page()
            draw_row(y, header_h, header_texts, header_bold)
            y -= header_h
        big_width = col_widths[0] + col_widths[1] + col_widths[2]
        c.rect(col_x[0], y - sum_row_h, big_width, sum_row_h)
        c.rect(col_x[3], y - sum_row_h, col_widths[3], sum_row_h)
        label = 'CAŁKOWITA WAGA NETTO (kg)'
        c.setFont(self.font_name, 10)
        c.drawString(col_x[0] + 6, y - sum_row_h / 2, label)
        sum_txt = f"{round(total_net, 2):.2f}"
        c.setFont(self.font_name, 12)
        c.drawString(col_x[3] + 6, y - sum_row_h / 2, sum_txt)
        c.drawString(col_x[3] + 6.4, y - sum_row_h / 2, sum_txt)
        draw_footer()
        c.showPage()
        c.save()


def save_combined_pdf(
    original_pdf_path: str,
    output_pdf_path: str,
    page_indices: Set[int],
    summary_pdf_path: Optional[str] = None,
) -> None:
    reader = PdfReader(original_pdf_path)
    writer = PdfWriter()
    for idx in sorted(page_indices):
        if 0 <= idx < len(reader.pages):
            writer.add_page(reader.pages[idx])
    if summary_pdf_path and os.path.exists(summary_pdf_path):
        s_reader = PdfReader(summary_pdf_path)
        for p in s_reader.pages:
            writer.add_page(p)
    if len(writer.pages) == 0:
        raise ValueError('Brak stron do zapisania w nowym PDF (nic nie znaleziono).')
    with open(output_pdf_path, 'wb') as f:
        writer.write(f)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('List przewozowy: Excel + PDF + zbiorczy list (refaktoryzacja)')
        self.pdf_path: Optional[str] = None
        self._build_widgets()

    def _build_widgets(self) -> None:
        self.btn_pdf = tk.Button(
            self.root, text='Wybierz plik PDF', command=self.choose_pdf
        )
        self.btn_pdf.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.lbl_pdf = tk.Label(self.root, text='Nie wybrano pliku')
        self.lbl_pdf.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        tk.Label(
            self.root, text='Numery zamówień (po przecinku):'
        ).grid(row=1, column=0, padx=10, pady=5, sticky='nw')
        self.entry_orders = tk.Text(self.root, height=3, width=40)
        self.entry_orders.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        self.btn_generate = tk.Button(
            self.root,
            text='Generuj Excel + PDF (ze zbiorczą stroną)',
            command=self.generate_files,
        )
        self.btn_generate.grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.lbl_status = tk.Label(self.root, text='', fg='blue')
        self.lbl_status.grid(row=2, column=1, padx=10, pady=10, sticky='w')

    def choose_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title='Wybierz plik PDF',
            filetypes=[('PDF files', '*.pdf'), ('All files', '*.*')],
        )
        if path:
            self.pdf_path = path
            self.lbl_pdf.config(text=path)

    @staticmethod
    def _parse_order_numbers(text: str) -> List[str]:
        return [x.strip() for x in text.replace('\n', ',').split(',') if x.strip()]

    def generate_files(self) -> None:
        if not self.pdf_path:
            messagebox.showerror('Błąd', 'Najpierw wybierz plik PDF.')
            return
        text_numbers = self.entry_orders.get('1.0', tk.END).strip()
        if not text_numbers:
            messagebox.showerror('Błąd', 'Podaj co najmniej jeden numer zamówienia.')
            return
        order_numbers = self._parse_order_numbers(text_numbers)
        if not order_numbers:
            messagebox.showerror('Błąd', 'Nie udało się odczytać numerów zamówienia.')
            return
        try:
            self.lbl_status.config(text='Przetwarzanie...', fg='blue')
            self.root.update_idletasks()
            extractor = PackingListExtractor(self.pdf_path, order_numbers)
            results, pages_to_keep = extractor.extract()
            if not pages_to_keep:
                raise ValueError('Nie znaleziono żadnych stron z podanymi numerami zamówień.')
            base_dir = os.path.dirname(self.pdf_path)
            today = datetime.now().strftime('%Y-%m-%d')
            excel_path = os.path.join(base_dir, f'list_przewozowy_{today}.xlsx')
            pdf_path = os.path.join(base_dir, f'list_przewozowy_{today}.pdf')
            summary_temp = os.path.join(base_dir, f'list_przewozowy_{today}_zbiorczy_tmp.pdf')
            ExcelExporter.save(results, order_numbers, excel_path)
            summary_builder = SummaryPdfBuilder(FONT_NAME, LOGO_PATH)
            summary_builder.build(summary_temp, results, order_numbers)
            save_combined_pdf(self.pdf_path, pdf_path, pages_to_keep, summary_temp)
            if os.path.exists(summary_temp):
                try:
                    os.remove(summary_temp)
                except Exception:
                    pass
            self.lbl_status.config(
                text=f'Zapisano:\n{excel_path}\n{pdf_path}', fg='green'
            )
            messagebox.showinfo(
                'Sukces', f'Zapisano pliki:\n{excel_path}\n{pdf_path}'
            )
        except Exception as e:
            self.lbl_status.config(text='Błąd podczas przetwarzania', fg='red')
            messagebox.showerror('Błąd', f'Wystąpił błąd:\n{e}')


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
