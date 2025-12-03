import re
import io
import os
import tempfile
from datetime import datetime

import math

import pdfplumber
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import streamlit as st

# =================== MAPA ADRESÓW ===================

ADRESY_ZLECEN = {
    "59743": "EMI Chopina 22 62-025 Kostrzyn Wlkp POLAND",
    "59745": "EMI Chopina 22 62-025 Kostrzyn Wlkp POLAND",
    "60060": "GR MAREK FRĄTCZAK Kuźnica Czarnkowska, Zamkowa 1 64-700 Czarnków POLAND",
    "60071": "ROBERT GREGOR Wiśniowa 12 88-230 Piotrków Kujawski POLAND",
    "60084": "TYMBARK o/TYMBARK Tymbark 156 34-650 Tymbark POLAND",
    "60103": "CHEMEA JOANNA DMOWSKA / Pet Cuisine Sp. z o.o. ul. Narodowych Sił Zbrojnych 2 09-400 Płock POLAND",
    "60130": "GOSPODARSTWO ROLNE JAN KASZTELAN Rempin, Szlachecka 20 09-213 Gozdowo POLAND",
    "60141": "OKRĘGOWA SPÓŁDZIELNIA MLECZARSKA W ŁOWICZU Przemysłowa 3 99-400 Łowicz POLAND",
    "60153": "TEFA SP. Z O. O. SP. K. Tomasz Szpila Kraśnik Dolny 40A 59-700 Kraśnik Dolny POLAND",
    "60175": "BOFERM SP. Z O.O. Górki 13 08-210 Platerów POLAND",
}

# =================== FONT + LOGO ===================

def init_font():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    ttf_path = os.path.join(base_dir, "DejaVuSans.ttf")
    if os.path.exists(ttf_path):
        try:
            pdfmetrics.registerFont(TTFont("DejaVuSans", ttf_path))
            return "DejaVuSans"
        except Exception:
            pass
    return "Helvetica"


def get_logo_path():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    path = os.path.join(base_dir, "logo_kersia.png")
    return path if os.path.exists(path) else None


FONT_NAME = init_font()
LOGO_PATH = get_logo_path()

# =================== POMOCNICZE ===================

def wyczysc_liczbe(val):
    """Zamienia '1 234,56' / '1p' / NaN na float albo None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return None
        return float(val)
    text = str(val).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace("\xa0", "").replace(" ", "")
    nums = re.findall(r"[0-9.,]+", text)
    if not nums:
        return None
    num = nums[0].replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


def _normalize_colname(name: str) -> str:
    """Normalizacja nazw kolumn (bez polskich znaków, spacje itp.)."""
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().lower()
    trans = str.maketrans("ąćęłńóśźż", "acelnoszz")
    name = name.translate(trans)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name

# =================== WAGA NETTO ===================

def znajdz_wage_netto_podsumowanie(linie):
    """
    Szuka wagi netto w sekcji podsumowania:
    - wiersz nagłówka zawiera 'Ilość palet' oraz 'Całkowita waga netto'
    - wiersz z wartościami jest poniżej; bierzemy drugą liczbę (kolumna netto)
    """
    if not linie:
        return None

    header_idx = None
    for i, linia in enumerate(linie):
        if "Ilość palet" in linia and (
            "Całkowita waga netto" in linia or "Total Net Weight" in linia
        ):
            header_idx = i
            break
    if header_idx is None:
        for i, linia in enumerate(linie):
            if "Całkowita waga netto" in linia or "Total Net Weight" in linia:
                header_idx = i
                break
    if header_idx is None:
        return None

    for j in range(header_idx + 1, min(header_idx + 7, len(linie))):
        row = linie[j].strip()
        if not row:
            continue
        nums = re.findall(r"[0-9.,]+", row)
        values = [wyczysc_liczbe(n) for n in nums]
        values = [v for v in values if v is not None]
        if len(values) >= 2:
            return values[1]
    return None

# =================== NUMER PRZESYŁKI ===================

def znajdz_numer_przesylki(linie):
    if not linie:
        return None

    for i, linia in enumerate(linie):
        low = linia.lower()
        if "numer przesy" in low or "nr przesy" in low:
            for j in range(i, min(i + 3, len(linie))):
                nums = re.findall(r"\d{6,}", linie[j])
                if nums:
                    return nums[-1]

    for i, linia in enumerate(linie):
        if "QUALITY CERTIFICATE" in linia.upper():
            for j in range(i + 1, min(i + 5, len(linie))):
                nums = re.findall(r"\d{6,}", linie[j])
                if nums:
                    return nums[-1]
    return None

# =================== ADRES DOSTAWY ===================

def znajdz_adres_dostawy(linie):
    """Adres między 'Adresat/Consignee' a 'Adres nabywcy', bez adresu zakładu."""
    if not linie:
        return None

    start = None
    end = None
    for i, linia in enumerate(linie):
        if "Adresat/Consignee" in linia:
            start = i + 1
        if start is not None and "Adres nabywcy" in linia:
            end = i
            break

    if start is None or end is None or end <= start:
        return None

    blok = [l.strip() for l in linie[start:end] if l.strip()]
    if not blok:
        return None

    cleaned = []
    for l in blok:
        upper = l.upper()
        if "NIEPRUSZEWO" in upper or "KASZTANOWA" in upper or "64-320" in upper or " BUK" in upper:
            continue
        cleaned.append(l.strip())
    if not cleaned:
        return None

    text = " ".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =================== PRZETWARZANIE ZAMÓWIEŃ W PDF ===================

def przetworz_zamowienia(pdf_path, numery_zamowien):
    wyniki = {
        nr: {"netto": None, "adres_dostawy": None, "numer_przesylki": None}
        for nr in numery_zamowien
    }
    strony_do_zachowania = set()

    with pdfplumber.open(pdf_path) as pdf:
        pages_text = []
        pages_lines = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
            pages_lines.append(text.split("\n") if text else [])

        zamowienie_strony = {nr: set() for nr in numery_zamowien}

        for page_index, text in enumerate(pages_text):
            for nr in numery_zamowien:
                if str(nr) in text:
                    zamowienie_strony[nr].add(page_index)
                    strony_do_zachowania.add(page_index)

        for nr in numery_zamowien:
            strony = sorted(zamowienie_strony[nr])
            if not strony:
                continue

            first_page_idx = strony[0]

            if wyniki[nr]["adres_dostawy"] is None:
                adres = znajdz_adres_dostawy(pages_lines[first_page_idx])
                if adres:
                    wyniki[nr]["adres_dostawy"] = adres

            if wyniki[nr]["numer_przesylki"] is None:
                numer_p = None
                for page_idx in strony:
                    numer_p = znajdz_numer_przesylki(pages_lines[page_idx])
                    if numer_p:
                        break
                if not numer_p:
                    last_idx = max(strony)
                    if last_idx + 1 < len(pages_lines):
                        numer_p = znajdz_numer_przesylki(pages_lines[last_idx + 1])
                        if numer_p:
                            strony_do_zachowania.add(last_idx + 1)
                if numer_p:
                    wyniki[nr]["numer_przesylki"] = numer_p

            netto = None
            for idx in strony:
                val = znajdz_wage_netto_podsumowanie(pages_lines[idx])
                if val is not None:
                    netto = val

            if netto is None:
                for idx in strony:
                    next_idx = idx + 1
                    if next_idx < len(pages_lines):
                        val = znajdz_wage_netto_podsumowanie(pages_lines[next_idx])
                        if val is not None:
                            netto = val
                            strony_do_zachowania.add(next_idx)
                            break

            wyniki[nr]["netto"] = netto

    for nr, dane in wyniki.items():
        addr = ADRESY_ZLECEN.get(str(nr))
        if addr:
            dane["adres_dostawy"] = addr

    return wyniki, strony_do_zachowania

# =================== PARSOWANIE WKLEJONEJ TABELI ===================

def parse_groups_from_pasted(text: str):
    """
    Oczekuje wklejonego tekstu z Excela:
    - wariant A: z nagłówkiem kolumn: ZLECENIE | ilość palet | przewoźnik | mp
    - wariant B: BEZ nagłówka – same dane w 4 kolumnach, w kolejności:
        ZLECENIE | ilość palet | przewoźnik | mp
    """
    if not text.strip():
        raise ValueError("Pole z tabelą jest puste – wklej dane z Excela.")

    if "\t" in text:
        sep = "\t"
    elif ";" in text:
        sep = ";"
    else:
        sep = ","

    buffer = io.StringIO(text.strip())
    df_try = pd.read_csv(buffer, sep=sep, engine="python")

    norm_to_orig = {}
    for col in df_try.columns:
        norm = _normalize_colname(col)
        norm_to_orig[norm] = col

    has_header = "zlecenie" in norm_to_orig and "mp" in norm_to_orig

    if has_header:
        df = df_try
        col_zlec = norm_to_orig["zlecenie"]
        col_mp = norm_to_orig["mp"]
        col_ilosc = norm_to_orig.get("iloscpalet")
        col_przew = norm_to_orig.get("przewoznik")
    else:
        buffer2 = io.StringIO(text.strip())
        df = pd.read_csv(buffer2, sep=sep, header=None, engine="python")
        if df.shape[1] < 4:
            raise ValueError(
                "W trybie bez nagłówków oczekuję 4 kolumn w kolejności: "
                "ZLECENIE, ilość palet, przewoźnik, mp."
            )
        df = df.iloc[:, :4].copy()
        col_zlec = "ZLECENIE"
        col_ilosc = "ilosc_palet"
        col_przew = "przewoznik"
        col_mp = "mp"
        df.columns = [col_zlec, col_ilosc, col_przew, col_mp]

    groups = []
    for _, row in df.iterrows():
        zlec_raw = row.get(col_zlec, "")
        zlec = str(zlec_raw).strip()
        if not zlec or zlec.lower() == "nan" or _normalize_colname(zlec) == "zlecenie":
            continue

        raw_mp = row.get(col_mp, None)
        mp_val = wyczysc_liczbe(raw_mp) if raw_mp is not None else None

        ilosc_pal_txt = None
        if col_ilosc is not None:
            ilosc_pal_txt = row.get(col_ilosc, None)
            if pd.isna(ilosc_pal_txt):
                ilosc_pal_txt = None

        przew = None
        if col_przew is not None:
            przew = row.get(col_przew, None)
            if pd.isna(przew):
                przew = None

        parts = [p.strip() for p in re.split(r"[+]", zlec) if p and str(p).strip().lower() != "nan"]
        if not parts:
            continue

        groups.append({
            "label": zlec,
            "orders": parts,
            "mp": mp_val,
            "ilosc_pal_tekst": ilosc_pal_txt,
            "przewoznik": przew,
        })

    if not groups:
        raise ValueError("Nie udało się odczytać żadnych zleceń z wklejonej tabeli.")

    cols_for_preview = []
    for c in [col_zlec, col_ilosc, col_przew, col_mp]:
        if c is not None and c in df.columns:
            cols_for_preview.append(c)
    df_preview = df[cols_for_preview].copy()

    return groups, df_preview

# =================== PODSUMOWANIE GRUP ===================

def zbuduj_podsumowanie_grup(wyniki, groups):
    """
    - Ilość palet (mp) – liczba z kolumny mp (sumowana)
    - Ilość palet – tekst z Excela
    - Jeżeli zlecenie jest łączone, adres bierzemy z tego zamówienia,
      które ma większą wagę netto (jeśli brak wag – pierwszy dostępny adres).
    """
    rows = []
    total_netto = 0.0
    total_palety = 0.0

    for g in groups:
        label = g["label"]
        orders = g["orders"]
        mp = g.get("mp")
        ilosc_txt = g.get("ilosc_pal_tekst")
        przew = g.get("przewoznik")

        netto_sum = 0.0
        any_netto = False
        przesylki = []

        best_addr = None
        best_addr_netto = -1.0
        fallback_addr = None

        any_data_for_group = False

        for nr in orders:
            d = wyniki.get(nr, {})

            if any(d.get(k) is not None for k in ("netto", "adres_dostawy", "numer_przesylki")):
                any_data_for_group = True

            n = d.get("netto")
            v = wyczysc_liczbe(n)
            if v is not None:
                any_netto = True
                netto_sum += v

                if d.get("adres_dostawy") and v > best_addr_netto:
                    best_addr_netto = v
                    best_addr = d["adres_dostawy"]

            if fallback_addr is None and d.get("adres_dostawy"):
                fallback_addr = d["adres_dostawy"]

            if d.get("numer_przesylki"):
                przesylki.append(str(d["numer_przesylki"]))

        if not any_data_for_group:
            continue

        adres = best_addr if best_addr is not None else fallback_addr

        netto_value = netto_sum if any_netto else None
        if netto_value is not None:
            total_netto += netto_value

        if mp is not None:
            val_mp = wyczysc_liczbe(mp)
            if val_mp is not None:
                total_palety += val_mp

        rows.append({
            "ZLECENIE": label,
            "Ilość palet (mp)": mp,
            "Ilość palet": ilosc_txt,
            "Przewoźnik": przew,
            "Numery zamówień": "+".join(orders),
            "Numery przesyłek": "+".join(sorted(set(przesylki))) if przesylki else "",
            "Całkowita waga netto (kg)": round(netto_value, 2) if netto_value is not None else None,
            "Adres dostawy": adres,
        })

    rows.append({
        "ZLECENIE": "RAZEM",
        "Ilość palet (mp)": total_palety,
        "Ilość palet": "",
        "Przewoźnik": "",
        "Numery zamówień": "",
        "Numery przesyłek": "",
        "Całkowita waga netto (kg)": round(total_netto, 2),
        "Adres dostawy": "",
    })

    df = pd.DataFrame(rows)
    return df, total_netto, total_palety

# =================== ZBIORCZY LIST (PDF) ===================

def utworz_zbiorczy_pdf(summary_path, df_summary, total_netto, total_palety, date_str):
    c = canvas.Canvas(summary_path, pagesize=landscape(A4))
    width, height = landscape(A4)

    def draw_header(title_suffix=""):
        c.setFont(FONT_NAME, 16)
        base_title = "Zbiorczy list przewozowy"
        if title_suffix:
            base_title += f" {title_suffix}"
        title_width = c.stringWidth(base_title, FONT_NAME, 16)
        c.drawString((width - title_width) / 2, height - 40, base_title)

        c.setFont(FONT_NAME, 10)
        sender_lines = [
            "Kersia",
            "ul. Kasztanowa 4",
            "64-320 Niepruszewo",
        ]
        y_sender = height - 65
        for line in sender_lines:
            c.drawString(40, y_sender, line)
            y_sender -= 12

        date_font_size = 10
        c.setFont(FONT_NAME, date_font_size)
        date_width = c.stringWidth(date_str, FONT_NAME, date_font_size)
        date_y = height - 65
        c.drawString(width - 40 - date_width, date_y, date_str)

        if LOGO_PATH is not None:
            logo_w = 130
            logo_h = 55
            logo_x = width - 40 - logo_w
            logo_y = date_y - logo_h - 5
            c.drawImage(
                LOGO_PATH,
                logo_x,
                logo_y,
                width=logo_w,
                height=logo_h,
                preserveAspectRatio=True,
                mask="auto",
            )
            bottom_logo = logo_y
        else:
            bottom_logo = height - 80

        bottom_sender = y_sender
        table_start_y = min(bottom_sender, bottom_logo) - 15
        return table_start_y

    def draw_footer():
        c.setFont(FONT_NAME, 10)
        y = 40
        left_x = width * 0.25
        right_x = width * 0.60

        c.drawString(left_x, y, "Podpis nadawcy")
        c.drawString(right_x, y, "Dane przewoźnika")

        c.setFont(FONT_NAME, 9)
        line_text = "................................."
        c.drawString(left_x, y - 20, line_text)
        c.drawString(right_x, y - 20, line_text)

    def wrap_address(adres, max_chars=55):
        """Ładniejsze zawijanie – ciaśniej, żeby nie wychodziło z komórek."""
        if adres is None:
            return [""]
        text = str(adres).strip()
        if not text:
            return [""]

        parts = [p.strip() for p in text.split(",") if p.strip()]
        lines = []

        def split_long(part):
            local = []
            words = part.split()
            current = ""
            for w in words:
                if not current:
                    current = w
                elif len(current) + 1 + len(w) <= max_chars:
                    current += " " + w
                else:
                    local.append(current)
                    current = w
            if current:
                local.append(current)
            return local

        if parts:
            for part in parts:
                if len(part) <= max_chars:
                    lines.append(part)
                else:
                    lines.extend(split_long(part))
        else:
            lines.extend(split_long(text))
        return lines or [""]

    start_y = draw_header()
    base_row_height = 26

    left_margin = 40
    right_margin = 40
    available_width = width - left_margin - right_margin

    col_widths = [
        0.10 * available_width,  # ZLECENIE
        0.10 * available_width,  # Ilość palet (mp)
        0.20 * available_width,  # Numery przesyłek
        0.20 * available_width,  # Waga netto
        0.40 * available_width,  # Adres
    ]

    col_x = [left_margin]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    headers = [
        "ZLECENIE",
        "Ilość palet (mp)",
        "Numery przesyłek",
        "Całkowita waga netto (kg)",
        "Adres dostawy",
    ]

    def split_header_texts():
        res = []
        for h in headers:
            if "Całkowita waga netto" in h:
                res.append(["Całkowita waga", "netto (kg)"])
            else:
                res.append([h])
        return res

    def draw_row_frame_and_text(y, h, texts_lines, bold_flags):
        """
        Rysuje komórki tabeli z wyśrodkowaniem,
        pilnuje też żeby tekst nie wyszedł poza komórkę (bezpieczny margines).
        """
        line_height = 11
        num_cols = len(col_x)
        safe_margin = 3  # minimalny margines od linii komórki

        for i in range(num_cols):
            x = col_x[i]
            w = col_widths[i]

            c.rect(x, y - h, w, h)

            lines = texts_lines[i]
            if not isinstance(lines, (list, tuple)):
                lines = ["" if lines is None else str(lines)]
            else:
                lines = ["" if ln is None else str(ln) for ln in lines]

            if len(lines) == 0:
                continue

            total_text_height = len(lines) * line_height
            y_center = y - h / 2
            start_y = y_center + total_text_height / 2 - line_height

            font_size = 10 if bold_flags[i] else 9
            c.setFont(FONT_NAME, font_size)

            for j, txt in enumerate(lines):
                text_width = c.stringWidth(txt, FONT_NAME, font_size)

                # jeśli tekst się mieści – centrowanie,
                # jeśli nie – wyrównanie do lewej z marginesem
                if text_width <= w - 2 * safe_margin:
                    text_x = x + (w - text_width) / 2
                else:
                    text_x = x + safe_margin

                text_y = start_y - j * line_height
                c.drawString(text_x, text_y, txt)

    def new_page(suffix="(cd.)"):
        c.showPage()
        return draw_header(suffix)

    y = start_y

    header_texts = split_header_texts()
    header_bold = [True] * len(headers)
    header_height = base_row_height
    draw_row_frame_and_text(y, header_height, header_texts, header_bold)
    y -= header_height

    records = df_summary[df_summary["ZLECENIE"] != "RAZEM"].to_dict("records")

    for rec in records:
        zlec = rec.get("ZLECENIE", "")
        mp_val = rec.get("Ilość palet (mp)", "")
        przes = rec.get("Numery przesyłek", "")
        netto = rec.get("Całkowita waga netto (kg)", "")
        adres = rec.get("Adres dostawy", "")

        adres_lines = wrap_address(adres)

        lines_in_cell = max(1, len(adres_lines))
        row_height = max(base_row_height, lines_in_cell * 11 + 4)

        if y - row_height < 80:
            draw_footer()
            y = new_page()
            draw_row_frame_and_text(y, header_height, header_texts, header_bold)
            y -= header_height

        texts_lines = [
            [str(zlec)],
            ["" if mp_val is None else str(mp_val)],
            [str(przes)],
            [f"{float(netto):.2f}"] if netto not in [None, ""] else [""],
            adres_lines,
        ]
        bold_flags = [False] * len(headers)

        draw_row_frame_and_text(y, row_height, texts_lines, bold_flags)
        y -= row_height
        y -= 2

    y -= 10

    row_height = base_row_height
    total_rows_height = row_height * 2

    if y - total_rows_height < 80:
        draw_footer()
        y = new_page()

    big_width = sum(col_widths[:-1])
    last_x = col_x[-1]
    last_w = col_widths[-1]

    c.rect(col_x[0], y - row_height, big_width, row_height)
    c.rect(last_x, y - row_height, last_w, row_height)

    label1 = "CAŁKOWITA WAGA NETTO (kg)"
    c.setFont(FONT_NAME, 10)
    c.drawString(col_x[0] + 6, y - row_height / 2, label1)

    suma_txt = f"{round(total_netto, 2):.2f}"
    c.setFont(FONT_NAME, 12)
    c.drawString(last_x + 6, y - row_height / 2, suma_txt)

    y -= row_height

    c.rect(col_x[0], y - row_height, big_width, row_height)
    c.rect(last_x, y - row_height, last_w, row_height)

    label2 = "CAŁKOWITA ILOŚĆ MIEJSC PALETOWYCH (mp)"
    c.setFont(FONT_NAME, 10)
    c.drawString(col_x[0] + 6, y - row_height / 2, label2)

    if float(total_palety).is_integer():
        suma_mp_txt = str(int(total_palety))
    else:
        suma_mp_txt = str(total_palety)

    c.setFont(FONT_NAME, 12)
    c.drawString(last_x + 6, y - row_height / 2, suma_mp_txt)

    draw_footer()
    c.showPage()
    c.save()

# =================== SKLEJANIE PDF ===================

def zapisz_pdf_z_stronami(input_pdf_path, output_pdf_path, strony_do_zachowania, summary_pdf_path):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    sorted_strony = sorted(set(strony_do_zachowania))
    for i in sorted_strony:
        if 0 <= i < len(reader.pages):
            writer.add_page(reader.pages[i])

    with open(summary_pdf_path, "rb") as f:
        summary_reader = PdfReader(f)
        for page in summary_reader.pages:
            writer.add_page(page)

    with open(output_pdf_path, "wb") as f:
        writer.write(f)

# =================== STREAMLIT APP ===================

def main():
    st.title("Zbiorczy list przewozowy – Kersia")

    st.write("1. Wgraj **oryginalny PDF**.")
    st.write("2. Wklej z Excela tabelę (4 kolumny: ZLECENIE, ilość palet, przewoźnik, mp).")
    st.write("   Możesz wkleić z nagłówkiem albo tylko same dane w tej kolejności.")
    st.write("3. Kliknij przycisk, żeby wygenerować Excel i PDF z podsumowaniem.")

    uploaded_pdf = st.file_uploader("Wgraj plik PDF", type=["pdf"])

    st.subheader("Wklejona tabela z Excela – będzie użyta do podziału na grupy")
    tabela_text = st.text_area(
        "Wklej tutaj dane skopiowane z Excela (C–F):",
        height=200,
    )

    if st.button("Generuj Excel + PDF (ze zbiorczą stroną)"):
        if uploaded_pdf is None:
            st.error("Najpierw wgraj plik PDF.")
            return

        try:
            groups, df_preview = parse_groups_from_pasted(tabela_text)
        except Exception as e:
            st.error(f"Błąd podczas interpretacji wklejonej tabeli: {e}")
            return

        st.subheader("Wklejona tabela (podgląd)")
        st.dataframe(df_preview, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_pdf.getbuffer())
            tmp_pdf_path = tmp_pdf.name

        try:
            wszystkie_zamowienia = sorted({nr for g in groups for nr in g["orders"]})

            wyniki, strony_do_zachowania = przetworz_zamowienia(tmp_pdf_path, wszystkie_zamowienia)

            df_summary, total_netto, total_palety = zbuduj_podsumowanie_grup(wyniki, groups)

            # data używana i w nagłówku, i w nazwach plików
            date_str = datetime.now().strftime("%Y-%m-%d")

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer) as writer:
                df_summary.to_excel(writer, index=False, sheet_name="Podsumowanie")
            excel_buffer.seek(0)

            excel_filename = f"podsumowanie_zamowien_{date_str}.xlsx"
            pdf_filename = f"zbiorczy_list_przewozowy_{date_str}.pdf"

            st.subheader("Pobierz plik Excel")
            st.download_button(
                label="Pobierz podsumowanie (Excel)",
                data=excel_buffer,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_summary_pdf:
                summary_pdf_path = tmp_summary_pdf.name

            utworz_zbiorczy_pdf(summary_pdf_path, df_summary, total_netto, total_palety, date_str)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_output_pdf:
                output_pdf_path = tmp_output_pdf.name

            zapisz_pdf_z_stronami(tmp_pdf_path, output_pdf_path, strony_do_zachowania, summary_pdf_path)

            with open(output_pdf_path, "rb") as f:
                out_pdf_bytes = f.read()

            st.subheader("Pobierz PDF ze zbiorczą stroną")
            st.download_button(
                label="Pobierz zbiorczy list przewozowy (PDF)",
                data=out_pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
            )

            st.subheader("Podsumowanie (tylko zlecenia znalezione w PDF)")
            st.dataframe(df_summary, use_container_width=True)

        finally:
            try:
                os.remove(tmp_pdf_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()
