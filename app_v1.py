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

# --------------------------------------------------------------------
# FONT + LOGO
# --------------------------------------------------------------------


def init_font():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    ttf_path = os.path.join(base_dir, "DejaVuSans.ttf")
    if os.path.exists(ttf_path):
        pdfmetrics.registerFont(TTFont("DejaVuSans", ttf_path))
        return "DejaVuSans"
    else:
        return "Helvetica"


FONT_NAME = init_font()


def load_logo():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    logo_path = os.path.join(base_dir, "logo_kersia.png")
    if os.path.exists(logo_path):
        return logo_path
    return None


LOGO_PATH = load_logo()

# --------------------------------------------------------------------
# POMOCNICZE
# --------------------------------------------------------------------


def wyczysc_liczbe(s):
    """
    Zamienia string typu "1 234,56" na float 1234.56
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip()
    if not txt or txt.lower() == "nan":
        return None
    txt = txt.replace(" ", "").replace("\xa0", "")
    txt = txt.replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


# --------------------------------------------------------------------
# WAGA NETTO
# --------------------------------------------------------------------


def znajdz_wage_netto_podsumowanie(linie):
    """
    Szuka wagi netto w sekcji podsumowania:
    - najpierw szuka wiersza nagłówka z "Ilość palet" i "Całkowita waga netto"
    - potem bierze z następnych kilku wierszy liczby i wybiera drugą liczbę (kolumna netto)
    - jeśli się nie uda, szuka pojedynczo po frazie "Całkowita waga netto"
    Zwraca float albo None.
    """
    if not linie:
        return None

    # 1) klasyczny układ tabeli: nagłówek + wiersz z wartościami poniżej
    for i, linia in enumerate(linie):
        if "Ilość palet" in linia and "Całkowita waga netto" in linia:
            # przechodzimy po kilku kolejnych liniach i szukamy liczb
            for j in range(i + 1, min(i + 6, len(linie))):
                row = linie[j].strip()
                if not row:
                    continue
                nums = re.findall(r"(\d[\d\s.,]*)", row)
                values = []
                for n in nums:
                    v = wyczysc_liczbe(n)
                    if v is not None:
                        values.append(v)
                # spodziewamy się min. dwóch liczb: [ilość palet, waga netto, ...]
                if len(values) >= 2:
                    return values[1]

    # 2) fallback – szukamy samej frazy "Całkowita waga netto"
    for i, linia in enumerate(linie):
        if "Całkowita waga netto" in linia or "Total Net Weight" in linia:
            nums = re.findall(r"(\d[\d\s.,]*)", linia)
            values = [wyczysc_liczbe(n) for n in nums]
            values = [v for v in values if v is not None]
            if values:
                return values[-1]
            # może liczba jest w następnym wierszu
            if i + 1 < len(linie):
                row = linie[i + 1]
                nums = re.findall(r"(\d[\d\s.,]*)", row)
                values = [wyczysc_liczbe(n) for n in nums]
                values = [v for v in values if v is not None]
                if values:
                    return values[-1]

    return None


# --------------------------------------------------------------------
# NUMER PRZESYŁKI
# --------------------------------------------------------------------


def znajdz_numer_przesylki(linie):
    if not linie:
        return None

    # klasyczny przypadek "Numer przesyłki"
    for i, linia in enumerate(linie):
        low = linia.lower()
        if ("numer przesy" in low) or ("nr przesy" in low):
            for j in range(i, min(i + 3, len(linie))):
                nums = re.findall(r"\d{6,}", linie[j])
                if nums:
                    return nums[-1]

    # fallback po "QUALITY CERTIFICATE"
    for i, linia in enumerate(linie):
        if "QUALITY CERTIFICATE" in linia.upper():
            for j in range(i + 1, min(i + 5, len(linie))):
                nums = re.findall(r"\d{6,}", linie[j])
                if nums:
                    return nums[-1]

    return None


# --------------------------------------------------------------------
# ADRES DOSTAWY (z PDF – ten jest już OK)
# --------------------------------------------------------------------


def znajdz_adres_dostawy(linie):
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

    # wyciągamy blok między "Adresat/Consignee" a "Adres nabywcy"
    blok = [l.strip() for l in linie[start:end] if l.strip()]
    if not blok:
        return None

    # czyścimy techniczne linie (nasz adres nadawcy itp.)
    cleaned = []
    for l in blok:
        if "Nadawca/Consignor" in l:
            continue
        words = l.split()
        # wyrzucamy NIEPRUSZEWO (żeby nie brać adresu zakładu)
        words = [w for w in words if w.upper() != "NIEPRUSZEWO"]
        if not words:
            continue
        cleaned.append(" ".join(words))

    if not cleaned:
        return None

    # dzielimy na segmenty po "POLAND" – każdy segment to jeden adres
    joined = " ".join(cleaned)
    segments_raw = re.split(r"(POLAND)", joined)
    segments = []
    current = []
    for part in segments_raw:
        if not part:
            continue
        current.append(part)
        if "POLAND" in part:
            segments.append(" ".join(current).strip())
            current = []
    if current:
        segments.append(" ".join(current).strip())

    if not segments:
        return None

    # jeśli pierwsza wygląda jak nasz zakład – bierzemy kolejną
    selected = None
    non_plant_segments = []
    for seg in segments:
        if "NIEPRUSZEWO" in seg.upper():
            continue
        non_plant_segments.append(seg.split())
    if non_plant_segments:
        selected = non_plant_segments[0]
    else:
        # awaryjnie: bierzemy drugi segment jeśli jest,
        # inaczej ostatni
        if len(segments) >= 2:
            selected = segments[1].split()
        else:
            selected = segments[-1].split()

    selected = [s for s in selected if s.strip()]
    if not selected:
        return None

    # NOWY FORMAT: wszystko w jednej linii ze spacjami
    # (bez przecinków, bez złamań linii)
    return " ".join(selected)


# --------------------------------------------------------------------
# PRZETWARZANIE ZAMÓWIEŃ
# --------------------------------------------------------------------


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
                # szukamy numeru przesyłki na wszystkich stronach, gdzie występuje zlecenie
                for page_idx in strony:
                    numer_p = znajdz_numer_przesylki(pages_lines[page_idx])
                    if numer_p:
                        break
                # awaryjnie sprawdzamy jeszcze stronę po ostatniej
                if not numer_p:
                    last_idx = max(strony)
                    if last_idx + 1 < len(pages_lines):
                        numer_p = znajdz_numer_przesylki(
                            pages_lines[last_idx + 1]
                        )
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

    return wyniki, strony_do_zachowania


# --------------------------------------------------------------------
# PARSOWANIE WKLEJONEGO TEKSTU (EXCEL → TEXTAREA)
# --------------------------------------------------------------------


def _normalize_colname(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().lower()
    repl = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ż": "z",
        "ź": "z",
    }
    for k, v in repl.items():
        name = name.replace(k, v)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def parse_groups_from_pasted(text: str):
    """
    Oczekuje wklejonego tekstu z Excela (4 kolumny C–F):
    ZLECENIE | ilość palet | przewoźnik | mp

    Zwraca:
      - groups: lista słowników (label, orders, mp, ilosc_pal_tekst, przewoznik)
      - df_preview: DataFrame z danymi (ładna tabela do pokazania)
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
    df = pd.read_csv(buffer, sep=sep, engine="python")

    norm_to_orig = {}
    for col in df.columns:
        norm = _normalize_colname(col)
        norm_to_orig[norm] = col

    if "zlecenie" not in norm_to_orig:
        raise ValueError("Nie znaleziono kolumny 'ZLECENIE' we wklejonych danych.")

    if "mp" not in norm_to_orig:
        raise ValueError("Nie znaleziono kolumny 'mp' we wklejonych danych.")

    col_zlec = norm_to_orig["zlecenie"]
    col_mp = norm_to_orig["mp"]
    col_ilosc = norm_to_orig.get("iloscpalet")
    col_przew = norm_to_orig.get("przewoznik")

    groups = []
    for _, row in df.iterrows():
        zlec_raw = row.get(col_zlec, "")
        zlec = str(zlec_raw).strip()
        # pomijamy ewentualny nagłówek skopiowany jako wiersz
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

    cols_for_preview = [c for c in [col_zlec, col_ilosc, col_przew, col_mp] if c is not None]
    df_preview = df[cols_for_preview].copy()

    return groups, df_preview


# --------------------------------------------------------------------
# PODSUMOWANIE GRUP
# --------------------------------------------------------------------


def zbuduj_podsumowanie_grup(wyniki, groups):
    """
    Tworzy DataFrame z podsumowaniem:
    - sumuje wagę netto dla zleceń połączonych
    - sumuje mp (ilość palet)
    - IGNORUJE grupy, dla których żadne zlecenie nie istnieje w PDF
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
        adres = None
        przesylki = []

        any_data_for_group = False

        for nr in orders:
            d = wyniki.get(nr, {})
            if any(d.get(k) is not None for k in ("netto", "adres_dostawy", "numer_przesylki")):
                any_data_for_group = True

            n = d.get("netto")
            if n is not None:
                any_netto = True
                val = wyczysc_liczbe(n)
                if val is not None:
                    netto_sum += val

            if (adres is None) and d.get("adres_dostawy"):
                adres = d["adres_dostawy"]

            if d.get("numer_przesylki"):
                przesylki.append(str(d["numer_przesylki"]))

        # jeśli w PDF w ogóle nie ma tych zleceń → pomijamy cały wiersz
        if not any_data_for_group:
            continue

        netto_value = netto_sum if any_netto else None
        if netto_value is not None:
            total_netto += netto_value

        if mp is not None:
            val_mp = wyczysc_liczbe(mp)
            # ignorujemy NaN / None
            if val_mp is not None and not (isinstance(val_mp, float) and math.isnan(val_mp)):
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


# --------------------------------------------------------------------
# ZBIORCZY PDF
# --------------------------------------------------------------------


def utworz_zbiorczy_pdf(summary_path, df_summary, total_netto, total_palety):
    c = canvas.Canvas(summary_path, pagesize=landscape(A4))
    width, height = landscape(A4)

    # ----------------- NAGŁÓWEK -----------------
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
            "ul. Podgórna 4",
            "64-320 Niepruszewo",
        ]
        y_sender = height - 65
        for line in sender_lines:
            c.drawString(40, y_sender, line)
            y_sender -= 12

        date_str = datetime.now().strftime("%Y-%m-%d")
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

    # ----------------- STOPKA -----------------
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

    # ----------------- POMOC: ŁAMANIE ADRESU -----------------
    def wrap_address(adres, max_chars=60):
        """
        Lepsze łamanie adresu:
        - w pierwszej kolejności dzieli po przecinkach (każda część w nowej linii),
        - jeśli część jest nadal za długa, dzieli po spacjach,
        - nie ucina słów w połowie.
        """
        if adres is None:
            return [""]

        text = str(adres).strip()
        if not text:
            return [""]

        # najpierw próbujemy dzielić po przecinkach
        parts = [p.strip() for p in text.split(",") if p.strip()]
        lines = []

        def split_long_part(part):
            local_lines = []
            words = part.split()
            current = ""
            for w in words:
                if not current:
                    current = w
                elif len(current) + 1 + len(w) <= max_chars:
                    current += " " + w
                else:
                    local_lines.append(current)
                    current = w
            if current:
                local_lines.append(current)
            return local_lines

        for part in parts:
            if len(part) <= max_chars:
                lines.append(part)
            else:
                lines.extend(split_long_part(part))

        if not lines:
            return [text]

        return lines or [""]

    # ----------------- KONFIGURACJA TABELI (5 kolumn) -----------------
    start_y = draw_header()
    base_row_height = 26

    # marginesy poziome 40 po lewej/prawej
    left_margin = 40
    right_margin = 40

    # szerokości kolumn:
    # 0: ZLECENIE
    # 1: Ilość palet (mp)
    # 2: Numery przesyłek
    # 3: Całkowita waga netto (kg)
    # 4: Adres
    col_widths = [
        0.10 * (width - left_margin - right_margin),
        0.10 * (width - left_margin - right_margin),
        0.20 * (width - left_margin - right_margin),
        0.20 * (width - left_margin - right_margin),
        0.40 * (width - left_margin - right_margin),
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
        line_height = 11
        num_cols = len(col_x)
        for i in range(num_cols):
            x = col_x[i]
            w = col_widths[i]
            c.rect(x, y - h, w, h)

            lines = texts_lines[i]
            if not isinstance(lines, (list, tuple)):
                lines = [str(lines)]

            total_lines_height = len(lines) * line_height
            start_y_text = y - (h + total_lines_height) / 2 + line_height

            for j, txt in enumerate(lines):
                txt = str(txt)
                if bold_flags[i]:
                    c.setFont(FONT_NAME, 10)
                else:
                    c.setFont(FONT_NAME, 9)
                c.drawString(x + 3, start_y_text - j * line_height, txt)

    y = start_y

    header_texts = split_header_texts()
    header_bold = [True] * len(headers)
    header_height = base_row_height
    draw_row_frame_and_text(y, header_height, header_texts, header_bold)
    y -= header_height

    # dane – bez wiersza "RAZEM"
    records = df_summary[df_summary["ZLECENIE"] != "RAZEM"].to_dict("records")

    def new_page(title_suffix="(cd.)"):
        c.showPage()
        table_start = draw_header(title_suffix)
        return table_start

    for rec in records:
        zlec = rec.get("ZLECENIE", "")
        mp_val = rec.get("Ilość palet (mp)", "")
        przes = rec.get("Numery przesyłek", "")
        netto = rec.get("Całkowita waga netto (kg)", "")
        adres = rec.get("Adres dostawy", "")

        # adres: lepsze łamanie
        adres_lines = wrap_address(adres, max_chars=60)

        lines_in_cell = max(1, len(adres_lines))
        row_height = max(base_row_height, lines_in_cell * 11 + 4)

        if y - row_height < 70:
            draw_footer()
            y = new_page("(cd.)")
            draw_row_frame_and_text(y, header_height, header_texts, header_bold)
            y -= header_height

        texts_lines = [
            [str(zlec)],                        # ZLECENIE
            ["" if mp_val is None else str(mp_val)],  # Ilość palet (mp)
            [str(przes)],                       # Numery przesyłek
            [f"{float(netto):.2f}"] if netto not in [None, ""] else [""],  # Waga
            adres_lines,                        # Adres
        ]
        bold_flags = [False] * len(headers)

        draw_row_frame_and_text(y, row_height, texts_lines, bold_flags)
        y -= row_height
        y -= 2

    # trochę odstępu
    y -= 10

    # sekcja sum (2 wiersze)
    row_height = base_row_height
    total_rows_height = row_height * 2

    if y - total_rows_height < 70:
        draw_footer()
        y = new_page("(cd.)")

    big_width = sum(col_widths[:-1])
    last_x = col_x[-1]
    last_w = col_widths[-1]

    # Wiersz: CAŁKOWITA WAGA NETTO
    c.rect(col_x[0], y - row_height, big_width, row_height)
    c.rect(last_x, y - row_height, last_w, row_height)

    label1 = "CAŁKOWITA WAGA NETTO (kg)"
    c.setFont(FONT_NAME, 10)
    c.drawString(col_x[0] + 6, y - row_height / 2, label1)

    suma_txt = f"{round(total_netto, 2):.2f}"
    bold_font_size = 12
    c.setFont(FONT_NAME, bold_font_size)
    c.drawString(last_x + 6, y - row_height / 2, suma_txt)

    y -= row_height

    # Wiersz: ILOŚĆ MIEJSC PALETOWYCH
    c.rect(col_x[0], y - row_height, big_width, row_height)
    c.rect(last_x, y - row_height, last_w, row_height)

    label2 = "ILOŚĆ MIEJSC PALETOWYCH (mp)"
    c.setFont(FONT_NAME, 10)
    c.drawString(col_x[0] + 6, y - row_height / 2, label2)

    if float(total_palety).is_integer():
        suma_mp_txt = str(int(total_palety))
    else:
        suma_mp_txt = str(total_palety)

    c.setFont(FONT_NAME, bold_font_size)
    c.drawString(last_x + 6, y - row_height / 2, suma_mp_txt)

    draw_footer()
    c.showPage()
    c.save()


# --------------------------------------------------------------------
# SKLEJANIE PDF
# --------------------------------------------------------------------


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


# --------------------------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------------------------


def main():
    st.title("Zbiorczy list przewozowy – Kersia")

    st.write("1. Wgraj **oryginalny PDF**.")
    st.write("2. Wklej z Excela tabelę (kolumny: ZLECENIE, ilość palet, przewoźnik, mp).")
    st.write("3. Kliknij przycisk, żeby wygenerować Excel i PDF z podsumowaniem.")

    uploaded_pdf = st.file_uploader("Wgraj plik PDF", type=["pdf"])

    st.subheader("Wklejona tabela z Excela – będzie użyta do podziału na grupy")
    st.markdown(
        "Oczekiwane kolumny (w dowolnej kolejności, ważne są nazwy):\n"
        "- **ZLECENIE**\n"
        "- **ilość palet** (opis – będzie w kolumnie 'Ilość palet')\n"
        "- **przewoźnik**\n"
        "- **mp** – liczba palet (będzie sumowana)\n\n"
        "Skopiuj (**Ctrl+C**) i wklej (**Ctrl+V**) poniżej do pola tekstowego."
    )

    tabela_text = st.text_area(
        "Wklej tutaj dane skopiowane z Excela (C–F, mogą być z nagłówkiem):",
        height=200,
        placeholder="ZLECENIE\tilość palet\tprzewoźnik\tmp\n59743+60175\t1p\tDHL\t2",
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

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer) as writer:  # bez engine="xlsxwriter"
                df_summary.to_excel(writer, index=False, sheet_name="Podsumowanie")
            excel_buffer.seek(0)

            st.subheader("Pobierz plik Excel")
            st.download_button(
                label="Pobierz podsumowanie (Excel)",
                data=excel_buffer,
                file_name="podsumowanie_zamowien.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_summary_pdf:
                summary_pdf_path = tmp_summary_pdf.name

            utworz_zbiorczy_pdf(summary_pdf_path, df_summary, total_netto, total_palety)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_output_pdf:
                output_pdf_path = tmp_output_pdf.name

            zapisz_pdf_z_stronami(tmp_pdf_path, output_pdf_path, strony_do_zachowania, summary_pdf_path)

            with open(output_pdf_path, "rb") as f:
                out_pdf_bytes = f.read()

            st.subheader("Pobierz PDF ze zbiorczą stroną")
            st.download_button(
                label="Pobierz zbiorczy list przewozowy (PDF)",
                data=out_pdf_bytes,
                file_name="zbiorczy_list_przewozowy.pdf",
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
