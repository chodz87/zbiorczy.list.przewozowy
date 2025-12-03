###############################################
#  app_v1.py – kompletny działający plik
###############################################

import re
import io
import os
import tempfile
from datetime import datetime

import pdfplumber
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import streamlit as st

###########################################################
# NARZĘDZIA
###########################################################

def wyczysc_liczbe(tekst):
    if tekst is None:
        return None
    if isinstance(tekst, (int, float)):
        return float(tekst)
    t = str(tekst).strip()
    t = t.replace(" ", "").replace("\u00a0", "")
    t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def znajdz_czcionke():
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


###########################################################
# WYCIĄGANIE Z PDF
###########################################################

def znajdz_numer_zamowienia(linie):
    for linia in linie:
        if "Numer zamówienia" in linia or "Order Number" in linia:
            match = re.search(r"(\d{4,})", linia)
            if match:
                return match.group(1)
    return None


def znajdz_numer_przesylki(linie):
    for linia in linie:
        if "Numer przesyłki" in linia or "Shipment Number" in linia:
            match = re.search(r"(\d+)", linia)
            if match:
                return match.group(1)
    return None


def znajdz_wage_netto_podsumowanie(linie_strony):
    for linia in linie_strony:
        if (
            "Całkowita waga netto" in linia
            or "Total Net Weight" in linia
            or "Ca\u0142kowita waga netto" in linia
        ):
            match = re.search(r"([\d\s.,]+)", linia)
            if match:
                return match.group(1)
    return None


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

    blok = [l.strip() for l in linie[start:end] if l.strip()]
    if not blok:
        return None

    cleaned = []
    for l in blok:
        if "Nadawca/Consignor" in l:
            continue
        words = l.split()
        words = [w for w in words if w.upper() != "NIEPRUSZEWO"]
        if not words:
            continue
        cleaned.append(" ".join(words))

    if not cleaned:
        return None

    segments = []
    current = []
    for l in cleaned:
        current.append(l)
        if l.strip().upper() == "POLAND":
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    if not segments:
        return None

    def is_plant_segment(seg):
        txt = " ".join(seg).upper()
        return ("64-320" in txt) or ("BUK" in txt) or ("KASZTANOWA" in txt)

    non_plant_segments = [seg for seg in segments if not is_plant_segment(seg)]

    selected = None
    if non_plant_segments:
        selected = non_plant_segments[0]
    else:
        if len(segments) >= 2:
            selected = segments[1]
        else:
            selected = segments[-1]

    if not selected:
        return None

    selected = [s for s in selected if s.strip()]
    adres = " ".join(selected)
    adres = re.sub(r"\s+", " ", adres).strip()
    return adres


###########################################################
# ODCZYT CAŁEGO PDF POD KILKA ZLECEŃ
###########################################################

def przetworz_zamowienia(pdf_path, numery_zamowien):
    wyniki = {
        nr: {"netto": None, "adres_dostawy": None, "numer_przesylki": None}
        for nr in numery_zamowien
    }

    with pdfplumber.open(pdf_path) as pdf:
        pages_lines = []
        for page in pdf.pages:
            tekst = page.extract_text() or ""
            pages_lines.append(tekst.split("\n"))

        mapowanie = {nr: [] for nr in numery_zamowien}

        for idx, linie in enumerate(pages_lines):
            nr = znajdz_numer_zamowienia(linie)
            if nr and nr in mapowanie:
                mapowanie[nr].append(idx)

        for nr, strony in mapowanie.items():
            if not strony:
                continue

            # przesyłki
            for idx in strony:
                p = znajdz_numer_przesylki(pages_lines[idx])
                if p:
                    wyniki[nr]["numer_przesylki"] = p

            # waga
            netto = None
            for idx in strony:
                val = znajdz_wage_netto_podsumowanie(pages_lines[idx])
                if val:
                    netto = val

            if netto is None:
                for idx in strony:
                    if idx + 1 < len(pages_lines):
                        val = znajdz_wage_netto_podsumowanie(pages_lines[idx + 1])
                        if val:
                            netto = val
                            break

            if netto:
                wyniki[nr]["netto"] = netto

            # adres
            adres = znajdz_adres_dostawy(pages_lines[strony[0]])
            if adres:
                wyniki[nr]["adres_dostawy"] = adres

    return wyniki, mapowanie


###########################################################
# RĘCZNIE POPRAWIONE ADRESY
###########################################################

SPECJALNE_ADRESY_DOSTAWY = {
    "59743": "EMI Chopina 22 62-025 Kostrzyn Wlkp POLAND",
    "60175": "BOFERM SP. Z O.O. Górki 13 08-210 Platerów POLAND",
    "60141": "OKRĘGOWA SPÓŁDZIELNIA MLECZARSKA W ŁOWICZU Przemysłowa 3 99-400 Łowicz POLAND",
    "60071": "ROBERT GREGOR Wiśniowa 12 88-230 Piotrków Kujawski POLAND",
    "60153": "TEFA SP. Z O. O. SP. K. Tomasz Szpila Kraśnik Dolny 40A 59-700 Kraśnik Dolny POLAND",
    "60084": "TYMBARK o/TYMBARK Tymbark 156 34-650 Tymbark POLAND",
    "59745": "EMI Chopina 22 62-025 Kostrzyn Wlkp POLAND",
    "60130": "GOSPODARSTWO ROLNE JAN KASZTELAN Rempin Szlachecka 20 09-213 Gozdowo POLAND",
    "60103": "CHEMEA JOANNA DMOWSKA Pet Cuisine Sp. z o.o. Narodowych Sił Zbrojnych 2 09-400 Płock POLAND",
    "60060": "GR MAREK FRĄTCZAK Kuźnica Czarnkowska Zamkowa 1 64-700 Czarnków POLAND",
}


###########################################################
# PODSUMOWANIE GRUP
###########################################################

def zbuduj_podsumowanie_grup(wyniki, groups):
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
        auta = []
        przesylki = []
        adres = None
        any_data = False

        for nr in orders:
            key = str(nr).strip()
            d = wyniki.get(key, {})

            if any(d.get(k) for k in ("netto", "adres_dostawy", "numer_przesylki")):
                any_data = True

            if d.get("netto"):
                val = wyczysc_liczbe(d["netto"])
                if val is not None:
                    netto_sum += val

            if not adres and d.get("adres_dostawy"):
                adres = d["adres_dostawy"]

            if d.get("numer_przesylki"):
                przesylki.append(d["numer_przesylki"])

        if len(orders) == 1:
            nr_single = str(orders[0]).strip()
            if nr_single in SPECJALNE_ADRESY_DOSTAWY:
                adres = SPECJALNE_ADRESY_DOSTAWY[nr_single]

        if not any_data:
            continue

        netto_val = netto_sum if netto_sum > 0 else None
        if netto_val:
            total_netto += netto_val

        if mp:
            val_mp = wyczysc_liczbe(mp)
            if val_mp:
                total_palety += val_mp

        rows.append({
            "ZLECENIE": label,
            "Ilość palet (mp)": mp,
            "Ilość palet": ilosc_txt,
            "Przewoźnik": przew,
            "Numery zamówień": "+".join(orders),
            "Numery przesyłek": "+".join(przesylki),
            "Całkowita waga netto (kg)": netto_val,
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


###########################################################
# PARSOWANIE TABELI WKLEJONEJ Z EXCELA
###########################################################

def parse_groups_from_pasted(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    groups = []

    for line in lines:
        parts = re.split(r"\t+| {2,}", line)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue

        if parts[0].isdigit():
            label = parts[0]
            if len(parts) > 1:
                orders = [o.strip() for o in parts[1].replace(",", "+").split("+")]
            else:
                orders = []

            mp = parts[2] if len(parts) > 2 else ""
            ilosc = parts[3] if len(parts) > 3 else ""
            przew = parts[4] if len(parts) > 4 else ""

            groups.append({
                "label": label,
                "orders": orders,
                "mp": mp,
                "ilosc_pal_tekst": ilosc,
                "przewoznik": przew,
            })

    df_preview = pd.DataFrame(groups)
    return groups, df_preview


###########################################################
# PDF PODSUMOWANIE
###########################################################

def narysuj_podsumowanie_pdf(out_path, df, logo_path=None):
    font = znajdz_czcionke()
    c = canvas.Canvas(out_path, pagesize=landscape(A4))
    width, height = landscape(A4)

    if logo_path and os.path.exists(logo_path):
        c.drawImage(logo_path, 40, height - 80, width=120)

    c.setFont(font, 14)
    c.drawString(200, height - 50, "ZBIORCZY LIST PRZEWOZOWY - PODSUMOWANIE")
    c.setFont(font, 10)
    c.drawRightString(width - 40, height - 50,
                      "Data: " + datetime.now().strftime("%Y-%m-%d"))

    x = {
        "ZLECENIE": 40,
        "Ilość palet (mp)": 110,
        "Ilość palet": 200,
        "Przewoźnik": 300,
        "Numery zamówień": 430,
        "Numery przesyłek": 560,
        "Całkowita waga netto (kg)": 700,
        "Adres dostawy": 850,
    }

    y = height - 100
    c.setFont(font, 8)

    headers = list(x.keys())
    for h in headers:
        c.drawString(x[h], y, h)
    y -= 15

    for _, row in df.iterrows():
        if y < 50:
            c.showPage()
            y = height - 100
            for h in headers:
                c.drawString(x[h], y, h)
            y -= 15

        for h in headers:
            t = str(row[h]) if row[h] is not None else ""
            if h == "Adres dostawy":
                c.setFont(font, 6)
            else:
                c.setFont(font, 7)

            c.drawString(x[h], y, t)

        y -= 15

    c.showPage()
    c.save()


###########################################################
# SCALENIE STRON PDF + PODSUMOWANIE
###########################################################

def zapisz_pdf_z_stronami(pdf_input, pdf_output, strony, summary_pdf_path):
    writer = PdfWriter()

    with open(pdf_input, "rb") as f:
        r = PdfReader(f)
        for idx in sorted(strony):
            if idx < len(r.pages):
                writer.add_page(r.pages[idx])

    with open(summary_pdf_path, "rb") as fsum:
        sr = PdfReader(fsum)
        for p in sr.pages:
            writer.add_page(p)

    with open(pdf_output, "wb") as fout:
        writer.write(fout)


###########################################################
# STREAMLIT UI
###########################################################

def main():
    st.set_page_config(page_title="List przewozowy Kersia", layout="wide")
    st.title("Generator listu przewozowego – Kersia")

    uploaded_pdf = st.file_uploader("Wgraj PDF Kersia", type=["pdf"])
    tabela_text = st.text_area("Wklej tabelę ze zleceniami", height=200)

    if st.button("GENERUJ"):
        if not uploaded_pdf:
            st.error("Wgraj PDF!")
            return
        if not tabela_text.strip():
            st.error("Wklej tabelę!")
            return

        groups, df_prev = parse_groups_from_pasted(tabela_text)
        if not groups:
            st.error("Nie udało się odczytać wierszy tabeli.")
            return

        st.subheader("Zinterpretowana tabela")
        st.dataframe(df_prev, use_container_width=True)

        all_orders = sorted({o for g in groups for o in g["orders"]})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        wyniki, strony = przetworz_zamowienia(pdf_path, all_orders)

        df_rows = []
        strony_do_zapisu = set()

        for nr in all_orders:
            d = wyniki.get(nr, {})
            df_rows.append({
                "Numer zamówienia": nr,
                "Waga netto": d.get("netto"),
                "Adres dostawy": d.get("adres_dostawy"),
                "Numer przesyłki": d.get("numer_przesylki")
            })
            if nr in strony:
                for s in strony[nr]:
                    strony_do_zapisu.add(s)

        df_zam = pd.DataFrame(df_rows)

        df_summary, total_netto, total_palety = zbuduj_podsumowanie_grup(
            wyniki, groups
        )

        # Excel
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as w:
            df_zam.to_excel(w, sheet_name="Zamówienia", index=False)
            df_summary.to_excel(w, sheet_name="Podsumowanie", index=False)
        excel_buf.seek(0)

        # PDF podsumowanie
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_sum:
            pdf_sum_path = tmp_sum.name
        narysuj_podsumowanie_pdf(pdf_sum_path, df_summary, get_logo_path())

        # PDF ostateczny
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
            final_pdf_path = tmp_out.name

        zapisz_pdf_z_stronami(pdf_path, final_pdf_path, strony_do_zapisu, pdf_sum_path)

        st.success("GOTOWE!")

        st.download_button(
            "Pobierz Excel",
            excel_buf.getvalue(),
            "wersja.xlsx"
        )

        with open(final_pdf_path, "rb") as f:
            st.download_button(
                "Pobierz PDF",
                f.read(),
                "wersja.pdf"
            )


if __name__ == "__main__":
    main()
