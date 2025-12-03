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


# --------- NARZĘDZIA OGÓLNE ---------


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


# --------- WYCIĄGANIE DANYCH Z PDF ---------


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
            match = re.search(r"(\d{4,})", linia)
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
        return ("64-320" in txt) or (" BUK" in txt) or ("KASZTANOWA" in txt)

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

    # Adres w jednej linii ze spacjami (jak w Excelu)
    adres = " ".join(selected)
    adres = re.sub(r"\s+", " ", adres).strip()
    return adres


# --------- GŁÓWNA LOGIKA: PDF ---------


def przetworz_zamowienia(pdf_path, numery_zamowien):
    wyniki = {
        nr: {"netto": None, "adres_dostawy": None, "numer_przesylki": None}
        for nr in numery_zamowien
    }

    with pdfplumber.open(pdf_path) as pdf:
        pages_lines = []
        for page in pdf.pages:
            tekst = page.extract_text() or ""
            linie = tekst.split("\n")
            pages_lines.append(linie)

        mapa_zamowien_strony = {nr: [] for nr in numery_zamowien}

        for idx, linie in enumerate(pages_lines):
            nr = znajdz_numer_zamowienia(linie)
            if nr and nr in mapa_zamowien_strony:
                mapa_zamowien_strony[nr].append(idx)

        for nr, strony in mapa_zamowien_strony.items():
            if not strony:
                continue

            numer_p = None
            for idx in strony:
                numer_p = znajdz_numer_przesylki(pages_lines[idx])
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
                            break

            if netto is not None:
                wyniki[nr]["netto"] = netto

            if strony:
                idx0 = strony[0]
                adres = znajdz_adres_dostawy(pages_lines[idx0])
                if adres:
                    wyniki[nr]["adres_dostawy"] = adres

    return wyniki, mapa_zamowien_strony


# --------- STALE ADRESY (dopasowane ręcznie) ---------


# Stałe, ręcznie zweryfikowane adresy dostawy dla konkretnych zleceń
SPECJALNE_ADRESY_DOSTAWY = {
    "59743": "EMI Chopina 22 62-025 Kostrzyn Wlkp POLAND",
    "60175": "BOFERM SP. Z O.O. Górki 13 08-210 Platerów POLAND",
    "60141": "OKRĘGOWA SPÓŁDZIELNIA MLECZARSKA W ŁOWICZU Przemysłowa 3 99-400 Łowicz POLAND",
    "60071": "ROBERT GREGOR Wiśniowa 12 88-230 Piotrków Kujawski POLAND",
    "60153": "TEFA SP. Z O. O. SP. K. Tomasz Szpila Kraśnik Dolny 40A 59-700 Kraśnik Dolny POLAND",
    "60084": "TYMBARK o/TYMBARK Tymbark 156 34-650 Tymbark POLAND",
    "59745": "EMI Chopina 22 62-025 Kostrzyn Wlkp POLAND",
    "60130": "GOSPODARSTWO ROLNE JAN KASZTELAN Rempin, Szlachecka 20 09-213 Gozdowo POLAND",
    "60103": "CHEMEA JOANNA DMOWSKA / Pet Cuisine Sp. z o.o. ul. Narodowych Sił Zbrojnych 2 09-400 Płock POLAND",
    "60060": "GR MAREK FRĄTCZAK Kuźnica Czarnkowska, Zamkowa 1 64-700 Czarnków POLAND",
}


# --------- PODSUMOWANIE GRUP ---------


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
            d = wyniki.get(nr, wyniki.get(int(nr), {}))
            if any(
                d.get(k) is not None
                for k in ("netto", "adres_dostawy", "numer_przesylki")
            ):
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

        # Ręcznie nadpisujemy adresy dla konkretnych jedno-zleceniowych grup
        if len(orders) == 1:
            nr_single = str(orders[0])
            if nr_single in SPECJALNE_ADRESY_DOSTAWY:
                adres = SPECJALNE_ADRESY_DOSTAWY[nr_single]

        if not any_data_for_group:
            continue

        netto_value = netto_sum if any_netto else None
        if netto_value is not None:
            total_netto += netto_value

        if mp is not None:
            val_mp = wyczysc_liczbe(mp)
            if val_mp is not None:
                total_palety += val_mp

        rows.append(
            {
                "ZLECENIE": label,
                "Ilość palet (mp)": mp,
                "Ilość palet": ilosc_txt,
                "Przewoźnik": przew,
                "Numery zamówień": "+".join(orders),
                "Numery przesyłek": "+".join(sorted(set(przesylki))) if przesylki else "",
                "Całkowita waga netto (kg)": round(netto_value, 2)
                if netto_value is not None
                else None,
                "Adres dostawy": adres,
            }
        )

    rows.append(
        {
            "ZLECENIE": "RAZEM",
            "Ilość palet (mp)": total_palety,
            "Ilość palet": "",
            "Przewoźnik": "",
            "Numery zamówień": "",
            "Numery przesyłek": "",
            "Całkowita waga netto (kg)": round(total_netto, 2),
            "Adres dostawy": "",
        }
    )

    df = pd.DataFrame(rows)
    return df, total_netto, total_palety


# --------- PARSOWANIE WKLEJONEJ TABELI ---------


def parse_groups_from_pasted(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    groups = []
    current = None

    for line in lines:
        parts = re.split(r"\t+| {2,}", line)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue

        if parts[0].isdigit():
            label = parts[0]
            orders = [p.strip() for p in parts[1].split("+")]
            mp = parts[2] if len(parts) > 2 else None
            ilosc_pal_tekst = parts[3] if len(parts) > 3 else ""
            przewoznik = parts[4] if len(parts) > 4 else ""

            current = {
                "label": label,
                "orders": orders,
                "mp": mp,
                "ilosc_pal_tekst": ilosc_pal_tekst,
                "przewoznik": przewoznik,
            }
            groups.append(current)
        else:
            continue

    df_preview = pd.DataFrame(groups)
    return groups, df_preview


# --------- GENEROWANIE PDF Z PODSUMOWANIEM ---------


def narysuj_podsumowanie_pdf(output_path, df_summary, logo_path=None):
    cfont = znajdz_czcionke()
    c = canvas.Canvas(output_path, pagesize=landscape(A4))
    width, height = landscape(A4)

    if logo_path and os.path.exists(logo_path):
        try:
            c.drawImage(logo_path, 40, height - 80, width=120, preserveAspectRatio=True)
        except Exception:
            pass

    c.setFont(cfont, 14)
    c.drawString(200, height - 50, "ZBIORCZY LIST PRZEWOZOWY - PODSUMOWANIE")

    c.setFont(cfont, 10)
    today_str = datetime.today().strftime("%Y-%m-%d")
    c.drawRightString(width - 40, height - 50, f"Data generacji: {today_str}")

    start_y = height - 100
    x_offsets = {
        "ZLECENIE": 40,
        "Ilość palet (mp)": 110,
        "Ilość palet": 210,
        "Przewoźnik": 310,
        "Numery zamówień": 430,
        "Numery przesyłek": 560,
        "Całkowita waga netto (kg)": 700,
        "Adres dostawy": 830,
    }

    c.setFont(cfont, 8)
    y = start_y

    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    c.rect(30, 40, width - 60, start_y - 10 - 40)

    headers = [
        "ZLECENIE",
        "Ilość palet (mp)",
        "Ilość palet",
        "Przewoźnik",
        "Numery zamówień",
        "Numery przesyłek",
        "Całkowita waga netto (kg)",
        "Adres dostawy",
    ]
    for h in headers:
        c.drawString(x_offsets[h], y, h)
    y -= 14
    c.line(30, y + 5, width - 30, y + 5)

    for _, row in df_summary.iterrows():
        if y < 60:
            c.showPage()
            if logo_path and os.path.exists(logo_path):
                try:
                    c.drawImage(
                        logo_path, 40, height - 80, width=120, preserveAspectRatio=True
                    )
                except Exception:
                    pass
            c.setFont(cfont, 10)
            c.drawString(200, height - 50, "ZBIORCZY LIST PRZEWOZOWY - PODSUMOWANIE")
            c.setFont(cfont, 8)
            y = start_y

            for h in headers:
                c.drawString(x_offsets[h], y, h)
            y -= 14
            c.line(30, y + 5, width - 30, y + 5)

        for h in headers:
            val = row.get(h, "")
            if pd.isna(val):
                val = ""
            text = str(val)

            if h == "Adres dostawy":
                max_width = width - x_offsets[h] - 40
                line = ""
                tx = x_offsets[h]
                ty = y
                for word in text.split():
                    test_line = (line + " " + word).strip()
                    if c.stringWidth(test_line, cfont, 7) > max_width and line:
                        c.setFont(cfont, 7)
                        c.drawString(tx, ty, line)
                        ty -= 10
                        line = word
                    else:
                        line = test_line
                if line:
                    c.setFont(cfont, 7)
                    c.drawString(tx, ty, line)
            else:
                c.setFont(cfont, 7)
                c.drawString(x_offsets[h], y, text)

        y -= 16

    c.showPage()
    c.save()


def zapisz_pdf_z_stronami(output_path, original_pdf_path, strony, summary_pdf_path):
    writer = PdfWriter()

    with open(original_pdf_path, "rb") as f:
        reader = PdfReader(f)
        for idx in sorted(set(strony)):
            if 0 <= idx < len(reader.pages):
                writer.add_page(reader.pages[idx])

    with open(summary_pdf_path, "rb") as fsum:
        summary_reader = PdfReader(fsum)
        for page in summary_reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as fout:
        writer.write(fout)


# --------- STREAMLIT UI ---------


def main():
    st.set_page_config(page_title="List przewozowy - Kersia", layout="wide")
    st.title("Generator listu przewozowego - Kersia")

    st.markdown(
        """
    1. Wgraj **PDF z dokumentami dostawy**.
    2. Wklej tabelę (jak z Excela) ze zleceniami i paletami.
    3. Kliknij **Generuj pliki** – dostaniesz:
       - Excel z wagami i paletami,
       - Zbiorczy PDF: strony z PDF + strona z podsumowaniem.
    """
    )

    uploaded_pdf = st.file_uploader(
        "Wgraj PDF z dokumentami dostawy (Kersia)", type=["pdf"]
    )

    tabela_text = st.text_area(
        "Wklej tabelę ze zleceniami (np. z Excela)",
        height=200,
        placeholder=(
            "Np.:\n"
            "1\t59743\t1\t1 paleta\tRABEN\n"
            "2\t60175\t2\t2 palety\tRABEN\n"
        ),
    )

    if st.button("Generuj pliki"):
        if uploaded_pdf is None:
            st.error("Najpierw wgraj plik PDF.")
            return

        if not tabela_text.strip():
            st.error("Wklej tabelę ze zleceniami.")
            return

        try:
            groups, df_preview = parse_groups_from_pasted(tabela_text)
        except Exception as e:
            st.error(f"Błąd podczas interpretacji wklejonej tabeli: {e}")
            return

        if not groups:
            st.error("Nie udało się odczytać żadnych zleceń z wklejonej tabeli.")
            return

        st.subheader("Wklejona tabela (rozbita jak w Excelu)")
        st.dataframe(df_preview, use_container_width=True)

        all_orders = sorted({o for g in groups for o in g["orders"]})

        with st.spinner("Przetwarzanie..."):
            tmp_pdf_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_pdf_path = tmp.name

                wyniki, mapa_stron = przetworz_zamowienia(tmp_pdf_path, all_orders)

                dane_wiersze = []
                strony_do_zapisania = set()

                for nr in all_orders:
                    w = wyniki.get(nr, {})
                    netto = w.get("netto")
                    adres = w.get("adres_dostawy")
                    nr_przes = w.get("numer_przesylki")

                    strony = mapa_stron.get(nr, [])
                    for s in strony:
                        strony_do_zapisania.add(s)

                    dane_wiersze.append(
                        {
                            "Numer zamówienia": nr,
                            "Waga netto (z podsumowania)": netto,
                            "Adres dostawy": adres,
                            "Numer przesyłki": nr_przes,
                        }
                    )

                df = pd.DataFrame(dane_wiersze)

                df_summary, total_netto, total_palety = zbuduj_podsumowanie_grup(
                    wyniki, groups
                )

                today = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    df.to_excel(
                        writer, sheet_name="Zamówienia", index=False
                    )
                    df_summary.to_excel(
                        writer, sheet_name="Podsumowanie", index=False
                    )
                excel_buffer.seek(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_sum:
                    summary_temp = tmp_sum.name

                logo_path = get_logo_path()
                narysuj_podsumowanie_pdf(summary_temp, df_summary, logo_path)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
                    pdf_path = tmp_out.name

                strony = sorted(list(strony_do_zapisania))
                zapisz_pdf_z_stronami(tmp_pdf_path, pdf_path, strony, summary_temp)

                with open(excel_buffer.getbuffer(), "rb") as f:
                    excel_bytes = f.read()
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.success("Gotowe! Poniżej możesz pobrać pliki.")
                st.write(f"**Suma wag netto:** {round(total_netto, 2)} kg")
                st.write(
                    f"**Suma palet (mp, tylko dla zleceń znalezionych w PDF):** {total_palety}"
                )

                st.download_button(
                    label="Pobierz Excel (z paletami)",
                    data=excel_bytes,
                    file_name=f"list_przewozowy_v1_{today}.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                    ),
                )

                st.download_button(
                    label="Pobierz PDF (strony + podsumowanie)",
                    data=pdf_bytes,
                    file_name=f"list_przewozowy_v1_{today}.pdf",
                    mime="application/pdf",
                )

                st.subheader("Podsumowanie (tylko zlecenia znalezione w PDF)")
                st.dataframe(df_summary, use_container_width=True)

            finally:
                if tmp_pdf_path is not None:
                    try:
                        os.remove(tmp_pdf_path)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
