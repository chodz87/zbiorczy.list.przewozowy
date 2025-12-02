import os
import tempfile
from typing import List

import streamlit as st
import pandas as pd

from wagi_zamowien_gui import (
    PackingListExtractor,
    ExcelExporter,
    SummaryPdfBuilder,
    save_combined_pdf,
    FONT_NAME,
    LOGO_PATH,
)


def parse_order_numbers(text: str) -> List[str]:
    return [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]


def process_pdf_and_orders(pdf_bytes: bytes, order_numbers: List[str]):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_pdf_path = os.path.join(tmpdir, "wejscie.pdf")
        with open(input_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        extractor = PackingListExtractor(input_pdf_path, order_numbers)
        results_dict, pages_to_keep = extractor.extract()

        if not pages_to_keep:
            raise ValueError("Nie znaleziono żadnych stron z podanymi numerami zamówień.")

        excel_path = os.path.join(tmpdir, "list_przewozowy.xlsx")
        pdf_output_path = os.path.join(tmpdir, "list_przewozowy.pdf")
        summary_temp_path = os.path.join(tmpdir, "list_przewozowy_zbiorczy_tmp.pdf")

        ExcelExporter.save(results_dict, order_numbers, excel_path)

        summary_builder = SummaryPdfBuilder(FONT_NAME, LOGO_PATH)
        summary_builder.build(summary_temp_path, results_dict, order_numbers)

        save_combined_pdf(input_pdf_path, pdf_output_path, pages_to_keep, summary_temp_path)

        with open(excel_path, "rb") as f:
            excel_bytes = f.read()
        with open(pdf_output_path, "rb") as f:
            pdf_bytes = f.read()

        rows = []
        total_net = 0.0
        for nr in order_numbers:
            data = results_dict.get(nr)
            if not data:
                continue
            netto = data.net_weight
            if netto is not None:
                try:
                    total_net += float(netto)
                except (TypeError, ValueError):
                    pass
            rows.append({
                "Numer zamówienia": data.order_number,
                "Numer przesyłki": data.shipment_number,
                "Całkowita waga netto (kg)": data.net_weight,
                "Adres dostawy": data.delivery_address,
            })

        rows.append({
            "Numer zamówienia": "RAZEM",
            "Numer przesyłki": "",
            "Całkowita waga netto (kg)": round(total_net, 2),
            "Adres dostawy": "",
        })

        df = pd.DataFrame(rows)

    return df, excel_bytes, pdf_bytes


def main():
    st.set_page_config(page_title="Wagi zamówień – Kersia", layout="wide")

    st.title("📦 Wagi zamówień / Zbiorczy list przewozowy (wersja web – Streamlit)")

    st.markdown(
        '''
Aplikacja webowa na podstawie wersji desktopowej:

- Wgrywasz PDF z listami (`WshPackingList...`),
- wpisujesz numery zamówień,
- generuje się:
  - plik **Excel** z wagami netto, numerami przesyłek i adresami dostawy,
  - plik **PDF** z wybranymi stronami + **zbiorczy list przewozowy** na końcu.
        '''
    )

    st.header("1️⃣ Wgraj plik PDF")
    uploaded_pdf = st.file_uploader("Wybierz plik PDF", type=["pdf"])

    st.header("2️⃣ Podaj numery zamówień")
    orders_text = st.text_area(
        "Numery zamówień (oddzielone przecinkami lub nową linią):",
        placeholder="Np. 59743, 60175, 60060",
        height=100,
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("▶️ Przetwórz", type="primary")
    with col_info:
        st.info("Upewnij się, że numery zamówień są dokładnie takie, jak w PDF.")

    if run_btn:
        if uploaded_pdf is None:
            st.error("Najpierw wgraj plik PDF.")
            return

        order_numbers = parse_order_numbers(orders_text)
        if not order_numbers:
            st.error("Podaj co najmniej jeden numer zamówienia.")
            return

        try:
            with st.spinner("Przetwarzanie PDF, wyciąganie danych, generowanie plików..."):
                pdf_bytes = uploaded_pdf.read()
                df, excel_bytes, out_pdf_bytes = process_pdf_and_orders(pdf_bytes, order_numbers)

            st.success("Gotowe! Pliki zostały wygenerowane.")

            st.subheader("📊 Podgląd danych (to samo co w Excelu)")
            st.dataframe(df, use_container_width=True)

            st.subheader("⬇️ Pobierz wygenerowane pliki")

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="⬇️ Pobierz Excel (list przewozowy)",
                    data=excel_bytes,
                    file_name="list_przewozowy_streamlit.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            with col2:
                st.download_button(
                    label="⬇️ Pobierz PDF (strony + zbiorczy list)",
                    data=out_pdf_bytes,
                    file_name="list_przewozowy_streamlit.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Wystąpił błąd podczas przetwarzania: {e}")


if __name__ == "__main__":
    main()
