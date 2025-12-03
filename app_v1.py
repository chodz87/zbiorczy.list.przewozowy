def main():
    st.title("Zbiorczy list przewozowy – wklejanie z Excela (tabela pod spodem)")

    uploaded_pdf = st.file_uploader("Wybierz plik PDF z zamówieniami:", type=["pdf"])

    st.markdown(
        "### 1. W Excelu zaznacz 4 kolumny (C–F)\n"
        "- **ZLECENIE**\n"
        "- **ilość palet** (opis – będzie w kolumnie 'Ilość palet')\n"
        "- **przewoźnik**\n"
        "- **mp** – liczba palet (będzie sumowana)\n\n"
        "Skopiuj (**Ctrl+C**) i wklej (**Ctrl+V**) poniżej."
    )

    tabela_text = st.text_area(
        "Wklej tutaj dane skopiowane z Excela (C–F, mogą być z nagłówkiem):",
        height=200,
        placeholder="ZLECENIE\tilość palet\tprzewoźnik\tmp\n53493+53498\tkarton\tdpd\t1\n56997+56998\t1p\tdsv\t1",
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

        # ładna tabela jak w Excelu – osobne wiersze i kolumny
        st.subheader("Wklejona tabela (rozbita jak w Excelu)")
        st.dataframe(df_preview, use_container_width=True)

        all_orders = sorted({o for g in groups for o in g["orders"]})

        with st.spinner("Przetwarzanie..."):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(uploaded_pdf.getbuffer())
                    tmp_pdf_path = tmp_pdf.name

                wyniki, strony = przetworz_zamowienia(tmp_pdf_path, all_orders)

                if len(strony) == 0:
                    st.error("Nie znaleziono żadnych stron z podanymi zleceniami w PDF.")
                    return

                df_summary, total_netto, total_palety = zbuduj_podsumowanie_grup(wyniki, groups)

                with tempfile.TemporaryDirectory() as tmpdir:
                    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    excel_path = os.path.join(tmpdir, f"list_przewozowy_v1_{today}.xlsx")
                    pdf_path = os.path.join(tmpdir, f"list_przewozowy_v1_{today}.pdf")
                    summary_temp = os.path.join(tmpdir, f"list_przewozowy_v1_{today}_zbiorczy_tmp.pdf")

                    df_summary.to_excel(excel_path, index=False)

                    df_for_pdf = df_summary[df_summary["ZLECENIE"] != "RAZEM"].copy()
                    utworz_zbiorczy_pdf(summary_temp, df_for_pdf, total_netto, total_palety)

                    zapisz_pdf_z_stronami(tmp_pdf_path, pdf_path, strony, summary_temp)

                    with open(excel_path, "rb") as f:
                        excel_bytes = f.read()
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()

                st.success("Gotowe! Poniżej możesz pobrać pliki.")
                st.write(f"**Suma wag netto:** {round(total_netto, 2)} kg")
                st.write(f"**Suma palet (mp, tylko dla zleceń znalezionych w PDF):** {total_palety}")

                st.download_button(
                    label="Pobierz Excel (z paletami)",
                    data=excel_bytes,
                    file_name=f"list_przewozowy_v1_{today}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.download_button(
                    label="Pobierz PDF (oryginalne strony + zbiorczy list)",
                    data=pdf_bytes,
                    file_name=f"list_przewozowy_v1_{today}.pdf",
                    mime="application/pdf",
                )

                st.subheader("Podsumowanie (tylko zlecenia znalezione w PDF)")
                st.dataframe(df_summary, use_container_width=True)

            finally:
                try:
                    os.remove(tmp_pdf_path)
                except Exception:
                    pass
