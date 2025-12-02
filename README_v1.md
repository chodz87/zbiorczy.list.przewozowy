# Kersia – List przewozowy (Streamlit v1)

Aplikacja webowa do:
- wczytywania zbiorczego PDF z zamówieniami,
- wyciągania wag netto, numerów przesyłek i adresów dostawy dla wybranych numerów zamówień,
- generowania:
  - pliku Excel z podsumowaniem,
  - PDF zawierającego oryginalne strony + zbiorczy list przewozowy.

## Wymagania

Python 3.9+ oraz pakiety z `requirements_v1.txt`.

Instalacja:

```bash
pip install -r requirements_v1.txt
```

## Uruchomienie

```bash
streamlit run app_v1.py
```

Następnie:
1. Wgraj plik PDF z zamówieniami.
2. Wpisz numery zamówień (po przecinku lub każdy w osobnym wierszu).
3. Kliknij **"Generuj Excel + PDF (ze zbiorczą stroną)"**.
4. Pobierz wygenerowane pliki (Excel + PDF).

## Pliki w repozytorium

- `app_v1.py` – główna aplikacja Streamlit.
- `requirements_v1.txt` – lista zależności.
- `logo_kersia.png` – logo używane w PDF.
- `DejaVuSans.ttf` – czcionka z polskimi znakami do raportów PDF.
