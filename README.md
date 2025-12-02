
# Wagi zamówień / Zbiorczy list przewozowy – wersja Python + Streamlit

To repo zawiera:

- logikę do:
  - odczytu danych z PDF (numery zamówień, wagi netto, adresy dostawy, numery przesyłek),
  - generowania Excela z podsumowaniem,
  - tworzenia zbiorczego PDF z listem przewozowym (landscape, logo Kersia, suma wag netto),
- dwie wersje interfejsu:
  - **desktop (tkinter)**,
  - **web (Streamlit)**.

## Struktura

- `wagi_zamowien_gui.py` – logika + wersja desktop (okienko tkinter).
- `streamlit_app.py` – wersja web (Streamlit) korzystająca z tej samej logiki.
- `DejaVuSans.ttf` – czcionka z polskimi znakami używana w PDF (podmień na prawdziwy plik u siebie).
- `logo_kersia.png` – logo firmy na zbiorczym liście (podmień na prawdziwe logo).
- `requirements.txt` – zależności.
- `.gitignore` – ignorowane pliki i katalogi (venv, cache itp.).

## Uruchomienie lokalne – wersja desktop (tkinter)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python wagi_zamowien_gui.py
```

## Uruchomienie lokalne – wersja web (Streamlit)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Po uruchomieniu Streamlit otwórz wskazany adres (np. http://localhost:8501).

## Deploy na Streamlit Cloud

1. Wgraj to repo na GitHub.
2. Wejdź na Streamlit Community Cloud (https://share.streamlit.io/).
3. Utwórz nową aplikację, wskaż:
   - repozytorium z GitHuba,
   - plik startowy: `streamlit_app.py`.
4. Zatwierdź – po chwili aplikacja będzie dostępna jako webowa.

Uwaga: w tym ZIP-ie font i logo są placeholderami (puste pliki), podmień je u siebie na prawdziwe `DejaVuSans.ttf` i `logo_kersia.png`.
