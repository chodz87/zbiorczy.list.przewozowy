
# Wagi zamówień / list przewozowy

Aplikacja do:
- odczytu danych z PDF (numery zamówień, wagi netto, adresy dostawy, numery przesyłek),
- generowania Excela z podsumowaniem,
- tworzenia zbiorczego PDF z listem przewozowym (landscape, logo Kersia, suma wag netto).

## Uruchomienie (Python)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python wagi_zamowien_gui.py
```

## Budowa EXE (Windows)

```bash
build_exe.bat
```

W folderze `dist/` pojawi się `wagi_zamowien_gui.exe`.
