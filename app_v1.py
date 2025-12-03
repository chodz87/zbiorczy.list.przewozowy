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
            # obsłuż przypadek gdy klucze w słowniku są int albo str
            d = wyniki.get(nr, wyniki.get(int(nr), {}))
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

        # *** TUTAJ NADPISUJEMY ADRESY DLA KONKRETNYCH ZLECEŃ ***
        # jeśli grupa ma dokładnie jedno zlecenie i mamy dla niego "ręczny" adres
        if len(orders) == 1:
            nr_single = str(orders[0])
            if nr_single in SPECJALNE_ADRESY_DOSTAWY:
                adres = SPECJALNE_ADRESY_DOSTAWY[nr_single]

        # jeśli w PDF w ogóle nie ma tych zleceń → pomijamy cały wiersz
        if not any_data_for_group:
            continue

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
