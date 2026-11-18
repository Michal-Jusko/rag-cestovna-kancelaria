"""
Stiahne doménové PDF (cestovný ruch / cestovná kancelária) do data/pdf/.
Každý dokument je vybraný tak, aby pokrýval niektorú časť ontológie (stĺpec `triedy`).

Spustenie:  python download_pdfs.py
Ak niektoré PDF zlyhá (napr. EUR-Lex niekedy blokuje skripty), otvor odkaz v prehliadači,
ulož PDF pod uvedeným názvom do data/pdf/ a spusti indexovanie.
"""
import requests

import config

PDFS = [
    {"file": "01_zakon_170_2018_o_zajazdoch.pdf",
     "url": "https://static.slov-lex.sk/pdf/SK/ZZ/2018/170/ZZ_2018_170_20220401.pdf",
     "popis": "Zákon č. 170/2018 Z. z. o zájazdoch (práva cestujúceho, zrušenie zájazdu, zodpovednosť CK)",
     "triedy": "Typ zájazdu, Zákazník, Doplnková služba, Ubytovanie, Doprava"},
    {"file": "02_zakon_170_2018_formular_informacii.pdf",
     "url": "https://static.slov-lex.sk/pdf/prilohy/SK/ZZ/2018/170/20190101_4884202-2.pdf",
     "popis": "Príloha zákona 170/2018 – formulár štandardných informácií k zmluve o zájazde",
     "triedy": "Typ zájazdu, Destinácia, Ubytovanie, Stravovanie"},
    {"file": "03_nariadenie_261_2004_letecka_doprava.pdf",
     "url": "https://eur-lex.europa.eu/legal-content/SK/TXT/PDF/?uri=CELEX:32004R0261",
     "popis": "Nariadenie (ES) 261/2004 – náhrady pri zrušení a meškaní letu, odmietnutí nástupu",
     "triedy": "Letecká doprava, Zákazník"},
    {"file": "04_sudny_dvor_prava_cestujucich_letecka.pdf",
     "url": "https://curia.europa.eu/jcms/upload/docs/application/pdf/2019-07/qd-04-17-650-sk-n.pdf",
     "popis": "Súdny dvor EÚ – brožúra o právach cestujúcich v leteckej doprave (judikatúra)",
     "triedy": "Letecká doprava"},
    {"file": "05_nariadenie_2021_782_zeleznicna_doprava.pdf",
     "url": "https://eur-lex.europa.eu/legal-content/SK/TXT/PDF/?uri=CELEX:32021R0782",
     "popis": "Nariadenie (EÚ) 2021/782 o právach cestujúcich v železničnej doprave",
     "triedy": "Vlaková doprava"},
    {"file": "06_nariadenie_181_2011_autobusova_doprava.pdf",
     "url": "https://eur-lex.europa.eu/legal-content/SK/TXT/PDF/?uri=CELEX:32011R0181",
     "popis": "Nariadenie (EÚ) 181/2011 o právach cestujúcich v autobusovej a autokarovej doprave",
     "triedy": "Autobusová doprava"},
    {"file": "07_nariadenie_1177_2010_lodna_doprava.pdf",
     "url": "https://eur-lex.europa.eu/legal-content/SK/TXT/PDF/?uri=CELEX:32010R1177",
     "popis": "Nariadenie (EÚ) 1177/2010 o právach cestujúcich pri cestovaní po mori a vnútrozemských vodných cestách",
     "triedy": "Lodná doprava, Okružná plavba"},
    {"file": "08_union_cestovne_poistenie_vpp.pdf",
     "url": "https://www.crz.gov.sk//data/att/5904184.pdf",
     "popis": "Union – Všeobecné poistné podmienky pre krátkodobé cestovné poistenie (VPPUKCP/2412)",
     "triedy": "Cestovné poistenie, Destinácia (územná platnosť), Zákazník"},
    {"file": "09_vop_ck_vida.pdf",
     "url": "https://www.crz.gov.sk//data/att/5929144.pdf",
     "popis": "Všeobecné obchodné podmienky Cestovnej kancelárie Vida (storno, reklamácie, ubytovanie)",
     "triedy": "Typ zájazdu, Ubytovanie, Cestovné poistenie, Zákazník"},
    {"file": "10_vzp_wachumba_ck.pdf",
     "url": "https://www.crz.gov.sk//data/att/4412715.pdf",
     "popis": "Všeobecné zmluvné podmienky o zájazde – WACHUMBA ck",
     "triedy": "Typ zájazdu, Zákazník, Doplnková služba"},
    {"file": "11_allianz_poistenie_upadok_ck.pdf",
     "url": "https://www.allianz.sk/content/dam/onemarketing/cee/azsk/podnikatelia/mal%C3%AD-a-stredn%C3%AD-podnikatelia/podnikatelia/VPP-Poistenie-pre-pr%C3%ADpad-%C3%BApadku-cestovnej-kancel%C3%A1rie-01102019.pdf",
     "popis": "Allianz – poistné podmienky pre prípad úpadku cestovnej kancelárie (repatriácia)",
     "triedy": "Cestovné poistenie, Doprava, Ubytovanie"},
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "application/pdf,*/*"}


def main():
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for d in PDFS:
        target = config.PDF_DIR / d["file"]
        if target.exists():
            print(f"[=] {d['file']} už existuje")
            continue
        try:
            r = requests.get(d["url"], headers=HEADERS, timeout=60)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("server nevrátil PDF (pravdepodobne ochrana proti robotom)")
            target.write_bytes(r.content)
            print(f"[+] {d['file']} ({len(r.content) // 1024} kB)")
        except Exception as e:  # noqa: BLE001
            failed.append(d)
            print(f"[!] {d['file']}: {e}")

    if failed:
        print("\nTieto PDF stiahni ručne v prehliadači a ulož do data/pdf/ pod uvedeným názvom:")
        for d in failed:
            print(f"  {d['file']}\n    {d['url']}")
    print(f"\nHotovo: {len(PDFS) - len(failed)}/{len(PDFS)} PDF. Ďalší krok: python ingest.py")


if __name__ == "__main__":
    main()
