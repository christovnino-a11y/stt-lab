#!/usr/bin/env python
"""stt-lab — prepis reci Canary-1B-v2 (NVIDIA), volitelne s diarizaci
Nemotron 3 Diarization (NVIDIA). Oba modely bezi primo v tomhle procesu
pres transformers/torch na Metalu (MPS) — zadna externi binarka.

Dva rezimy:
- Nahravani: mikrofon v prohlizeci posila zvuk po pulsekundach, lab ho v
  pauzach kraji na useky (3-20 s) a prepisuje je na pozadi, jak prichazeji.
  Text naskakuje postupne se zpozdenim par sekund. Neni to „opravdovy"
  streaming (Canary je offline model), ale na klidne nahravani to staci.
- Soubor: nahrany zvuk nebo video se nakraji stejne a prepise cele.

Diarizace bezi az nad celou nahravkou, protoze potrebuje slyset vsechny
mluvci. Useky, ve kterych se mluvci stridaji, se na hranicich rozriznou a
prepisou znovu — jinak by rychla vymena replik dostala jednoho mluvciho.

Oba modely zustavaji nahrane v pameti — cely proces zabira ~4,3 GB
(namereno `footprint`, Canary v bf16), lab tedy neni lehky.
"""

import argparse
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr
import torch
from transformers import (
    AutoModelForAudioFrameClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

def _slozka_dat() -> Path:
    if cesta := os.environ.get("STT_LAB_DATA_DIR"):
        return Path(cesta).expanduser().resolve()
    if sys.platform == "win32":
        lokalni = os.environ.get("LOCALAPPDATA")
        return (Path(lokalni) if lokalni else Path.home() / "AppData" / "Local") / "STT Lab"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "STT Lab"
    return Path.home() / ".local" / "share" / "stt-lab"


VYSTUPY = _slozka_dat() / "vystupy"
PORT = 7864
SR = 16000

CANARY_MODEL = "nvidia/canary-1b-v2"
DIAR_MODEL = "nvidia/Nemotron-3-Diarization"
CANARY_REVIZE = "d455706339a6b32e1aa40f82c713a482a0c938e2"
DIAR_REVIZE = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"

# MPS je na M-cipech nejrychlejsi; kdyby chybel, jede se na CPU.
ZARIZENI = "mps" if torch.backends.mps.is_available() else "cpu"
# Canary v bf16: 1,9 GB misto 3,9 GB, na testu 8/8 useku stejny text jako
# fp32 a jen o chlup pomalejsi (namereno 2026-09-25). Na CPU bf16 nema smysl.
CANARY_DTYPE = torch.bfloat16 if ZARIZENI == "mps" else torch.float32

JAZYKY = {
    "čeština": "cs",
    "angličtina": "en",
    "němčina": "de",
    "slovenština": "sk",
    "polština": "pl",
    "ukrajinština": "uk",
    "ruština": "ru",
    "francouzština": "fr",
    "italština": "it",
    "španělština": "es",
}

FFMPEG_ARGS = ["-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le"]


def _inferencni(model):
    """Prepne model do rezimu inference (vypne dropout). Totez co .eval()."""
    model.train(False)
    return model


# ---------------------------------------------------------------------------
# Canary — prepis
# ---------------------------------------------------------------------------
# Namereno 2026-09-25 na M4 Pro (MPS, fp32, po zahrati): 5 s zvuku 0,3 s,
# 20 s zvuku 0,7 s, 40 s zvuku 1 s — 20-40x rychleji nez realtime.
# PAST: nad ~30 s Canary TISE VYNECHAVA cele vety (40s usek: chybely dve
# repliky, 30s byl kompletni). Proto se nic delsiho nez MAX_USEK_S neposila.
# Prvni prepis po startu je pomaly (zahrivani Metalu, desitky sekund), proto
# se model nahrava a zahriva uz pri startu na pozadi.

_CANARY = None
_CANARY_NACITANI = threading.Lock()
_CANARY_BEH = threading.Lock()  # generate() na jednom MPS modelu jen z jednoho vlakna


def _nacti_canary():
    global _CANARY
    with _CANARY_NACITANI:
        if _CANARY is None:
            processor = AutoProcessor.from_pretrained(CANARY_MODEL, revision=CANARY_REVIZE)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                CANARY_MODEL, revision=CANARY_REVIZE, dtype=CANARY_DTYPE
            ).to(ZARIZENI)
            _CANARY = (processor, _inferencni(model))
    return _CANARY


def _prepis_kus(zvuk: np.ndarray, jazyk: str, cil: str) -> str:
    """Prepise jeden usek (16 kHz mono float32, nejvys MAX_USEK_S)."""
    processor, model = _nacti_canary()
    with _CANARY_BEH:
        inputs = processor.apply_transcription_request(
            audio=zvuk, source_language=jazyk, target_language=cil,
        ).to(ZARIZENI, dtype=model.dtype)
        with torch.inference_mode():
            vystup = model.generate(**inputs, max_new_tokens=256)
        return processor.decode(vystup, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Krajeni na useky — spolecne pro nahravani i soubory
# ---------------------------------------------------------------------------
# Energeticky detektor reci po 30ms ramcich. Usek se uzavira v pauze, aby se
# neusekla slova, a drzi se mezi MIN a MAX: kratky usek ztraci kontext (horsi
# presnost), dlouhy narazi na past s vynechavanim vet (viz vyse).
RAMEC = 480                   # 30 ms pri 16 kHz
RAMCU_S = SR / RAMEC          # 33,3 ramce za sekundu
PAUZA_S = 0.5                 # takova pauza uz je predel mezi vetami
MIN_USEK_S = 3.0
CIL_USEK_S = 12.0             # odtud staci i kratsi nadech
MAX_USEK_S = 20.0
MIN_RECI_S = 0.3              # mene reci = ticho, neprepisovat (Canary by si vymyslel)


def _db(zvuk: np.ndarray) -> np.ndarray:
    n = len(zvuk) // RAMEC
    if n == 0:
        return np.zeros(0, np.float32)
    ramce = zvuk[: n * RAMEC].reshape(n, RAMEC)
    return 20 * np.log10(np.sqrt(np.mean(ramce * ramce, axis=1)) + 1e-6)


def _prah(hluk: float) -> float:
    """Hranice reci: 10 dB nad hlukem pozadi, ale v rozumnych mezich —
    kdyz je v okne skoro sama rec, percentil by vysel az na urovni reci."""
    return min(max(hluk + 10.0, -52.0), -35.0)


def _ma_rec(zvuk: np.ndarray, hluk: float) -> bool:
    return (_db(zvuk) > _prah(hluk)).sum() >= MIN_RECI_S * RAMCU_S


def _pauzy(rec: np.ndarray) -> list[tuple[int, int]]:
    """Behy ne-reci jako (zacatek, konec) v ramcich."""
    behy, i, n = [], 0, len(rec)
    while i < n:
        if rec[i]:
            i += 1
            continue
        j = i
        while j < n and not rec[j]:
            j += 1
        behy.append((i, j))
        i = j
    return behy


def _dalsi_rez(zvuk: np.ndarray, hluk: float, posledni: bool) -> int | None:
    """Kde uzavrit prvni usek v `zvuk` (index vzorku od jeho zacatku).

    None = pockat na dalsi zvuk. `posledni` = zadny dalsi zvuk neprijde.
    """
    db = _db(zvuk)
    rec = db > _prah(hluk)
    n = len(db)
    if rec.sum() < MIN_RECI_S * RAMCU_S:
        # V okne je jen ticho. Zahodit ho, jen konec si nechat — muze v nem
        # zacinat slovo, ktere pokracuje v dalsim kousku.
        if posledni:
            return len(zvuk)
        return len(zvuk) - SR // 2 if len(zvuk) >= 2 * SR else None

    behy = _pauzy(rec)
    reci_pred = np.concatenate([[0], np.cumsum(rec)])   # ramcu reci pred ramcem i

    def stred(a: int, b: int) -> int:
        return ((a + b) // 2) * RAMEC

    # 1) prvni poradna pauza po MIN_USEK_S, pred kterou uz zaznela rec
    for a, b in behy:
        if (b - a >= PAUZA_S * RAMCU_S and (a + b) // 2 >= MIN_USEK_S * RAMCU_S
                and reci_pred[a] >= MIN_RECI_S * RAMCU_S):
            return stred(a, b)
    if posledni and len(zvuk) <= MAX_USEK_S * SR:
        return len(zvuk)
    # 2) po CIL_USEK_S staci i kratsi nadech
    for a, b in behy:
        if (b - a >= 0.25 * RAMCU_S and (a + b) // 2 >= CIL_USEK_S * RAMCU_S
                and reci_pred[a] >= MIN_RECI_S * RAMCU_S):
            return stred(a, b)
    # 3) strop — rizni v nejtissim miste poslednich 4 s pred nim
    strop = int(MAX_USEK_S * RAMCU_S)
    if n >= strop:
        od = strop - int(4 * RAMCU_S)
        return (od + int(np.argmin(db[od:strop]))) * RAMEC
    return len(zvuk) if posledni else None


def _useky_souboru(zvuk: np.ndarray) -> list[tuple[int, int]]:
    """Nakraji celou nahravku na useky (zacatek, konec) ve vzorcich."""
    hluk = float(np.percentile(_db(zvuk), 10)) if len(zvuk) >= RAMEC else -60.0
    hranice, pos = [], 0
    while pos < len(zvuk):
        # Pravidla se divaji nejvys MAX_USEK_S dopredu, vic nema smysl pocitat.
        okno = zvuk[pos: pos + int((MAX_USEK_S + 1) * SR)]
        rez = _dalsi_rez(okno, hluk, pos + len(okno) >= len(zvuk)) or len(okno)
        if _ma_rec(zvuk[pos: pos + rez], hluk):
            hranice.append((pos, pos + rez))
        pos += rez
    return hranice


# ---------------------------------------------------------------------------
# Nemotron 3 Diarization — kdo kdy mluvi
# ---------------------------------------------------------------------------
_DIAR = None
_DIAR_NACITANI = threading.Lock()
_DIAR_BEH = threading.Lock()

# Plna pozornost pres celou nahravku je nejpresnejsi, ale model ma omezenou
# pozicni tabulku (NVIDIA uvadi ~6,6 min). Delsi nahravky jdou proudove po
# 1s kouscich s pameti mluvcich — DER podle NVIDIE jen o chlup horsi.
DIAR_OFFLINE_MAX_S = 300


def _nacti_diarizaci():
    global _DIAR
    with _DIAR_NACITANI:
        if _DIAR is None:
            processor = AutoProcessor.from_pretrained(DIAR_MODEL, revision=DIAR_REVIZE)
            model = AutoModelForAudioFrameClassification.from_pretrained(
                DIAR_MODEL, revision=DIAR_REVIZE, dtype=torch.float32
            ).to(ZARIZENI)
            _DIAR = (processor, _inferencni(model))
    return _DIAR


def _diarizuj_proudove(processor, model, zvuk: np.ndarray) -> torch.Tensor:
    """Postup podle karty modelu (sekce Streaming diarization)."""
    processor.set_streaming_mode("low_latency")

    def vstupy():
        yield processor(zvuk[: processor.num_samples_first_audio_chunk], sampling_rate=SR,
                        is_streaming=True, is_first_audio_chunk=True)
        mel = processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel)
        while (konec := start + processor.num_samples_per_audio_chunk) <= zvuk.shape[0]:
            yield processor(zvuk[start:konec], sampling_rate=SR,
                            is_streaming=True, is_first_audio_chunk=False)
            mel += processor.num_mel_frames_per_step
            start = processor.audio_chunk_start(mel)
        yield processor(zvuk[start:], sampling_rate=SR, is_streaming=True,
                        is_first_audio_chunk=False, is_last_audio_chunk=True)

    cache, logits = None, []
    for inputs in vstupy():
        inputs = inputs.to(ZARIZENI, dtype=model.dtype)
        vystup = model(**inputs, speaker_cache=cache)
        logits.append(vystup.logits)
        cache = vystup.speaker_cache
    return torch.cat(logits, dim=1)


def _diarizuj(zvuk: np.ndarray) -> list[tuple[float, float, int]]:
    """Useky (zacatek, konec, mluvci). Mluvci jsou cislovani podle toho, kdo
    promluvil PRVNI (Sortformer), ne podle podobnosti hlasu jako pyannote."""
    processor, model = _nacti_diarizaci()
    with _DIAR_BEH, torch.inference_mode():
        if len(zvuk) <= DIAR_OFFLINE_MAX_S * SR:
            inputs = processor(zvuk, sampling_rate=SR).to(ZARIZENI, dtype=model.dtype)
            logits = model(**inputs).logits
            segmenty = processor.extract_speaker_dict(logits, inputs.get("attention_mask"))[0]
        else:
            segmenty = processor.extract_speaker_dict(_diarizuj_proudove(processor, model, zvuk))[0]
    return [(s["Start"], s["End"], s["Speaker"]) for s in segmenty]


def _slouc(casti: list[list]) -> list[list]:
    """Spoji po sobe jdouci kusy stejneho mluvciho."""
    vysledek = []
    for a, b, s in casti:
        if vysledek and vysledek[-1][2] == s:
            vysledek[-1][1] = b
        else:
            vysledek.append([a, b, s])
    return vysledek


def _nejblizsi_mluvci(od: float, do: float, mluvci) -> int:
    stred = (od + do) / 2
    return min(mluvci, key=lambda m: 0 if m[0] <= stred <= m[1]
               else min(abs(m[0] - stred), abs(m[1] - stred)))[2]


def _rozdel_podle_mluvcich(useky, mluvci, zvuk, jazyk, cil):
    """Ke kazdemu useku priradi mluvciho. Kdyz se v useku mluvci stridaji,
    rozrizne ho na hranicich replik a casti prepise znovu — prepis je po
    usecich, takze jinak by cely usek dostal jednoho mluvciho."""
    if not mluvci:
        return useky
    vysledek = []
    for od, do, _, text in useky:
        casti = _slouc(sorted(
            [max(od, z), min(do, k), s] for z, k, s in mluvci if min(do, k) - max(od, z) > 0.05
        ))
        # Kousky pod 0,2 s jsou spis sum diarizace nez replika.
        dlouhe = [c for c in casti if c[1] - c[0] >= 0.2]
        casti = _slouc(dlouhe) if dlouhe else casti
        if len({c[2] for c in casti}) <= 1:
            s = casti[0][2] if casti else _nejblizsi_mluvci(od, do, mluvci)
            vysledek.append((od, do, s, text))
            continue
        # Hranice mezi replikami klademe doprostred mezery mezi nimi.
        hranice = [od] + [(casti[i][1] + casti[i + 1][0]) / 2
                          for i in range(len(casti) - 1)] + [do]
        for i, (_, _, s) in enumerate(casti):
            a, b = hranice[i], hranice[i + 1]
            if b - a < 0.3:
                continue
            t = _prepis_kus(zvuk[int(a * SR): int(b * SR)], jazyk, cil)
            if t:
                vysledek.append((a, b, s, t))
    return vysledek


# ---------------------------------------------------------------------------
# Vystupy
# ---------------------------------------------------------------------------

def _mmss(sekundy: float) -> str:
    return f"{int(sekundy // 60):02d}:{int(sekundy % 60):02d}"


def _s_na_srt(sekundy: float) -> str:
    ms = int(round(max(sekundy, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _stitek(mluvci) -> str:
    return f"[Mluvčí {mluvci + 1}] " if mluvci is not None else ""


def _vysledny_text(useky, casy: bool) -> str:
    """Bez casovych znacek spoji po sobe jdouci repliky stejneho mluvciho do
    jednoho odstavce — cte se to jako rozhovor, ne jako seznam useku."""
    radky, predchozi = [], None
    for od, _, s, text in useky:
        if casy:
            radky.append(f"[{_mmss(od)}] {_stitek(s)}{text}")
        elif s is not None and s == predchozi:
            radky[-1] += " " + text
        else:
            radky.append(_stitek(s) + text)
        predchozi = s
    return "\n".join(radky)


def _rozdel_titulky(od: float, do: float, text: str, max_znaku: int = 84):
    """Usek muze mit az 20 s — na titulky je to moc. Rozdeli ho na kusy do
    dvou radku a cas rozpocita podle poctu znaku."""
    casti, akt = [], ""
    for slovo in text.split():
        if akt and len(akt) + 1 + len(slovo) > max_znaku:
            casti.append(akt)
            akt = slovo
        else:
            akt = f"{akt} {slovo}".strip()
    if akt:
        casti.append(akt)
    celkem = sum(len(c) for c in casti) or 1
    vysledek, t = [], od
    for c in casti:
        d = (do - od) * len(c) / celkem
        vysledek.append((t, t + d, c))
        t += d
    return vysledek


def _sestav_srt(useky) -> str:
    titulky = []
    for od, do, s, text in useky:
        for i, (a, b, kus) in enumerate(_rozdel_titulky(od, do, text)):
            titulky.append((a, b, (_stitek(s) if i == 0 else "") + kus))
    return "\n\n".join(f"{i}\n{_s_na_srt(a)} --> {_s_na_srt(b)}\n{t}"
                       for i, (a, b, t) in enumerate(titulky, 1))


def _datum_prepisu() -> str:
    ted = datetime.now()
    return f"{ted.day}. {ted.month}. {ted:%Y %H:%M}"


def _zapis_md(text: str, nadpis: str, cil: Path) -> None:
    hlavicka = (f"# Přepis — {nadpis}\n\n"
                f"*Model: Canary-1B-v2 · {_datum_prepisu()}*\n\n")
    cil.write_text(hlavicka + text + "\n", encoding="utf-8")


def _zapis_docx(text: str, nadpis: str, cil: Path) -> None:
    from docx import Document
    doc = Document()
    doc.add_heading(f"Přepis — {nadpis}", level=1)
    doc.add_paragraph(f"Model: Canary-1B-v2 · {_datum_prepisu()}").italic = True
    for odstavec in text.split("\n"):
        if odstavec.strip():
            doc.add_paragraph(odstavec.strip())
    doc.save(cil)


def _uloz(useky, zaklad: str, nadpis: str, casy: bool, mluvci_zap: bool, extra=()):
    """Zapise .md, .docx a .srt. Vraci (text do okna, soubory, stav)."""
    VYSTUPY.mkdir(parents=True, exist_ok=True)
    zaklad_out = VYSTUPY / zaklad
    dokument = _vysledny_text(useky, casy=False)
    md = zaklad_out.with_suffix(".md")
    _zapis_md(dokument, nadpis, md)
    docx = zaklad_out.with_suffix(".docx")
    _zapis_docx(dokument, nadpis, docx)
    srt = zaklad_out.with_suffix(".srt")
    srt.write_text(_sestav_srt(useky), encoding="utf-8")

    slov = len(" ".join(u[3] for u in useky).split())
    stav = f"{slov} slov · uloženo jako {zaklad_out.name}.*"
    if mluvci_zap:
        stav += f" · mluvčích: {len({u[2] for u in useky if u[2] is not None})}"
    soubory = [str(p) for p in extra] + [str(md), str(docx), str(srt)]
    return _vysledny_text(useky, casy), soubory, stav


# ---------------------------------------------------------------------------
# Rezim: soubor
# ---------------------------------------------------------------------------

def _nacti_zvuk(cesta: Path) -> np.ndarray:
    """Cokoli, co umi ffmpeg (i video), na 16 kHz mono float32."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "vstup.wav"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
                        "-i", str(cesta), "-map", "0:a:0", *FFMPEG_ARGS, str(wav)],
                       check=True, capture_output=True)
        zvuk, _ = sf.read(str(wav), dtype="float32")
    return zvuk


def _jazyky(jazyk_popis: str, prelozit: bool) -> tuple[str, str]:
    jazyk = JAZYKY[jazyk_popis]
    return jazyk, ("en" if (prelozit and jazyk != "en") else jazyk)


def prepis(soubor, video, jazyk_popis, casy, mluvci_zap, prelozit, progress=gr.Progress()):
    # Video ma prednost — kdyz ho nahral, chce prepsat jeho, i kdyz nahore
    # zbyla stara nahravka z predchoziho pokusu.
    zdroj_cesta = video or soubor
    if not zdroj_cesta:
        raise gr.Error("Nahraj zvuk nebo video.")
    jazyk, cil = _jazyky(jazyk_popis, prelozit)

    t0 = time.time()
    progress(0.02, desc="Připravuji zvuk…")
    try:
        zvuk = _nacti_zvuk(Path(zdroj_cesta))
    except subprocess.CalledProcessError as chyba:
        detail = chyba.stderr.decode(errors="replace").strip()
        if "matches no streams" in detail:
            raise gr.Error("Ten soubor nemá zvukovou stopu.") from chyba
        raise gr.Error(f"FFmpeg neumí otevřít soubor. {detail[-300:]}") from chyba
    except (OSError, RuntimeError) as chyba:
        raise gr.Error(f"FFmpeg se nepodařilo spustit. {chyba}") from chyba

    hranice = _useky_souboru(zvuk)
    useky = []
    for i, (a, b) in enumerate(hranice):
        progress(0.05 + 0.8 * i / len(hranice), desc=f"Přepisuji úsek {i + 1}/{len(hranice)}…")
        text = _prepis_kus(zvuk[a:b], jazyk, cil)
        if text:
            useky.append((a / SR, b / SR, None, text))
    if mluvci_zap and useky:
        progress(0.9, desc="Rozlišuji mluvčí (Nemotron 3)…")
        useky = _rozdel_podle_mluvcich(useky, _diarizuj(zvuk), zvuk, jazyk, cil)
    if not useky:
        raise gr.Error("Přepis vyšel prázdný — v nahrávce nejspíš není řeč.")

    znacka = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    nadpis = Path(zdroj_cesta).stem
    jmeno = re.sub(r"[^\w-]+", "-", nadpis)[:40] or "prepis"
    zobrazeny, soubory, stav = _uloz(useky, f"{znacka}-{jmeno}", nadpis, casy, mluvci_zap)
    return zobrazeny, soubory, f"{stav} · za {time.time() - t0:.0f} s"


# ---------------------------------------------------------------------------
# Rezim: nahravani z mikrofonu
# ---------------------------------------------------------------------------
# Prohlizec posila kousky zvuku (Gradio streaming). Kazdy kousek se prevzorkuje
# na 16 kHz a pripoji k nahravce. Hotove useky jdou do fronty a vlakno na
# pozadi je po poradi prepisuje; casovac v UI kazdou sekundu ukaze, co je
# hotove. Nahravka se zaroven prubezne zapisuje do vystupy/, takze se neztrati,
# ani kdyby lab spadl.
#
# Je-li zapnute rozliseni mluvcich, bezi Nemotron PROUDOVE ve vlastnim vlakne
# (krok 0,72 s + 0,32 s dopredu, ~11x rychleji nez realtime) a usek se uzavira
# i na stridani mluvcich, ne jen v pauze. Kazdy usek tak patri jednomu
# mluvcimu a stitek naskoci hned s textem. Po stopu se nic neprepocitava —
# cisla mluvcich zustanou, jak byla videt behem nahravani.
STREAM_S = 0.5
SNIMKU_S = 100           # diarizace: jeden snimek = 10 ms
MIN_BEH_MLUVCIHO = 20    # 0,2 s — kratsi „jiny mluvci" je spis sum nebo prekryv
MIN_RECI_MLUVCIHO = 30   # 0,3 s reci prvniho mluvciho, nez ma smysl riznout

_RELACE: dict[str, dict] = {}
_RELACE_ZAMEK = threading.Lock()
_FRONTA: "queue.Queue[tuple[str, int, np.ndarray]]" = queue.Queue()
_DIAR_FRONTA: "queue.Queue[str]" = queue.Queue()


def _rust(buf: np.ndarray, n: int, pridat: np.ndarray) -> tuple[np.ndarray, int]:
    """Pripoji data do rostouciho bufferu. Kapacita se zdvojnasobuje, at se u
    hodinove nahravky nekopiruje cely zvuk pri kazdem kousku."""
    konec = n + len(pridat)
    if konec > len(buf):
        novy = np.empty((max(konec, 2 * len(buf)),) + buf.shape[1:], dtype=buf.dtype)
        novy[:n] = buf[:n]
        buf = novy
    buf[n:konec] = pridat
    return buf, konec


def _pracovnik():
    while True:
        sid, idx, kus = _FRONTA.get()
        rel = _RELACE.get(sid)
        if rel is None:
            continue
        try:
            text = _prepis_kus(kus, rel["jazyk"], rel["cil"])
        except Exception as chyba:  # jedna chyba nesmi zastavit cele nahravani
            text = ""
            rel["chyba"] = str(chyba)[:200]
        with _RELACE_ZAMEK:
            od, do, mluvci, _ = rel["useky"][idx]
            rel["useky"][idx] = (od, do, mluvci, text)
            rel["hotovo"] += 1


threading.Thread(target=_pracovnik, daemon=True, name="canary-fronta").start()


# --- proudova diarizace ------------------------------------------------------

def _diar_krok(rel: dict, posledni: bool) -> None:
    """Dopocita diarizaci tam, kam uz dosahuje nahravka (postup podle karty
    modelu, sekce Streaming). `posledni` = nahravka skoncila, zpracuje i zbytek."""
    processor, model = _nacti_diarizaci()
    with _DIAR_BEH:
        processor.set_streaming_mode("low_latency")
        while not rel["d_hotovo"]:
            with rel["zamek"]:
                zvuk, n = rel["buf"], rel["n"]
            if rel["d_prvni"]:
                potreba = processor.num_samples_first_audio_chunk
                if n < potreba and not posledni:
                    return
                start, konec = 0, min(n, potreba)
                posledni_kus = posledni and n <= potreba
                dalsi_mel = processor.num_mel_frames_per_step
            else:
                start = processor.audio_chunk_start(rel["d_mel"])
                konec = start + processor.num_samples_per_audio_chunk
                if konec > n and not posledni:
                    return
                posledni_kus = konec > n
                konec = min(konec, n)
                dalsi_mel = rel["d_mel"] + processor.num_mel_frames_per_step
            if konec - start < RAMEC:  # zbytek kratsi nez 30 ms — neni co pocitat
                rel["d_hotovo"] = True
                return
            inputs = processor(zvuk[start:konec], sampling_rate=SR, is_streaming=True,
                               is_first_audio_chunk=rel["d_prvni"],
                               is_last_audio_chunk=posledni_kus)
            with torch.inference_mode():
                vystup = model(**inputs.to(ZARIZENI, dtype=model.dtype),
                               speaker_cache=rel["d_cache"])
            prob = vystup.logits[0].float().sigmoid().cpu().numpy()
            with rel["zamek"]:
                if rel["d_prob"] is None:
                    rel["d_prob"] = np.zeros((60 * SNIMKU_S, prob.shape[1]), np.float32)
                rel["d_prob"], rel["d_n"] = _rust(rel["d_prob"], rel["d_n"], prob)
            rel["d_cache"] = vystup.speaker_cache
            rel["d_prvni"] = False
            rel["d_mel"] = dalsi_mel
            if posledni_kus:
                rel["d_hotovo"] = True


def _diar_pracovnik():
    while True:
        sid = _DIAR_FRONTA.get()
        rel = _RELACE.get(sid)
        if rel is None or not rel["mluvci"] or rel["d_hotovo"]:
            continue
        try:
            _diar_krok(rel, posledni=False)
        except Exception as chyba:
            # Krajeni na diarizaci ceka — kdyby spadla, stal by i prepis.
            # Radsi ji vypnout a prepisovat dal bez mluvcich.
            rel["mluvci"] = False
            rel["chyba"] = f"rozlišení mluvčích selhalo: {chyba}"[:200]


threading.Thread(target=_diar_pracovnik, daemon=True, name="nemotron-proud").start()


def _pravdepodobnosti(rel: dict, od: float, do: float) -> np.ndarray:
    """Snimky diarizace (cas x mluvci) mezi casy, jen ty uz spocitane.
    Volat pod rel["zamek"]."""
    a, b = int(od * SNIMKU_S), min(int(do * SNIMKU_S), rel["d_n"])
    if rel["d_prob"] is None or b <= a:
        return np.zeros((0, 8), np.float32)
    return rel["d_prob"][a:b]


def _hlavni_mluvci(prob: np.ndarray) -> int | None:
    aktivni = (prob > 0.5).sum(axis=0) if len(prob) else np.zeros(1)
    return int(aktivni.argmax()) if aktivni.max() > 0 else None


def _zmena_mluvciho(prob: np.ndarray) -> int | None:
    """Kde se ujme slova jiny mluvci (index snimku od zacatku `prob`), nebo None.
    Rez padne doprostred mezery mezi replikami."""
    if not len(prob):
        return None
    dom = np.where((prob > 0.5).any(axis=1), prob.argmax(axis=1), -1)
    prvni, reci, konec_prvniho = None, 0, 0
    i, n = 0, len(dom)
    while i < n:
        if dom[i] < 0:
            i += 1
            continue
        j = i
        while j < n and dom[j] == dom[i]:
            j += 1
        if prvni is None:
            if j - i >= MIN_BEH_MLUVCIHO:
                prvni, reci, konec_prvniho = dom[i], j - i, j
        elif dom[i] == prvni:
            reci += j - i
            konec_prvniho = j
        elif j - i >= MIN_BEH_MLUVCIHO and reci >= MIN_RECI_MLUVCIHO:
            return (konec_prvniho + i) // 2
        i = j
    return None


# --- relace nahravani --------------------------------------------------------

def _nova_relace(jazyk_popis: str, prelozit: bool, mluvci_zap: bool) -> tuple[str, dict]:
    jazyk, cil = _jazyky(jazyk_popis, prelozit)
    VYSTUPY.mkdir(parents=True, exist_ok=True)
    znacka = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    cesta = VYSTUPY / f"{znacka}-nahravka.wav"
    rel = {
        "zamek": threading.Lock(),
        "jazyk": jazyk, "cil": cil, "znacka": znacka, "mluvci": bool(mluvci_zap),
        "wav": sf.SoundFile(str(cesta), "w", SR, 1, "PCM_16"), "wav_cesta": cesta,
        "buf": np.zeros(60 * SR, np.float32), "n": 0,   # cela nahravka, 16 kHz
        "rez": 0,                                        # vzorek posledniho rezu
        "db": np.zeros(0, np.float32),                   # hlasitost ramcu kvuli odhadu hluku
        "sr_in": None, "resampler": None,
        # proudova diarizace
        "d_prob": None, "d_n": 0, "d_cache": None, "d_mel": 0,
        "d_prvni": True, "d_hotovo": False,
        # prepis: idx -> (od, do, mluvci, text); text None = ceka na Canary
        "useky": {}, "dalsi": 0, "hotovo": 0,
        "konec": False, "konec_cas": 0.0,
        "vysledek": None, "zobrazeno": None, "chyba": None,
    }
    sid = uuid.uuid4().hex
    with _RELACE_ZAMEK:
        _RELACE[sid] = rel
    return sid, rel


def _na_mono(data) -> np.ndarray:
    x = np.asarray(data)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def _pridej(rel: dict, zvuk: np.ndarray) -> None:
    if not len(zvuk):
        return
    rel["wav"].write(zvuk)
    rel["buf"], rel["n"] = _rust(rel["buf"], rel["n"], zvuk)
    rel["db"] = np.concatenate([rel["db"][-4000:], _db(zvuk)])   # posledni ~2 min


def _zpracuj(sid: str, rel: dict, posledni: bool) -> None:
    """Odrizne z cekajiciho zvuku hotove useky a posle je k prepisu.
    Volat pod rel["zamek"]."""
    hluk = float(np.percentile(rel["db"], 10)) if len(rel["db"]) else -60.0
    while rel["rez"] < rel["n"]:
        zacatek = rel["rez"]
        okno = rel["buf"][zacatek: min(rel["n"], zacatek + int((MAX_USEK_S + 1) * SR))]
        rez = _dalsi_rez(okno, hluk, posledni and zacatek + len(okno) == rel["n"])
        mluvci = None
        if rel["mluvci"]:
            od = zacatek / SR
            prob = _pravdepodobnosti(rel, od, od + len(okno) / SR)
            zmena = _zmena_mluvciho(prob)
            if zmena is not None:
                rez_zmena = int(zmena * SR / SNIMKU_S)
                if not rez or rez_zmena < rez:
                    rez = rez_zmena
            elif rez and not rel["d_hotovo"] and rez * SNIMKU_S / SR > len(prob):
                # Diarizace sem jeste nedosla — nevime, jestli se v useku
                # nestridaji mluvci. Pockat na ni (text se zpozdi zhruba o 1 s).
                rez = None
            if rez:
                mluvci = _hlavni_mluvci(prob[: int(rez * SNIMKU_S / SR)])
        if not rez:
            break
        kus = rel["buf"][zacatek: zacatek + rez].copy()
        rel["rez"] = zacatek + rez
        if _ma_rec(kus, hluk):
            with _RELACE_ZAMEK:
                idx = rel["dalsi"]
                rel["dalsi"] += 1
                rel["useky"][idx] = (zacatek / SR, rel["rez"] / SR, mluvci, None)
            _FRONTA.put((sid, idx, kus))


def prijmi_kus(kus, sid, jazyk_popis, prelozit, mluvci_zap):
    """Kousek zvuku z mikrofonu (sr, pole). Musi byt rychle — jen krajeni,
    prepis i diarizace bezi ve vlaknech na pozadi."""
    if kus is None:
        return sid
    sr_in, data = kus
    rel = _RELACE.get(sid) if sid else None
    if rel is not None and rel["konec"]:
        # Opozdeny kousek prave ukoncene nahravky — zahodit, jinak by zalozil
        # novou relaci a na obrazovce by misto vysledku zbylo prazdno.
        if time.time() - rel["konec_cas"] < 2:
            return sid
        rel = None
    if rel is None:
        sid, rel = _nova_relace(jazyk_popis, prelozit, mluvci_zap)
    zvuk = _na_mono(data)
    with rel["zamek"]:
        if rel["sr_in"] is None:
            rel["sr_in"] = sr_in
            if sr_in != SR:
                rel["resampler"] = soxr.ResampleStream(sr_in, SR, 1, dtype="float32")
        if rel["resampler"] is not None:
            zvuk = rel["resampler"].resample_chunk(zvuk)
        _pridej(rel, zvuk)
        _zpracuj(sid, rel, posledni=False)
    if rel["mluvci"]:
        _DIAR_FRONTA.put(sid)
    return sid


def ukonci_nahravani(sid, casy, mluvci_zap, progress=gr.Progress()):
    rel = _RELACE.get(sid) if sid else None
    if rel is None or rel["konec"]:
        return gr.skip(), gr.skip(), gr.skip()
    # Posledni kousek z prohlizece muze dorazit az po udalosti konce.
    time.sleep(STREAM_S + 0.3)
    with rel["zamek"]:
        rel["konec"] = True
        rel["konec_cas"] = time.time()
        if rel["resampler"] is not None:
            _pridej(rel, rel["resampler"].resample_chunk(np.zeros(0, np.float32), last=True))
            rel["resampler"] = None
        rel["wav"].close()
    if rel["mluvci"]:
        progress(0.05, desc="Dopočítávám mluvčí…")
        try:
            _diar_krok(rel, posledni=True)
        except Exception as chyba:
            rel["mluvci"] = False
            rel["chyba"] = f"rozlišení mluvčích selhalo: {chyba}"[:200]
    with rel["zamek"]:
        _zpracuj(sid, rel, posledni=True)

    while True:
        with _RELACE_ZAMEK:
            hotovo, celkem = rel["hotovo"], rel["dalsi"]
        if hotovo >= celkem:
            break
        progress(hotovo / celkem, desc=f"Dopisuji poslední úseky ({hotovo}/{celkem})…")
        time.sleep(0.3)

    zvuk = rel["buf"][: rel["n"]]
    useky = [u for u in (rel["useky"][i] for i in sorted(rel["useky"])) if u[3]]
    mluvci_vystup = rel["mluvci"] or mluvci_zap
    if not rel["mluvci"] and mluvci_zap and useky:
        # Mluvci zapnuti az za nahravani — dopocitat nad celou nahravkou.
        progress(0.9, desc="Rozlišuji mluvčí (Nemotron 3)…")
        useky = _rozdel_podle_mluvcich(useky, _diarizuj(zvuk), zvuk, rel["jazyk"], rel["cil"])
    rel["buf"] = rel["d_prob"] = rel["d_cache"] = None   # uvolnit pamet — nahravka je na disku

    delka = f"nahrávka {len(zvuk) / SR / 60:.1f} min"
    if not useky:
        rel["vysledek"] = ""
        return "", [str(rel["wav_cesta"])], f"V nahrávce nebyla rozpoznaná řeč · {delka} uložena"
    nadpis = f"nahrávka {datetime.now().strftime('%-d. %-m. %Y %H:%M')}"
    zobrazeny, soubory, stav = _uloz(useky, f"{rel['znacka']}-nahravka", nadpis,
                                     casy, mluvci_vystup, extra=[rel["wav_cesta"]])
    rel["vysledek"] = zobrazeny
    if rel["chyba"]:
        stav += f" · {rel['chyba']}"
    return zobrazeny, soubory, f"{stav} · {delka}"


def obnov(sid, casy):
    """Casovac: kazdou sekundu ukaze, co uz je prepsane. Kdyz se text
    nezmenil, nic neposila — jinak by okno skakalo a nedalo se v nem cist."""
    rel = _RELACE.get(sid) if sid else None
    if rel is None:
        if _CANARY is None:
            return gr.skip(), "Načítám Canary do paměti… nahrávat už můžeš, přepis doběhne."
        return gr.skip(), gr.skip()
    if rel["vysledek"] is not None:
        if rel["zobrazeno"] == rel["vysledek"]:
            return gr.skip(), gr.skip()
        rel["zobrazeno"] = rel["vysledek"]
        return rel["vysledek"], gr.skip()
    if rel["konec"]:
        return gr.skip(), gr.skip()   # ukonceni ma vlastni ukazatel prubehu

    with _RELACE_ZAMEK:
        useky = [rel["useky"][i] for i in sorted(rel["useky"])]
        hotovo, celkem = rel["hotovo"], rel["dalsi"]
    stav = f"● Nahrávám {_mmss(rel['n'] / SR)} · přepsáno {hotovo}/{celkem} úseků"
    if rel["mluvci"]:
        stav += f" · mluvčí: {len({u[2] for u in useky if u[2] is not None})}"
    if rel["chyba"]:
        stav += f" · {rel['chyba']}"
    # Rozpracovane useky ukazeme jako „…", at je videt, ze se na nich pracuje.
    text = _vysledny_text([(od, do, s, "…" if t is None else t)
                           for od, do, s, t in useky if t != ""], casy)
    if text == rel["zobrazeno"]:
        return gr.skip(), stav
    rel["zobrazeno"] = text
    return text, stav


def otevri_slozku():
    if sys.platform == "win32":
        os.startfile(VYSTUPY)
    else:
        prikaz = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([prikaz, str(VYSTUPY)], check=True)
    return f"Otevřeno: {VYSTUPY}"


def _predehrej():
    """Nahraje oba modely a jednou je protahne naprazdno. Prvni volani na
    Metalu je pomale (kompilace), at to nepocituje prvni usek nahravky."""
    try:
        _prepis_kus(np.zeros(SR, np.float32), "cs", "cs")
        _diarizuj(np.zeros(2 * SR, np.float32))
        processor, model = _nacti_diarizaci()
        with _DIAR_BEH, torch.inference_mode():
            _diarizuj_proudove(processor, model, np.zeros(3 * SR, np.float32))
    except Exception as chyba:
        print(f"Předehřátí modelů selhalo: {chyba}")


with gr.Blocks(title="STT Lab", analytics_enabled=False) as demo:
    gr.Markdown(
        "# 🎧 STT — Canary-1B-v2\n"
        "Přepis **Canary-1B-v2** (NVIDIA, WER 7,9 % na češtině), rozlišení mluvčích "
        "**Nemotron 3 Diarization** (NVIDIA). Oba modely zpracovávají zvuk lokálně.\n\n"
        "**Nahrávat:** klikni na mikrofon a mluv. Text naskakuje po větách se zpožděním "
        "pár sekund — lab čeká na pauzu, aby neusekl slovo. S rozlišením mluvčích se úsek "
        "uzavře i při střídání mluvčích a štítek naskočí hned s textem. Zvuk se průběžně "
        "ukládá do složky s výstupy.\n\n"
        "Při prvním spuštění se stahují modely. Na Windows běží přepis na procesoru."
    )
    sid = gr.State(None)
    with gr.Row():
        jazyk = gr.Dropdown(list(JAZYKY), value="čeština", label="Jazyk",
                            info="u nahrávání platí od začátku nahrávky")
        casy = gr.Checkbox(value=True, label="Časové značky")
        mluvci_zap = gr.Checkbox(value=False, label="Rozlišit mluvčí",
                                 info="Nemotron 3 — průběžně; zapni před nahráváním")
        prelozit = gr.Checkbox(value=False, label="Přeložit do angličtiny")

    with gr.Tabs():
        with gr.Tab("🎙️ Nahrávat"):
            mikrofon = gr.Audio(sources=["microphone"], streaming=True, type="numpy",
                                label="Mikrofon")
            zivy = gr.Textbox(label="Přepis — naskakuje průběžně", lines=18, autoscroll=True)
            stav_nahravani = gr.Textbox(label="Stav", lines=1, interactive=False)
            soubory_nahravani = gr.Files(label="Ke stažení — nahrávka .wav, .md, .docx, titulky .srt")

        with gr.Tab("📁 Soubor"):
            with gr.Row():
                with gr.Column(scale=3):
                    soubor = gr.Audio(label="Nahrávka", type="filepath", sources=["upload"])
                    video = gr.File(label="…nebo video / jiný soubor", type="filepath",
                                    file_types=["video", "audio"])
                    tlacitko = gr.Button("Přepsat", variant="primary", size="lg")
                with gr.Column(scale=3):
                    vystup = gr.Textbox(label="Přepis", lines=18)
                    soubory = gr.Files(label="Ke stažení — .md, .docx a titulky .srt")
                    stav = gr.Textbox(label="Stav", lines=2, interactive=False)

    slozka = gr.Button("Otevřít složku s výstupy", size="sm")
    casovac = gr.Timer(1.0)

    mikrofon.stream(prijmi_kus, inputs=[mikrofon, sid, jazyk, prelozit, mluvci_zap],
                    outputs=[sid], stream_every=STREAM_S, time_limit=None,
                    show_progress="hidden")
    mikrofon.stop_recording(ukonci_nahravani, inputs=[sid, casy, mluvci_zap],
                            outputs=[zivy, soubory_nahravani, stav_nahravani])
    casovac.tick(obnov, inputs=[sid, casy], outputs=[zivy, stav_nahravani],
                 show_progress="hidden")
    tlacitko.click(prepis, inputs=[soubor, video, jazyk, casy, mluvci_zap, prelozit],
                   outputs=[vystup, soubory, stav])
    slozka.click(otevri_slozku, outputs=stav)


def main():
    parser = argparse.ArgumentParser(description="Spustí místní webové rozhraní STT Lab.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Načte modely až při prvním použití.")
    args = parser.parse_args()
    VYSTUPY.mkdir(parents=True, exist_ok=True)
    print(f"STT Lab · http://127.0.0.1:{args.port} · {ZARIZENI.upper()}", flush=True)
    if not args.no_warmup:
        print("Načítám modely. První spuštění může stáhnout přibližně 4 GB.", flush=True)
        threading.Thread(target=_predehrej, daemon=True, name="predehrati").start()
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                        inbrowser=args.open_browser, allowed_paths=[str(VYSTUPY)])


if __name__ == "__main__":
    main()
