# STT Lab

Lokální přepis češtiny a dalších jazyků z mikrofonu, zvukového souboru nebo videa. Umí rozlišit mluvčí a vyexportovat přepis do Markdownu, Wordu a titulků SRT. Přepis i rozlišení mluvčích běží na tvém počítači.

## Instalace jedním příkazem

### Mac s Apple Silicon

Vlož do Terminálu.

```bash
curl -fsSL https://raw.githubusercontent.com/christovnino-a11y/stt-lab/v0.1.0/install.sh | /bin/bash
```

### Windows x64

Vlož do PowerShellu.

```powershell
& ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/christovnino-a11y/stt-lab/v0.1.0/install.ps1')))
```

Příkaz nainstaluje uv, Python 3.12, aplikaci a její závislosti včetně FFmpeg do složky tvého uživatele. Účet na GitHubu, API klíč ani práva správce nejsou potřeba. Zdrojové skripty si můžeš prohlédnout před spuštěním. [Mac](install.sh) · [Windows](install.ps1)

Po instalaci se otevře [STT Lab v prohlížeči](http://127.0.0.1:7864). Při prvním spuštění potřebuješ internet ke stažení modelů.

## Hardwarové nároky

- **Mac** s Apple Silicon (M1 nebo novější) a macOS 14 nebo novějším. PyTorch využije integrovanou grafiku přes Metal (MPS), pokud je dostupná. Jinak aplikace běží na procesoru.
- **Windows x64** s procesorem Intel nebo AMD. V této verzi běží přepis i rozlišení mluvčích na CPU. Instalaci a přepis jsme ověřili na virtuálním stroji se 4 vCPU a 16 GB RAM. [Ověření na Windows](https://github.com/christovnino-a11y/stt-lab/actions/runs/42989102447) · [Parametry testovacího stroje](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories)
- **Paměť a disk.** Pro běžný provoz doporučujeme 16 GB RAM. Pro instalaci nech alespoň 8 GB volného místa, z toho přibližně 4 GB zabírají modely. Pro vlastní nahrávky počítej s dalším místem.

Samostatnou grafickou kartu aplikace nevyžaduje. Při testu krátké nahrávky s přepisem i rozlišením mluvčích na Macu v režimu CPU měl proces špičku přibližně 4,3 GiB RAM.

## Spuštění příště

**Mac**

```bash
"$HOME/Library/Application Support/STT Lab/stt-lab"
```

**Windows**

```powershell
& "$env:LOCALAPPDATA\STT Lab\STT Lab.cmd"
```

Aplikaci ukončíš pomocí Ctrl+C v okně, ze kterého běží. Pokud port 7864 používá jiná aplikace, přidej za spouštěcí příkaz `--port 7869`.

## První přepis

- V záložce **Nahrávat** povol prohlížeči mikrofon. Výsledek se doplňuje v průběhu nahrávání po úsecích.
- V záložce **Soubor** vyber audio nebo video a klikni na **Přepsat**. Pro štítky mluvčích zaškrtni **Rozlišit mluvčí**. Při použití mikrofonu zapni tuto volbu před nahráváním.
- Výsledky stáhneš z rozhraní nebo najdeš přes **Otevřít složku s výstupy**. Na Macu jsou v `~/Library/Application Support/STT Lab/vystupy`, na Windows v `%LOCALAPPDATA%\STT Lab\vystupy`.

Pokud se instalace přeruší, spusť stejný příkaz znovu. První přepis může chvíli čekat na dokončení stahování a načtení modelů.

## Co dělají jednotlivé nástroje

- **Canary-1B-v2** převádí řeč na text a umí ji volitelně přeložit do angličtiny.
- **Nemotron 3 Diarization** rozpoznává, kdy v nahrávce mluví jednotliví lidé. Když zapneš **Rozlišit mluvčí**, STT Lab podle něj přidá k větám štítky jako `[Mluvčí 1]` a `[Mluvčí 2]`.
- **Instalační skripty** připraví prostředí v uživatelské složce, nainstalují Python 3.12, aplikaci a její knihovny. Ke správě Pythonu a závislostí používají **uv**.
- **Python** spouští aplikaci a její logiku. **Gradio** zajišťuje webové rozhraní, které se otevře v prohlížeči na tomto počítači.
- **FFmpeg** vytáhne zvuk z audia nebo videa a převede ho na mono se vzorkovací frekvencí 16 kHz. **soundfile** čte a ukládá zvuk, **NumPy** pracuje se vzorky, **soxr** převzorkuje mikrofonní vstup a **librosa** pomáhá připravit zvuk pro model.
- **PyTorch** spouští výpočty modelů. **Transformers** načítá jejich váhy a připravuje vstupy pro přepis a rozlišení mluvčích.
- **Hugging Face** hostuje veřejné váhy modelů, které aplikace stáhne při prvním spuštění.
- Aplikace ukládá hotové přepisy do místní složky `vystupy`. **python-docx** vytváří soubory pro Word. Aplikace přímo zapisuje i Markdown a titulky SRT.

## Instalace s pomocí AI

Přilož soubor [INSTALL-AGENT.md](INSTALL-AGENT.md) do Codexu, Claude Code nebo jiného asistenta. Obsahuje postup pro instalaci a kontrolu, že funguje přepis i export.

## Modely

Aplikace při prvním spuštění stáhne veřejné váhy NVIDIA z Hugging Face a uloží je do místní cache. Samotné váhy mají vlastní licence. [Canary-1B-v2, CC BY 4.0](https://huggingface.co/nvidia/canary-1b-v2) · [Nemotron 3 Diarization, OpenMDW 1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization)
