# STT Lab

Lokální přepis češtiny a dalších jazyků z mikrofonu, zvukového souboru nebo videa. Umí rozlišit mluvčí a vyexportovat přepis do Markdownu, Wordu a titulků SRT. Přepis i rozlišení mluvčích běží na tvém počítači.

## Instalace jedním příkazem

### Mac s Apple Silicon

Vlož do Terminálu na Macu s čipem M1 nebo novějším a macOS 14 nebo novějším.

```bash
curl -fsSL https://raw.githubusercontent.com/christovnino-a11y/stt-lab/v0.1.0/install.sh | /bin/bash
```

### Windows x64

Vlož do PowerShellu na 64bitových Windows s procesorem Intel nebo AMD.

```powershell
& ([scriptblock]::Create((Invoke-RestMethod 'https://raw.githubusercontent.com/christovnino-a11y/stt-lab/v0.1.0/install.ps1')))
```

Příkaz nainstaluje uv, Python 3.12, aplikaci a její závislosti včetně FFmpeg do složky tvého uživatele. Účet na GitHubu, API klíč ani práva správce nejsou potřeba. Zdrojové skripty si můžeš prohlédnout před spuštěním. [Mac](install.sh) · [Windows](install.ps1)

Po instalaci se otevře [STT Lab v prohlížeči](http://127.0.0.1:7864). Při prvním spuštění potřebuješ internet a přibližně 8 GB volného místa pro instalaci a váhy modelů. Samotné modely mají přibližně 4 GB. Doporučená RAM je 16 GB. Na Windows běží přepis na CPU, takže zpracování může být pomalejší než na Apple Silicon.

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

## Instalace s pomocí AI

Přilož soubor [INSTALL-AGENT.md](INSTALL-AGENT.md) do Codexu, Claude Code nebo jiného asistenta. Obsahuje postup pro instalaci a kontrolu, že funguje přepis i export.

## Modely

Aplikace při prvním spuštění stáhne veřejné váhy NVIDIA z Hugging Face a uloží je do místní cache. Samotné váhy mají vlastní licence. [Canary-1B-v2, CC BY 4.0](https://huggingface.co/nvidia/canary-1b-v2) · [Nemotron 3 Diarization, OpenMDW 1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization)
