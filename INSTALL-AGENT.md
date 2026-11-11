# Nainstaluj mi STT Lab

Tento soubor můžeš přiložit do Codexu, Claude Code nebo jiného asistenta.

## Zadání pro asistenta

Pomoz mi nainstalovat a spustit STT Lab. Aplikace lokálně přepisuje řeč z mikrofonu nebo souboru, umí rozlišit mluvčí a exportovat do Markdownu, Wordu a titulků SRT.

1. Ověř můj operační systém a typ procesoru. Vezmi instalační příkaz z [README vydání v0.1.0](https://github.com/christovnino-a11y/stt-lab/blob/v0.1.0/README.md). Podporovaný je Mac s Apple Silicon a Windows x64. Pokud mám jiný systém, řekni mi to.
2. Použij příkaz pro můj systém. Pokud můžeš spouštět příkazy v mém počítači, spusť ho. Jinak mi dej jeden příkaz ke zkopírování do Terminálu nebo PowerShellu. První spuštění stáhne modely NVIDIA z Hugging Face, přibližně 4 GB.
3. Otevři místní adresu, kterou vypíše aplikace. Nech mě povolit mikrofon v prohlížeči a zkusit krátký přepis. Pak mě proveď přepisem souboru, rozlišením mluvčích a stažením výstupů `.md`, `.docx` a `.srt`.
4. Při potížích se řiď konkrétní chybou v terminálu. U prvního spuštění zkontroluj dostupnost Hugging Face, u mikrofonu oprávnění prohlížeče. Pokud běží jiná aplikace na portu 7864, použij spouštěcí příkaz z README s parametrem `--port 7869`. Na konci mi napiš, co prošlo, jak aplikaci spustím příště a kde najdu výstupy.
