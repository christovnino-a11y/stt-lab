"""Check the installed copy using a synthetic fixture or an optional speech file."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
import wave


project = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project))

with tempfile.TemporaryDirectory(prefix="STT test ") as tmp:
    data = Path(tmp)
    os.environ["STT_LAB_DATA_DIR"] = str(data)
    import app

    if len(sys.argv) > 2:
        # Run the app's normal file-processing path with both real models.
        text, files, status = app.prepis(
            sys.argv[2], None, "angličtina", True, True, False,
        )
        assert "transcription" in text.lower(), text
        assert "[Mluvčí" in text, text
        print(status)
    else:
        wav = data / "zvuk s mezerou.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", ""))
            stream.writeframes(bytes(32000))
        assert app._nacti_zvuk(wav).shape == (16000,)
        assert app.gr.processing_utils.audio_from_file(str(wav))[0] == 16000
        _, files, _ = app._uloz(
            [(0, 1, 0, "Zkouška přepisu.")], "test", "Zkouška", True, True,
        )
    outputs = {Path(path).suffix: Path(path) for path in files}
    assert outputs.keys() == {".md", ".docx", ".srt"}, files
    assert all(path.stat().st_size for path in outputs.values()), files
    assert "-->" in outputs[".srt"].read_text(encoding="utf-8")
    assert app.VYSTUPY == data.resolve() / "vystupy"

    if len(sys.argv) == 2:
        with subprocess.Popen(
            [sys.executable, str(project / "app.py"), "--no-warmup", "--port", "7869"],
            cwd=project,
        ) as server:
            try:
                for _ in range(60):
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited with code {server.returncode}")
                    try:
                        with urlopen("http://127.0.0.1:7869/", timeout=2) as response:
                            assert b"STT Lab" in response.read()
                            break
                    except URLError:
                        time.sleep(1)
                else:
                    raise RuntimeError("The local STT Lab server did not start")
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()

print("STT Lab installation checks passed.")
