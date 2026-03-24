# nllb_translator

## Overview

**nllb_translator** appears to be a local translation toolkit built around an NLLB-style model, with three user-facing components:
- a Python translation backend,
- a tray application,
- a browser extension package (`.xpi`) for translating webpages.

The project is oriented toward local translation rather than cloud APIs.

## Top-level contents

- `499bd024a58c4fa5ba70-0.2.0.xpi`
- `icon.png`
- `nllb_server.py`
- `requirements.txt`
- `start_combo.sh`
- `translate.py`
- `tray_translator.py`

## What this project appears to do

### Translation backend
`nllb_server.py` suggests a local server process that keeps the model loaded and accepts translation requests.

### Standalone translation
`translate.py` likely provides a direct translation interface for scripts or command-line use.

### Desktop tray control
`tray_translator.py` strongly suggests a lightweight system tray application for desktop interaction.

### Browser integration
The bundled `.xpi` suggests a Firefox/LibreWolf extension that talks to the local translation backend to translate webpages.

## Installation

```bash
git clone https://github.com/RauchendeKanonen/nllb_translator.git
cd nllb_translator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the model is not downloaded automatically, the README should explain where it comes from and how much disk and RAM it needs.

## Running

Backend:

```bash
python3 nllb_server.py
```

Tray app:

```bash
python3 tray_translator.py
```

Combined startup:

```bash
./start_combo.sh
```

## Browser extension note

Because the repository includes an unsigned `.xpi`, the README should explicitly document:
- which browsers were tested,
- how the extension connects to the local service,
- what permissions it needs,
- the security implications of side-loading it.

## License

No visible license from the public top-level snapshot.
