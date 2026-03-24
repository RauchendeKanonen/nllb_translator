#!/bin/bash
source /home/florian/bin/nllb/.venv/bin/activate
python /home/florian/bin/nllb/nllb_server.py &
sleep 1 &&
python /home/florian/bin/nllb/tray_translator.py
