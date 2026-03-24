import sys
import traceback

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QSettings
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QApplication,
    QSystemTrayIcon,
    QMenu,
    QMainWindow,
    QDockWidget,
    QTextEdit,
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QMessageBox,
)


import faulthandler
faulthandler.enable()


# -----------------------------
# Selection watcher (debounced)
# -----------------------------
from PySide6.QtGui import QGuiApplication, QClipboard

import json
import re
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"


def sanitize_for_translation(text: str) -> str:
    """
    Browser selections can contain invisible/control characters (NUL, NBSP,
    zero-width chars, unicode separators) that sometimes lead to partial model
    output. Normalize them for the request while keeping the original for UI.
    """
    if not text:
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")

    text = text.replace("\u00A0", " ").replace("\u202F", " ")

    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"):
        text = text.replace(ch, "")

    text = "".join((c if (c == "\n" or c == "\t" or ord(c) >= 0x20) else " ") for c in text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()

def http_json(method: str, url: str, payload=None, timeout=30):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
            return r.status, json.loads(body) if body else None
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except URLError as e:
        return None, str(e)


def wait_for_server():
    
    for _ in range(50):
        status, body = http_json("GET", f"{BASE}/health", None, timeout=2)
        if status == 200:
            return body
        time.sleep(0.1)
    raise RuntimeError("Server did not become ready")





class SelectionWatcher(QObject):
    """
    Watches PRIMARY selection (mouse select) and CLIPBOARD (Ctrl+C).
    Debounces rapid changes so you don't translate while user is still dragging selection.
    """
    text_ready = Signal(str)

    def __init__(self, debounce_ms: int = 350, min_len: int = 2, max_len: int = 4000):
        super().__init__()
        self.clipboard = QGuiApplication.clipboard()
        self.debounce_ms = debounce_ms
        self.min_len = min_len
        self.max_len = max_len

        self._pending_text = ""
        self._last_emitted = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._emit_pending)

        # PRIMARY selection (if supported)
        self.clipboard.selectionChanged.connect(self._on_selection_changed)
        # Ctrl+C clipboard
        self.clipboard.dataChanged.connect(self._on_clipboard_changed)

    def _on_selection_changed(self):
        # PRIMARY selection (mouse select)
        text = self.clipboard.text(QClipboard.Selection)
        self._set_pending(text)

    def _on_clipboard_changed(self):
        # Standard clipboard (Ctrl+C)
        text = self.clipboard.text(QClipboard.Clipboard)
        self._set_pending(text)

    def _set_pending(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        if len(text) < self.min_len:
            return
        if len(text) > self.max_len:
            return

        self._pending_text = text
        self._timer.start(self.debounce_ms)

    def _emit_pending(self):
        text = (self._pending_text or "").strip()
        if text and text != self._last_emitted:
            self._last_emitted = text
            self.text_ready.emit(text)

    def current_selection_text(self) -> str:
        # Try PRIMARY first, fall back to CLIPBOARD
        t = (self.clipboard.text(QClipboard.Selection) or "").strip()
        if t:
            return t
        return (self.clipboard.text(QClipboard.Clipboard) or "").strip()


# -----------------------------
# UI: floating dock "window"
# -----------------------------
class MainHost(QMainWindow):
    """
    Hidden host window required for QDockWidget.
    We won't show this; we only show the floating dock.
    """
    def __init__(self):
        super().__init__()
        self.setCentralWidget(QWidget())

        self.dock = QDockWidget("Translator", self)
        self.dock.setObjectName("TranslatorDock")
        self.dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )

        self.output = QTextEdit()
        self.output.setPlaceholderText("Select text, then translation will appear here…")
        self.dock.setWidget(self.output)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        # Start as its own floating window
        self.dock.setFloating(True)
        self.dock.resize(640, 360)


# -----------------------------
# Config dialog
# -----------------------------

LANG_PRESETS = [
    # Common (NLLB codes)
    "eng_Latn",
    "deu_Latn",
    "hrv_Latn",
    "fra_Latn",
    "spa_Latn",
    "ita_Latn",
    "por_Latn",
    "nld_Latn",
    "pol_Latn",
    "ces_Latn",
    "slv_Latn",
    "srp_Latn",
    "srp_Cyrl",
    "bul_Cyrl",
    "rus_Cyrl",
    "ukr_Cyrl",
]


class ConfigDialog(QDialog):
    def __init__(self, parent: QWidget, src_lang: str, tgt_lang: str):
        super().__init__(parent)
        self.setWindowTitle("Translator Configuration")
        self.setModal(True)

        root = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Source language:"))
        self.src_combo = QComboBox()
        self.src_combo.setEditable(True)
        self.src_combo.addItem("auto")
        for c in LANG_PRESETS:
            self.src_combo.addItem(c)
        self._set_combo_value(self.src_combo, src_lang)
        row1.addWidget(self.src_combo, 1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Target language:"))
        self.tgt_combo = QComboBox()
        self.tgt_combo.setEditable(True)
        for c in LANG_PRESETS:
            self.tgt_combo.addItem(c)
        self._set_combo_value(self.tgt_combo, tgt_lang)
        row2.addWidget(self.tgt_combo, 1)
        root.addLayout(row2)

        hint = QLabel(
            "Tip: Use NLLB codes like 'deu_Latn' or 'hrv_Latn'.\n"
            "For autodetection set source to 'auto'."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        root.addLayout(buttons)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str):
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText(value)

    def values(self):
        return (self.src_combo.currentText().strip(), self.tgt_combo.currentText().strip())




class TextInputDialog(QDialog):
    def __init__(self, parent: QWidget, src_lang: str, tgt_lang: str):
        super().__init__(parent)
        self.setWindowTitle("Translate Text")
        self.setModal(True)
        self.resize(720, 420)

        root = QVBoxLayout(self)

        langs = QHBoxLayout()
        langs.addWidget(QLabel("Source:"))
        self.src_combo = QComboBox()
        self.src_combo.setEditable(True)
        self.src_combo.addItem("auto")
        for c in LANG_PRESETS:
            self.src_combo.addItem(c)
        ConfigDialog._set_combo_value(self.src_combo, src_lang)
        langs.addWidget(self.src_combo, 1)

        langs.addWidget(QLabel("Target:"))
        self.tgt_combo = QComboBox()
        self.tgt_combo.setEditable(True)
        for c in LANG_PRESETS:
            self.tgt_combo.addItem(c)
        ConfigDialog._set_combo_value(self.tgt_combo, tgt_lang)
        langs.addWidget(self.tgt_combo, 1)
        root.addLayout(langs)

        root.addWidget(QLabel("Text to translate:"))
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Enter or paste text here…")
        root.addWidget(self.input_edit, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        translate_btn = QPushButton("Translate")
        cancel_btn = QPushButton("Close")
        translate_btn.clicked.connect(self.on_translate_clicked)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(translate_btn)
        buttons.addWidget(cancel_btn)
        root.addLayout(buttons)

        self.output_edit = QTextEdit()
        root.addWidget(self.output_edit, 2)
        self.output_edit.setReadOnly(True)
        
        self.status_label = QLabel("")
        root.addWidget(self.status_label, 1)
        
    def on_translate_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        self.status_label.setText("Translating...")

        sanitized = sanitize_for_translation(text)

        status, body = http_json(
            "POST",
            f"{BASE}/translate",
            {
                "text": sanitized,
                "src_lang": self.src_combo.currentText().strip(),
                "tgt_lang": self.tgt_combo.currentText().strip(),
            },
        )

        if status != 200 or not isinstance(body, dict):
            self.status_label.setText("Error")
            self.output_edit.setPlainText(str(body))
            return

        translated = sanitize_for_translation(body.get("text", ""))
        self.output_edit.setPlainText(translated)
        self.status_label.setText("Done")
    
    def values(self):
        return (
            self.input_edit.toPlainText().strip(),
            self.src_combo.currentText().strip(),
            self.tgt_combo.currentText().strip(),
        )

# -----------------------------
# Tray app controller
# -----------------------------
class TrayApp:
    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings()

        # Translation config (persisted)
        self.src_lang = self.settings.value("translator/src_lang", "auto", type=str)
        self.tgt_lang = self.settings.value("translator/tgt_lang", "eng_Latn", type=str)
        self.auto_translate = self.settings.value("ui/auto_translate", True, type=bool)
        self.keep_on_top = self.settings.value("ui/keep_on_top", True, type=bool)

        # Host window + floating dock
        self.host = MainHost()
        self.dock = self.host.dock

        # Always-on-top behavior for dock
        self._apply_on_top()

        # Lazy translator (loaded on first use)
        self._translator = None

        # Selection watcher
        self.watcher = SelectionWatcher(debounce_ms=350)
        self.watcher.text_ready.connect(self.on_text_selected)

        # Tray
        self.tray = QSystemTrayIcon(QIcon("icon.png"), self.app)
        self.tray.setToolTip("NLLB Translator")

        self.menu = QMenu()

        self.action_toggle = QAction("Show translator")
        self.action_toggle.triggered.connect(self.toggle_dock)
        self.menu.addAction(self.action_toggle)

        self.action_translate_now = QAction("Translate current selection now")
        self.action_translate_now.triggered.connect(self.translate_current_selection_now)
        self.menu.addAction(self.action_translate_now)

        self.action_translate_text = QAction("Translate typed text…")
        self.action_translate_text.triggered.connect(self.translate_typed_text)
        self.menu.addAction(self.action_translate_text)

        self.menu.addSeparator()

        self.action_auto = QAction("Auto-translate selection", self.menu)
        self.action_auto.setCheckable(True)
        self.action_auto.setChecked(self.auto_translate)
        self.action_auto.triggered.connect(self.toggle_auto_translate)
        self.menu.addAction(self.action_auto)

        self.action_ontop = QAction("Keep translator on top", self.menu)
        self.action_ontop.setCheckable(True)
        self.action_ontop.setChecked(self.keep_on_top)
        self.action_ontop.triggered.connect(self.toggle_on_top)
        self.menu.addAction(self.action_ontop)

        self.menu.addSeparator()

        self.action_config = QAction("Configuration…", self.menu)
        self.action_config.triggered.connect(self.open_config)
        self.menu.addAction(self.action_config)

        self.menu.addSeparator()

        self.action_quit = QAction("Quit")
        self.action_quit.triggered.connect(self.quit)
        self.menu.addAction(self.action_quit)

        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()

        # Start hidden
        self.dock.hide()
        self._update_toggle_text()
        health = wait_for_server()


    def _apply_on_top(self):
        # Make the floating dock stay on top (works best on X11; on Wayland compositor may limit)
        flags = self.dock.windowFlags()
        if self.keep_on_top:
            self.dock.setWindowFlags(flags | Qt.Tool | Qt.WindowStaysOnTopHint)
        else:
            # remove on-top hint; keep it as tool window
            self.dock.setWindowFlags((flags | Qt.Tool) & ~Qt.WindowStaysOnTopHint)

        # Important: after changing flags, re-show if it was visible
        was_visible = self.dock.isVisible()
        if was_visible:
            self.dock.hide()
            self.dock.show()
            self.dock.raise_()
            self.dock.activateWindow()

    def on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.toggle_dock()

    def _update_toggle_text(self):
        self.action_toggle.setText("Hide translator" if self.dock.isVisible() else "Show translator")

    def toggle_dock(self):
        if self.dock.isVisible():
            self.dock.hide()
        else:
            self.dock.setFloating(True)
            self._apply_on_top()
            self.dock.show()
            self.dock.raise_()
            self.dock.activateWindow()
        self._update_toggle_text()

    def toggle_auto_translate(self):
        self.auto_translate = self.action_auto.isChecked()
        self.settings.setValue("ui/auto_translate", self.auto_translate)

    def toggle_on_top(self):
        self.keep_on_top = self.action_ontop.isChecked()
        self.settings.setValue("ui/keep_on_top", self.keep_on_top)
        self._apply_on_top()

    def open_config(self):
        dlg = ConfigDialog(self.dock, self.src_lang, self.tgt_lang)
        if dlg.exec() != QDialog.Accepted:
            return

        src, tgt = dlg.values()
        if not src or not tgt:
            QMessageBox.warning(self.dock, "Invalid configuration", "Source and target must not be empty.")
            return
        if tgt == "auto":
            QMessageBox.warning(self.dock, "Invalid configuration", "Target language cannot be 'auto'.")
            return

        self.src_lang = src
        self.tgt_lang = tgt
        self.settings.setValue("translator/src_lang", self.src_lang)
        self.settings.setValue("translator/tgt_lang", self.tgt_lang)

    def translate_current_selection_now(self):
        text = self.watcher.current_selection_text()
        if not text:
            self.host.output.setPlainText("(No selection/clipboard text)")
            self._ensure_visible_for_result()
            return
        self._start_translation(text)

    def translate_typed_text(self):
        dlg = TextInputDialog(self.dock, self.src_lang, self.tgt_lang)
        if dlg.exec() != QDialog.Accepted:
            return

        text, src, tgt = dlg.values()
        if not text:
            QMessageBox.warning(self.dock, "No text", "Please enter some text to translate.")
            return
        if not src or not tgt:
            QMessageBox.warning(self.dock, "Invalid configuration", "Source and target must not be empty.")
            return
        if tgt == "auto":
            QMessageBox.warning(self.dock, "Invalid configuration", "Target language cannot be 'auto'.")
            return

        self.src_lang = src
        self.tgt_lang = tgt
        self.settings.setValue("translator/src_lang", self.src_lang)
        self.settings.setValue("translator/tgt_lang", self.tgt_lang)
        self._start_translation(text)

    def on_text_selected(self, text: str):
        if not self.auto_translate:
            return
        self._start_translation(text)

    def _ensure_visible_for_result(self):
        if not self.dock.isVisible():
            self.toggle_dock()

    def _start_translation(self, text: str):
        self._ensure_visible_for_result()
        self.host.output.setPlainText(
            f"Translating ({self.src_lang} → {self.tgt_lang})…\n\n{text}"
        )

        sanitized = sanitize_for_translation(text)

        status, body = http_json(
            "POST",
            f"{BASE}/translate_batch",
           {"texts": [sanitized], "src_lang": self.src_lang, "tgt_lang": self.tgt_lang},
        )
        print("Translate:", status, body)

        if status != 200:
            self._on_translation_error(str(body))
            return
        if not isinstance(body, dict) or "texts" not in body:
            self._on_translation_error(f"Unexpected response: {body!r}")
            return

        self._on_translation_done(text, body)

    def _on_translation_done(self, src_text: str, resp: dict):
        translated = resp.get("text", "")
        src_lang = resp.get("src_lang", self.src_lang)
        tgt_lang = resp.get("tgt_lang", self.tgt_lang)
        detected = resp.get("detected_src_lang")

        lines = []
        lines.append(f"{src_lang} → {tgt_lang}")
        if self.src_lang == "auto" and detected:
            lines.append(f"Detected: {detected}")
        lines.append("")
        lines.append("Original:")
        lines.append(src_text)
        lines.append("")
        lines.append("Translation:")
        translated_list = resp.get("texts", [])
        translated = translated_list[0] if translated_list else ""
        translated = sanitize_for_translation(translated)
        lines.append(translated)

        self.host.output.setPlainText("\n".join(lines))

    def _on_translation_error(self, err: str):
        self.host.output.setPlainText("ERROR while translating:\n\n" + err)

    def quit(self):
        self.tray.hide()
        self.app.quit()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # Ensure QSettings are stored consistently across runs
    app.setOrganizationName("nllb")
    app.setOrganizationDomain("local")
    app.setApplicationName("nllb-translator")
    app.setDesktopFileName("nllb-translator")

    # IMPORTANT: keep a reference so it won't be garbage-collected
    tray_app = TrayApp(app)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
