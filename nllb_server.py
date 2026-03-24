#!/usr/bin/env python3
"""
HTTP translation server for translate.py

Endpoints:
- GET  /health
- POST /warmup
- POST /translate
- POST /translate_batch

Run:
  pip install fastapi uvicorn pydantic
  python server.py --host 0.0.0.0 --port 8080

Env options:
  TRANSLATOR = nllb | google          (default: nllb)
  MODEL_NAME = HF model name          (default: facebook/nllb-200-distilled-600M)
  DEVICE     = cpu | cuda | auto      (default: auto)
"""

from __future__ import annotations

import os
import re
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


from typing import Dict, Tuple

# Build a best-effort reverse map: ISO -> NLLB
# NOTE: multiple NLLB codes can map to the same ISO; we pick a deterministic first.
# NLLB -> ISO-639-1 (or closest ISO where 2-letter doesn't exist)
NLLB_TO_ISO = {
    # --- Major European ---
    "eng_Latn": "en",
    "deu_Latn": "de",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "nld_Latn": "nl",
    "dan_Latn": "da",
    "swe_Latn": "sv",
    "nor_Latn": "no",
    "fin_Latn": "fi",
    "isl_Latn": "is",
    "gle_Latn": "ga",
    "gla_Latn": "gd",
    "cym_Latn": "cy",
    "eus_Latn": "eu",
    "cat_Latn": "ca",
    "glg_Latn": "gl",

    # --- Germanic minorities ---
    "fry_Latn": "fy",
    "ltz_Latn": "lb",

    # --- Romance minorities ---
    "oci_Latn": "oc",
    "ast_Latn": "ast",
    "vec_Latn": "vec",
    "scn_Latn": "scn",
    "srd_Latn": "sc",
    "ron_Latn": "ro",
    "mol_Cyrl": "ro",  # historical Moldovan

    # --- Slavic ---
    "pol_Latn": "pl",
    "ces_Latn": "cs",
    "slk_Latn": "sk",
    "slv_Latn": "sl",
    "hrv_Latn": "hr",
    "bos_Latn": "bs",
    "srp_Cyrl": "sr",
    "srp_Latn": "sr",
    "mkd_Cyrl": "mk",
    "bul_Cyrl": "bg",
    "rus_Cyrl": "ru",
    "ukr_Cyrl": "uk",
    "bel_Cyrl": "be",

    # --- Baltic ---
    "lit_Latn": "lt",
    "lav_Latn": "lv",
    "est_Latn": "et",

    # --- Uralic minorities ---
    "hun_Latn": "hu",
    "sme_Latn": "se",  # Northern Sami

    # --- Balkan / SE Europe ---
    "ell_Grek": "el",
    "sqi_Latn": "sq",
    "tur_Latn": "tr",

    # --- Caucasus (Europe border) ---
    "kat_Geor": "ka",
    "hye_Armn": "hy",
    "aze_Latn": "az",
    "aze_Cyrl": "az",

    # --- Major global (recommended for autodetect robustness) ---
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "ara_Arab": "ar",
    "heb_Hebr": "he",
    "fas_Arab": "fa",
    "urd_Arab": "ur",
    "hin_Deva": "hi",
    "ben_Beng": "bn",
    "mar_Deva": "mr",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "tha_Thai": "th",
    "vie_Latn": "vi",
    "ind_Latn": "id",
    "msa_Latn": "ms",
    "fil_Latn": "tl",
    "swa_Latn": "sw",
    "amh_Ethi": "am",

    # --- Others often seen in real traffic ---
    "lat_Latn": "la",
    "epo_Latn": "eo",
}

ISO_TO_NLLB = {}
for nllb_code, iso in NLLB_TO_ISO.items():
    ISO_TO_NLLB.setdefault(iso, nllb_code)

import re
from typing import List, Tuple

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZČĆĐŠŽ])', re.UNICODE)

def _split_preserve_separators(text: str) -> List[Tuple[str, str]]:
    """
    Split into blocks but preserve the separator that follows each block.
    Returns list of (block, sep_after_block).
    """
    # Split on blank lines, keep the blank line separators
    parts = re.split(r'(\n\s*\n+)', text)
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(parts):
        block = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if block == "" and sep:
            # edge case: text starts with separators
            out.append(("", sep))
        else:
            out.append((block, sep))
        i += 2
    return out

def _split_sentences_fallback(paragraph: str) -> List[str]:
    """
    Fallback sentence-ish split. Not perfect for all languages, but good enough.
    Keeps line breaks inside paragraph by treating each line separately.
    """
    lines = paragraph.split("\n")
    sent_blocks: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            sent_blocks.append("")  # keep empty line
            continue
        # Split on punctuation + whitespace + (capital letter) heuristic
        sents = _SENT_SPLIT_RE.split(line)
        sent_blocks.extend(sents)
    return sent_blocks

def split_text_to_token_chunks(translator, text: str, max_input_length: int, safety_margin: int = 16) -> List[str]:
    """
    Token-budgeted chunking using the same tokenizer as the model.
    """
    budget = max(32, max_input_length - safety_margin)
    chunks: List[str] = []
    cur = ""

    def tok_len(s: str) -> int:
        # cheap length check using tokenizer (no tensors)
        return len(translator.tokenizer(s, add_special_tokens=True, truncation=False)["input_ids"])

    blocks = _split_preserve_separators(text)

    for block, sep in blocks:
        candidate = (block + sep)
        if not candidate.strip():
            # pure whitespace/seps: just append to current if possible
            if cur and tok_len(cur + candidate) <= budget:
                cur += candidate
            else:
                if cur:
                    chunks.append(cur)
                    cur = ""
                chunks.append(candidate)
            continue

        # If it fits, try to pack into current
        if cur and tok_len(cur + candidate) <= budget:
            cur += candidate
            continue

        # If current has content, flush it
        if cur:
            chunks.append(cur)
            cur = ""

        # If the block+sep fits alone, take it
        if tok_len(candidate) <= budget:
            cur = candidate
            continue

        # Too large: split within the block (sentences fallback)
        # We will re-add the original sep at the very end of paragraph handling.
        subparts = _split_sentences_fallback(block)
        sub_cur = ""
        for j, sub in enumerate(subparts):
            # Recreate with line breaks between original lines.
            # _split_sentences_fallback loses exact spacing; we add a single space between sentences.
            piece = (sub if sub_cur == "" else (" " + sub)) if sub else ("\n" if sub_cur else "\n")
            if sub_cur and tok_len(sub_cur + piece) > budget:
                chunks.append(sub_cur)
                sub_cur = piece.lstrip()
            else:
                sub_cur += piece
        if sub_cur:
            # add the paragraph separator to the last sub-chunk
            sub_cur += sep
            chunks.append(sub_cur)
        else:
            # nothing produced, still preserve sep
            chunks.append(sep)

    if cur:
        chunks.append(cur)

    # Final cleanup: avoid empty chunks
    return [c for c in chunks if c != ""]


def sanitize_for_translation(text: str) -> str:
    """
    Normalize browser-copied text that may contain invisible/control characters
    (NUL, NBSP, zero-width chars, unicode separators) which can sometimes lead
    to partial/garbled model output.
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


def detect_iso_lang(text: str) -> str:
    """
    Detect language and return an ISO code (usually ISO-639-1, e.g. 'en', 'de', 'hr').
    Requires 'langid' to be installed.
    """
    try:
        import langid  # type: ignore
    except Exception as e:
        raise ValueError(
            "src_lang='auto' requires the optional dependency 'langid'. "
            "Install it with: pip install langid"
        ) from e

    # langid.classify returns (lang, score)
    lang, _score = langid.classify(text)
    return lang


def resolve_src_lang(tr: Translator, text: str, src_lang: str) -> Tuple[str, str]:
    """
    Returns (resolved_src_lang, resolved_src_iso).
    - If src_lang != 'auto': resolved_src_lang == src_lang, src_iso best-effort.
    - If src_lang == 'auto': detect ISO and map to what the translator expects.
    """
    if src_lang != "auto":
        # best-effort ISO for reporting
        src_iso = NLLB_TO_ISO.get(src_lang, src_lang)
        return src_lang, src_iso

    src_iso = detect_iso_lang(text)

    # If we're using NLLB locally, map ISO -> NLLB code if possible
    if isinstance(tr, NLLBTranslator):
        if src_iso not in ISO_TO_NLLB:
            raise ValueError(
                f"Auto-detected src language '{src_iso}' but no ISO->NLLB mapping is available."
            )
        return ISO_TO_NLLB[src_iso], src_iso

    # GoogleDeepTranslator (wrapped in MappedTranslator) is happiest with ISO codes
    return src_iso, src_iso





# Import your existing implementation
from translate import (  # <-- your file translate.py
    Translator,
    NLLBTranslator,
    GoogleDeepTranslator,
    MappedTranslator,
#   NLLB_TO_ISO,
)

# -----------------------------
# Configuration / lazy init
# -----------------------------

_app = FastAPI(title="Translation Server", version="1.0")

from fastapi.middleware.cors import CORSMiddleware

_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


_translator: Optional[Translator] = None
_init_lock = threading.Lock()


def _build_translator() -> Translator:
    mode = os.environ.get("TRANSLATOR", "nllb").strip().lower()
    device = os.environ.get("DEVICE", "auto").strip().lower()
    small_model = "facebook/nllb-200-distilled-600M"
    mid_model = "facebook/nllb-200-1.3B"
    big_model = "facebook/nllb-200-3.3B"
    model_name = os.environ.get("MODEL_NAME", big_model).strip()
   
    if device == "auto":
        device_arg = None
    elif device in ("cpu", "cuda"):
        device_arg = device
    else:
        raise ValueError("DEVICE must be cpu|cuda|auto")

    if mode == "nllb":
        # Local HF model (GPU if available unless DEVICE forces cpu)
        return NLLBTranslator(model_name=model_name, device=device_arg)
    elif mode == "google":
        # Online translator; wrap with mapping so callers can still use NLLB codes if they want
        return MappedTranslator(
            GoogleDeepTranslator(),
            src_map=NLLB_TO_ISO,
            tgt_map=NLLB_TO_ISO,
        )
    else:
        raise ValueError("TRANSLATOR must be nllb|google")


def get_translator() -> Translator:
    global _translator
    if _translator is not None:
        return _translator
    with _init_lock:
        if _translator is None:
            _translator = _build_translator()
    return _translator


# -----------------------------
# Request/response models
# -----------------------------

class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    src_lang: str = Field(..., description="Source language (e.g. hrv_Latn, hr, or 'auto')")
    tgt_lang: str = Field(..., description="Target language (e.g. deu_Latn or de)")


class TranslateResponse(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    detected_src_lang: Optional[str] = None  # ISO (e.g. 'hr') when auto was used


class TranslateBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to translate")
    src_lang: str
    tgt_lang: str


class TranslateBatchResponse(BaseModel):
    texts: List[str]
    src_lang: str
    tgt_lang: str
    detected_src_langs: Optional[List[str]] = None  # ISO per item when auto is used


class WarmupRequest(BaseModel):
    src_lang: str = "eng_Latn"
    tgt_lang: str = "deu_Latn"


# -----------------------------
# Routes
# -----------------------------

@_app.get("/health")
def health():
    # Do not force model load; just report config and whether loaded
    return {
        "ok": True,
        "loaded": _translator is not None,
        "translator_mode": os.environ.get("TRANSLATOR", "nllb"),
    }


@_app.post("/warmup")
def warmup(req: WarmupRequest):
    tr = get_translator()
    # Only NLLBTranslator has warmup; ignore for others
    if hasattr(tr, "warmup"):
        try:
            tr.warmup(req.src_lang, req.tgt_lang)  # type: ignore[attr-defined]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")
    return {"ok": True}



_BR_TO_NL_RE = re.compile(r"\s*<\s*br\s*/?\s*>\s*", re.IGNORECASE)

@_app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    tr = get_translator()
    try:
        clean_text = sanitize_for_translation(req.text)
        print(clean_text)
        print(req.src_lang)
        resolved_src, detected_iso = resolve_src_lang(tr, clean_text, req.src_lang)

        # Replace newlines with a visible marker that MT models tend to preserve better.
        # Keep spaces around it so it tokenizes as its own token most of the time.
        text_for_mt = clean_text.replace("\n", " <br> ")

        # Chunk safely within tokenizer limits
        chunks = split_text_to_token_chunks(tr, text_for_mt, tr.max_input_length/8, safety_margin=16)

        # Translate
        translated_chunks = tr.translate_batch(chunks, resolved_src, req.tgt_lang)
        out_text = "".join(r.text for r in translated_chunks)

        # Convert <br> markers back to real newlines (tolerate spacing/case variations)
        out_text = _BR_TO_NL_RE.sub("\n", out_text)
        print(out_text)
        return TranslateResponse(
            text=out_text,
            src_lang=resolved_src,
            tgt_lang=req.tgt_lang,
            detected_src_lang=(detected_iso if req.src_lang == "auto" else None),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@_app.post("/translate_batch", response_model=TranslateBatchResponse)
def translate_batch(req: TranslateBatchRequest):
    tr = get_translator()
    try:
        out_texts: List[str] = []
        detected_isos: List[str] = []
        resolved_srcs: List[str] = []

        for raw_text in req.texts:
            print(f"Src Lang: {req.src_lang}", flush=True)
            print(f"Tgt Lang: {req.tgt_lang}", flush=True)
            print(f"Src Txt: {raw_text}", flush=True)

            clean_text = sanitize_for_translation(raw_text)

            resolved_src, detected_iso = resolve_src_lang(tr, clean_text, req.src_lang)
            resolved_srcs.append(resolved_src)
            detected_isos.append(detected_iso if req.src_lang == "auto" else None)

            max_tokens = int(getattr(tr, "max_input_length", 1024)) - 16
            if max_tokens < 32:
                max_tokens = 32

            chunks = split_text_to_token_chunks(
                tr,
                clean_text,
                max_tokens,
                safety_margin=16
            )

            translated_chunks = tr.translate_batch(chunks, resolved_src, req.tgt_lang)
            out_text = "".join(r.text for r in translated_chunks)

            print(f"Tgt Txt: {out_text}", flush=True)
            out_texts.append(out_text)

        if req.src_lang == "auto":
            src_lang_out = (
                resolved_srcs[0]
                if resolved_srcs and all(s == resolved_srcs[0] for s in resolved_srcs)
                else "mixed"
            )
            return TranslateBatchResponse(
                texts=out_texts,
                src_lang=src_lang_out,
                tgt_lang=req.tgt_lang,
                detected_src_langs=detected_isos,
            )

        return TranslateBatchResponse(
            texts=out_texts,
            src_lang=req.src_lang,
            tgt_lang=req.tgt_lang,
            detected_src_langs=None,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    uvicorn.run(
        _app,   # <-- pass the app object directly
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
