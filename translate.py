from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

# -----------------------------
# Common interface / base types
# -----------------------------

@dataclass(frozen=True)
class TranslationResult:
    text: str


class Translator(ABC):
    """Swappable translation interface."""

    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        raise NotImplementedError

    def translate_batch(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> List[TranslationResult]:
        """Default batch implementation (can be overridden for speed)."""
        return [self.translate(t, src_lang, tgt_lang) for t in texts]


# -----------------------------
# NLLB implementation (local)
# -----------------------------

class NLLBTranslator(Translator):
    """
    Local NLLB translator using HuggingFace Transformers.

    - Auto uses GPU if available (unless forced to 'cpu')
    - Uses safetensors if available
    - Supports fast batching (important for PDFs)
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,          # None => auto
        use_fp16_on_cuda: bool = True,
        max_input_length: int = 512,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
        print(model_name)
        self.torch = torch
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Tokenizer: use NllbTokenizer (slow) to avoid missing lang helpers in fast backend
        self.tokenizer = NllbTokenizer.from_pretrained(model_name)

        dtype = torch.float16 if (device == "cuda" and use_fp16_on_cuda) else torch.float32

        # use_safetensors=True avoids torch.load restrictions on older torch versions
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(device)

        self.model.eval()

        # Small warm-up can reduce first-call latency (optional)
        self._warmed_up = False
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        try:
            import torch
            torch._dynamo.disable()
        except Exception:
            pass

    def warmup(self, src_lang: str = "eng_Latn", tgt_lang: str = "deu_Latn") -> None:
        """Optional: run a tiny generation once to warm kernels/caches."""
        if self._warmed_up:
            return
        _ = self.translate("hello", src_lang=src_lang, tgt_lang=tgt_lang)
        self._warmed_up = True

    def _tgt_token_id(self, tgt_lang: str) -> int:
        # For NLLB, language codes like "deu_Latn" are actual vocab tokens
        tid = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        if tid is None or tid < 0:
            raise ValueError(f"Unknown target language token: {tgt_lang}")
        return int(tid)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        out = self.translate_batch([text], src_lang, tgt_lang)[0]
        return out

    def translate_batch(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> List[TranslationResult]:
        import torch

        # Keep alignment with input indices
        # Avoid wasting work on empty strings
        idx_map = []
        clean_texts = []
        for i, t in enumerate(texts):
            t2 = (t or "")
            if t2.strip():                 # only for emptiness check
                idx_map.append(i)
                clean_texts.append(t2)     # preserve whitespace/newlines

        results: List[Optional[TranslationResult]] = [None] * len(texts)
        for i, t in enumerate(texts):
            if not (t or "").strip():
                results[i] = TranslationResult(text="")

        if not clean_texts:
            return [r for r in results if r is not None]  # type: ignore

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            list(clean_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        forced_bos = self._tgt_token_id(tgt_lang)
        input_len = inputs["input_ids"].shape[1]
        max_new = min(self.max_new_tokens, int(input_len * 1.5) + 10)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,

                max_new_tokens=max_new,

                num_beams=4,
                early_stopping=True,

                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        for j, translated in enumerate(decoded):
            original_index = idx_map[j]
            results[original_index] = TranslationResult(text=translated)

        # At this point, all are filled
        return [r for r in results if r is not None]  # type: ignore


# --------------------------------
# Google Translate implementation
# --------------------------------

class GoogleDeepTranslator(Translator):
    """
    Online translation via deep-translator (GoogleTranslator).

    Keeps the same interface so you can swap it with NLLBTranslator.
    """

    def __init__(self) -> None:
        from deep_translator import GoogleTranslator
        self.GoogleTranslator = GoogleTranslator

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        t = (text or "").strip()
        if not t:
            return TranslationResult(text="")

        # deep-translator uses ISO-like short codes (e.g. "hr", "de")
        # whereas NLLB uses "hrv_Latn", "deu_Latn"
        # So you either pass "hr"/"de" here OR add a mapping layer outside.
        translator = self.GoogleTranslator(source=src_lang, target=tgt_lang)
        return TranslationResult(text=translator.translate(t))


# -----------------------------
# Optional: small language mapper
# -----------------------------

NLLB_TO_ISO = {
    "hrv_Latn": "hr",
    "deu_Latn": "de",
    "eng_Latn": "en",
    # add more as needed
}

ISO_TO_NLLB = {v: k for k, v in NLLB_TO_ISO.items()}


class MappedTranslator(Translator):
    """
    Wrap any translator and map language codes.
    Useful to keep your app using NLLB codes everywhere (hrv_Latn/deu_Latn),
    while Google expects hr/de, etc.
    """

    def __init__(self, inner: Translator, src_map: dict, tgt_map: dict):
        self.inner = inner
        self.src_map = src_map
        self.tgt_map = tgt_map

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> TranslationResult:
        return self.inner.translate(
            text,
            self.src_map.get(src_lang, src_lang),
            self.tgt_map.get(tgt_lang, tgt_lang),
        )

    def translate_batch(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> List[TranslationResult]:
        return self.inner.translate_batch(
            texts,
            self.src_map.get(src_lang, src_lang),
            self.tgt_map.get(tgt_lang, tgt_lang),
        )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Local NLLB (uses GPU automatically)
    nllb = NLLBTranslator()
    nllb.warmup("hrv_Latn", "deu_Latn")

    print(nllb.translate("Ovo je službeni dokument.", "hrv_Latn", "deu_Latn").text)

    # Google (wrap with mapping so caller can still use NLLB codes)
    google = MappedTranslator(
        GoogleDeepTranslator(),
        src_map=NLLB_TO_ISO,
        tgt_map=NLLB_TO_ISO,
    )
    print(google.translate("Ovo je službeni dokument.", "hrv_Latn", "deu_Latn").text)

    # Batch example (best for PDFs)
    batch = ["Dobar dan.", "Ovo je službeni dokument.", ""]
    print([r.text for r in nllb.translate_batch(batch, "hrv_Latn", "deu_Latn")])
