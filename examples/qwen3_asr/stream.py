"""
Streaming ASR session for Qwen3-ASR matmul decoder example.

Provides sliding-window chunking with prefix rollback for long audio.
Simplified from jetson-voice (no VAD, no speculative encoding).
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Callable

SAMPLE_RATE = 16000


def _detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    """Detect and remove excessive repetitions in transcribed text."""
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i + count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        found = False
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while (end_index + k <= n
                           and s[end_index:end_index + k] == pattern):
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(
                        s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return ''.join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def _parse_asr_output(raw: str, user_language: str = None):
    """
    Parse raw LLM output into (language, text).

    The model may output: "language Chinese<asr_text>transcribed text..."
    or just plain text if language was forced in prompt.
    """
    if not raw:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = _detect_and_fix_repetitions(s)

    if user_language:
        return user_language, s

    ASR_TEXT_TAG = "<asr_text>"
    LANG_PREFIX = "language "

    if ASR_TEXT_TAG in s:
        meta, text = s.split(ASR_TEXT_TAG, 1)
        if "language none" in meta.lower():
            return "", text.strip() if text.strip() else ""
        lang = ""
        for line in meta.splitlines():
            line = line.strip()
            if line.lower().startswith(LANG_PREFIX):
                val = line[len(LANG_PREFIX):].strip()
                if val:
                    lang = val[:1].upper() + val[1:].lower()
                break
        return lang, text.strip()
    else:
        return "", s.strip()


def _apply_itn(text: str) -> str:
    """
    Apply Inverse Text Normalization (Chinese number conversion etc.).

    Returns original text if ITN module is not available.
    """
    try:
        from qwen_asr_gguf import chinese_itn
        return chinese_itn.chinese_to_num(text)
    except ImportError:
        pass

    try:
        import chinese_itn
        return chinese_itn.chinese_to_num(text)
    except ImportError:
        pass

    return text


class StreamSession:
    """
    Streaming ASR session with sliding-window chunking.

    Usage::

        stream = StreamSession(engine, language="Chinese")
        stream.feed_audio(pcm_chunk)
        result = stream.get_result()
        final = stream.finish()
    """

    def __init__(self, engine,
                 language: Optional[str] = "Chinese",
                 context: str = "",
                 chunk_size: float = 5.0,
                 memory_num: int = 2,
                 unfixed_chunks: int = 2,
                 rollback_tokens: int = 5,
                 max_new_tokens: int = 128,
                 on_text: Callable = None,
                 vad=None):
        """
        Args:
            engine:           Parent Qwen3ASREngine instance
            language:         Language hint (None = auto-detect)
            context:          Context description
            chunk_size:       Seconds per audio chunk (capped by encoder)
            memory_num:       Sliding-window width (>= 2)
            unfixed_chunks:   First N chunks don't commit text
            rollback_tokens:  Remove last N tokens from prefix for stability
            max_new_tokens:   Max tokens to generate per chunk
            on_text:          Callback(text: str) on each new text
            vad:              Unused (kept for interface compat)
        """
        self.engine = engine
        self.language = language
        self.context = context
        self.chunk_size = min(chunk_size, engine.max_chunk_seconds)
        self.memory_num = max(2, memory_num)
        self.unfixed_chunks = unfixed_chunks
        self.rollback_tokens = rollback_tokens
        self.max_new_tokens = max_new_tokens
        self.on_text = on_text

        self.chunk_samples = int(self.chunk_size * SAMPLE_RATE)
        self.buffer = np.zeros(0, dtype=np.float32)

        # Sliding window of (embedding, committed_text) pairs
        self._segments = deque(maxlen=self.memory_num)
        self._archive_text = ""
        self._chunk_id = 0
        self._current_text = ""
        self._current_language = language or ""

        # Stats
        self._total_enc_ms = 0.0
        self._total_llm_ms = 0.0
        self._total_audio_s = 0.0
        self._total_chunks = 0

    # -------------------------------------------------------------- #
    # Public API                                                      #
    # -------------------------------------------------------------- #

    def feed_audio(self, pcm16k: np.ndarray) -> dict:
        """
        Feed audio data and process any complete chunks.

        Returns:
            dict with keys: language, text, is_final, chunks_processed
        """
        x = np.asarray(pcm16k, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0

        self.buffer = np.concatenate([self.buffer, x])
        chunks_processed = 0
        while len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples:]
            self._process_chunk(chunk)
            chunks_processed += 1

        return {
            "language": self._current_language,
            "text": self._current_text,
            "is_final": False,
            "is_speech": True,
            "chunks_processed": chunks_processed,
        }

    def get_result(self) -> dict:
        """Get current recognition result without processing new audio."""
        return {
            "language": self._current_language,
            "text": self._current_text,
            "is_final": False,
            "is_speech": True,
        }

    def finish(self, apply_itn_flag: bool = True) -> dict:
        """
        Finish streaming: process remaining buffer and return final result.

        Returns:
            dict with keys: language, text, is_final, stats
        """
        # Process remaining raw buffer
        if len(self.buffer) > 0:
            buf = self.buffer
            if len(buf) < int(0.5 * SAMPLE_RATE):
                buf = np.pad(buf, (0, int(0.5 * SAMPLE_RATE) - len(buf)))
            self._process_chunk(buf)
            self.buffer = np.zeros(0, dtype=np.float32)

        text = self._current_text
        if apply_itn_flag:
            text = _apply_itn(text)

        return {
            "language": self._current_language,
            "text": text,
            "is_final": True,
            "stats": {
                "total_enc_ms": self._total_enc_ms,
                "total_llm_ms": self._total_llm_ms,
                "total_audio_s": self._total_audio_s,
                "total_vad_ms": 0.0,
                "total_chunks": self._total_chunks,
                "utterances": 0,
                "avg_utterance_latency_ms": 0,
                "rtf": ((self._total_enc_ms + self._total_llm_ms) / 1000.0
                        / max(self._total_audio_s, 0.1)),
            },
        }

    # -------------------------------------------------------------- #
    # Core processing pipeline                                        #
    # -------------------------------------------------------------- #

    def _process_chunk(self, audio_chunk: np.ndarray):
        """Encode audio chunk, then decode with sliding window."""
        chunk_sec = len(audio_chunk) / SAMPLE_RATE
        self._total_audio_s += chunk_sec

        t0 = time.perf_counter()
        result_enc = self.engine.encoder.encode(audio_chunk)
        if isinstance(result_enc, tuple):
            audio_embd = result_enc[0]
        else:
            audio_embd = result_enc
        enc_ms = (time.perf_counter() - t0) * 1000
        self._total_enc_ms += enc_ms

        self._decode_with_window(audio_embd, chunk_sec, enc_ms)

    def _decode_with_window(self, audio_embd: np.ndarray,
                            chunk_sec: float, enc_ms: float):
        """Sliding window -> rollback -> decode -> commit."""
        # 1. Update sliding window
        if len(self._segments) >= self.memory_num:
            oldest_embd, oldest_text = self._segments.popleft()
            self._archive_text += oldest_text
        self._segments.append((audio_embd, ""))

        # 2. Concatenate audio embeddings from window
        all_audio = np.concatenate([s[0] for s in self._segments], axis=0)

        # 3. Prefix text from in-window completed segments only
        raw_prefix = "".join(
            self._segments[i][1] for i in range(len(self._segments) - 1))

        # 4. Token rollback
        prefix_str = self._apply_rollback(raw_prefix)

        # 5. Build full embedding & decode
        full_embd, n_tokens = self.engine.build_embed(
            all_audio, prefix_str, self.language, self.context)

        t1 = time.perf_counter()
        result = self.engine.decoder.run_embed(full_embd, n_tokens)
        llm_ms = (time.perf_counter() - t1) * 1000
        self._total_llm_ms += llm_ms

        # 6. Parse output
        raw_text = result["text"]
        was_aborted = result.get("aborted", False)

        if self.language:
            new_text, lang = raw_text, self.language
        else:
            lang, new_text = _parse_asr_output(raw_text)

        if was_aborted:
            new_text = ""
        if self._chunk_id < self.unfixed_chunks:
            new_text = ""

        # 7. Rollback alignment: trim preceding segment's text
        if self.rollback_tokens > 0 and len(self._segments) > 1:
            prev_texts = [self._segments[i][1]
                          for i in range(len(self._segments) - 1)]
            earlier = "".join(prev_texts[:-1])
            trimmed = prefix_str[len(earlier):]
            idx = len(self._segments) - 2
            self._segments[idx] = (self._segments[idx][0], trimmed)

        # 8. Commit new text
        last_embd, _ = self._segments[-1]
        self._segments[-1] = (last_embd, new_text)

        self._current_text = (self._archive_text
                              + "".join(s[1] for s in self._segments))
        self._current_language = lang or self._current_language
        self._chunk_id += 1
        self._total_chunks += 1

        if self.on_text:
            self.on_text(self._current_text)

        # 9. Performance log
        total_ms = enc_ms + llm_ms
        rtf = total_ms / 1000.0 / max(chunk_sec, 0.01)
        perf = result.get("perf", {})
        rb = f" rb={self.rollback_tokens}" if self.rollback_tokens else ""
        abort = " [ABORTED]" if was_aborted else ""
        print(f"  [chunk {self._chunk_id}] enc={enc_ms:.0f}ms "
              f"llm={llm_ms:.0f}ms rtf={rtf:.2f} "
              f"prefill={perf.get('prefill_time_ms', 0):.0f}ms "
              f"gen_tok={perf.get('generate_tokens', 0)}{rb}"
              f"{abort} | {new_text[:60]}...", flush=True)

    def _apply_rollback(self, text: str) -> str:
        """Strip the last ``rollback_tokens`` tokens from text."""
        if self.rollback_tokens <= 0 or not text:
            return text
        ids = self.engine.tokenizer.encode(text).ids
        if len(ids) <= self.rollback_tokens:
            return ""
        return self.engine.tokenizer.decode(ids[:-self.rollback_tokens])
