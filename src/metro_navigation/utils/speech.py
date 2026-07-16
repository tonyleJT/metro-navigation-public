"""Non-blocking text-to-speech support."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

from metro_navigation.config import SpeechSettings

LOGGER = logging.getLogger(__name__)
_STOP = object()


class NullSpeaker:
    """Speaker implementation that intentionally produces no audio."""

    def say(self, text: str, *, force: bool = False) -> bool:
        return bool(text)

    def stop(self) -> None:
        return None


@dataclass(slots=True, frozen=True)
class _SpeechItem:
    text: str


class Speaker:
    """Queue speech on a worker thread so inference remains non-blocking."""

    def __init__(self, settings: SpeechSettings) -> None:
        self._settings = settings
        self._queue: queue.Queue[_SpeechItem | object] = queue.Queue()
        self._last_accepted_at = 0.0
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name="metro-navigation-tts",
            daemon=True,
        )
        self._thread.start()

    def say(self, text: str, *, force: bool = False) -> bool:
        """Queue speech unless the global cooldown rejects it."""

        text = text.strip()
        if not text:
            return False

        now = time.monotonic()
        with self._lock:
            if self._stopped:
                return False
            if not force and now - self._last_accepted_at < self._settings.global_cooldown_seconds:
                return False
            self._last_accepted_at = now
            self._queue.put(_SpeechItem(text))
        return True

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._settings.rate_words_per_minute)
            engine.setProperty("volume", self._settings.volume)
        except Exception:
            LOGGER.exception("Failed to initialize text-to-speech; audio is disabled")
            self._drain_until_stopped()
            return

        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                assert isinstance(item, _SpeechItem)
                LOGGER.info("Speaking: %s", item.text)
                engine.say(item.text)
                engine.runAndWait()
        finally:
            engine.stop()

    def _drain_until_stopped(self) -> None:
        while self._queue.get() is not _STOP:
            pass

    def stop(self) -> None:
        """Stop the worker once; safe to call during exception cleanup."""

        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._queue.put(_STOP)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            LOGGER.warning("Text-to-speech worker did not stop within 10 seconds")
