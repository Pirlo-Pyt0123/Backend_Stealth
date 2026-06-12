"""
VoiceNarrator — LUCY's voice module.

Converts DeepTrafficEngine analysis results into contextual spoken narration
using Microsoft Edge Neural TTS (edge-tts) — the same neural Transformer voices
behind Azure Cognitive Services, free and without API key.

Voice options (change EDGE_VOICE):
    es-ES-AlvaroNeural   male,   Spain
    es-ES-ElviraNeural   female, Spain
    es-MX-DaliaNeural    female, Mexico
    es-MX-JorgeNeural    male,   Mexico

Fallback: pyttsx3 (offline, SAPI5) if no internet available.

Usage:
    narrator = VoiceNarrator()
    narrator.speak(engine_result)   # non-blocking
"""

from __future__ import annotations
import os
import sys
import time
import queue
import asyncio
import tempfile
import threading
import subprocess
from typing import Optional

EDGE_VOICE = "es-BO-MarceloNeural"


# ── Contextual narrative generator ────────────────────────────────────────────

def build_narrative(result: dict) -> str:
    """Genera mensaje educativo para niños según la situación detectada."""

    # ── Pelea / conflicto (prioridad máxima — siempre crítico) ──────────────
    if result.get("fight_detected"):
        return "¡Cuidado pequeño! Hay una pelea muy cerca. Aléjate rápido y busca un adulto."

    level     = result.get("risk_level", "SAFE")
    stats     = result.get("stats", {})
    vehiculos = stats.get("vehiculos", 0)
    bicis     = stats.get("bicicletas", 0)
    grandes   = stats.get("grandes", 0)

    if level == "SAFE":
        return "¡Muy bien, estás seguro! Recuerda siempre usar el paso de cebra al cruzar."

    if level == "WARNING":
        if bicis > 0:
            return "¡Cuidado! Hay un ciclista cerca. Espera a que pase y cruza por el paso de cebra."
        if grandes > 0:
            return "¡Cuidado! Hay un camión o bus cerca. Espera a que pase y usa el paso de cebra."
        if vehiculos > 0:
            return "Ten cuidado al cruzar. Mira a la izquierda y a la derecha, y usa el paso de cebra."
        return "Ten cuidado en esta zona. Usa el paso de cebra para cruzar."

    # CRITICAL
    if grandes > 0:
        return "¡Cuidado pequeño! Hay un vehículo grande muy cerca. Retírate a la acera ahora."
    if vehiculos > 0:
        return "¡Cuidado pequeño! Hay carros muy cerca. Sal de la carretera y ponte en un lugar seguro."
    return "¡Cuidado pequeño! Estás en peligro. Aléjate de la carretera y busca un adulto."


# ── Audio daemon — proceso persistente que corre MCI en su hilo principal ───────
#
# MCI falla desde hilos de fondo de Python. La solución es lanzar UN proceso
# daemon al inicio y enviarle las rutas MP3 por stdin. El daemon reproduce cada
# archivo usando MCI desde su propio hilo principal (confiable en cualquier
# contexto: uvicorn, FastAPI, threads, etc.) y responde "done\n" cuando termina.

_DAEMON_CODE = """\
import sys, ctypes
m = ctypes.windll.winmm.mciSendStringW
while True:
    path = sys.stdin.buffer.readline().decode().strip()
    if not path:
        break
    m('open "' + path + '" type mpegvideo alias a', None, 0, None)
    m('play a wait', None, 0, None)
    m('close a', None, 0, None)
    sys.stdout.buffer.write(b'done\\n')
    sys.stdout.buffer.flush()
"""


# ── Voice narrator ─────────────────────────────────────────────────────────────

class VoiceNarrator:
    """Non-blocking voice narrator for LUCY using edge-tts neural voices."""

    # Tiempo mínimo entre dos frases seguidas — evita disparos dobles por jitter
    COOLDOWN_TRANSITION = 3.0

    def __init__(self):
        self._mode               = "loading"
        self._last_traffic_level = None    # último nivel de tráfico anunciado
        self._last_fight_state   = False   # último estado de pelea anunciado
        self._last_spoken        = 0.0
        self._queue              = queue.Queue(maxsize=1)
        self._ready_event        = threading.Event()

        # Proceso daemon persistente para reproducción MCI en hilo principal
        self._audio_proc = subprocess.Popen(
            [sys.executable, "-c", _DAEMON_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        t = threading.Thread(target=self._worker, daemon=True, name="lucy-voice")
        t.start()
        print(f"[VoiceNarrator] Inicializando edge-tts ({EDGE_VOICE})...")

    def speak(self, result: dict) -> None:
        """Habla solo cuando cambia el estado — una vez por widget abierto."""
        if self._mode in ("loading", "none"):
            return

        now           = time.time()
        fight_now     = result.get("fight_detected", False) and result.get("fight_ready", False)
        traffic_level = result.get("risk_level", "SAFE")

        # Anti-jitter: ignora si acabamos de hablar hace menos de COOLDOWN_TRANSITION segundos
        if now - self._last_spoken < self.COOLDOWN_TRANSITION:
            return

        # Canal pelea: habla solo cuando la pelea EMPIEZA (False → True)
        if fight_now and not self._last_fight_state:
            self._last_fight_state = True
            self._last_spoken      = now
            self._enqueue(build_narrative(result))
            return

        # Limpia estado de pelea cuando termina
        if not fight_now:
            self._last_fight_state = False

        # Canal tráfico: habla solo cuando cambia el nivel de riesgo
        # (no habla mientras hay pelea activa para no solapar audios)
        if not fight_now and traffic_level != self._last_traffic_level:
            self._last_traffic_level = traffic_level
            self._last_spoken        = now
            self._enqueue(build_narrative(result))

    def _enqueue(self, text: str) -> None:
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def wait_until_ready(self, timeout: float = 30) -> bool:
        return self._ready_event.wait(timeout=timeout)

    @property
    def status(self) -> dict:
        return {
            "mode":  self._mode,
            "ready": self._mode not in ("loading", "none"),
            "voice": EDGE_VOICE,
        }

    def shutdown(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._audio_proc.stdin.write(b"\n")
            self._audio_proc.stdin.flush()
            self._audio_proc.wait(timeout=2)
        except Exception:
            self._audio_proc.kill()

    def _play_mp3(self, path: str) -> None:
        try:
            self._audio_proc.stdin.write(f"{path}\n".encode())
            self._audio_proc.stdin.flush()
            self._audio_proc.stdout.readline()
        except Exception as exc:
            print(f"[VoiceNarrator] _play_mp3 error: {exc}")

    def _worker(self) -> None:
        pyttsx3_engine = None
        if self._init_edge():
            pass
        else:
            pyttsx3_engine = self._init_pyttsx3()
            if pyttsx3_engine is None:
                self._mode = "none"
                print("[VoiceNarrator] Sin motor TTS. Voz desactivada.")
                self._ready_event.set()
                return

        self._ready_event.set()

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                if self._mode == "edge_tts":
                    self._speak_edge(text)
                else:
                    pyttsx3_engine.say(text)
                    pyttsx3_engine.runAndWait()
            except Exception as exc:
                print(f"[VoiceNarrator] Error: {exc}")

    def _init_edge(self) -> bool:
        try:
            import edge_tts  # noqa
            self._speak_edge("Sistema listo.", play=False)
            self._mode = "edge_tts"
            print(f"[VoiceNarrator] edge-tts listo | {EDGE_VOICE}")
            return True
        except Exception as exc:
            print(f"[VoiceNarrator] edge-tts no disponible: {exc}")
        return False

    def _init_pyttsx3(self) -> Optional[object]:
        try:
            import pyttsx3
            eng = pyttsx3.init()
            for v in eng.getProperty("voices"):
                if "es" in v.id.lower() or "spanish" in v.name.lower():
                    eng.setProperty("voice", v.id)
                    break
            eng.setProperty("rate", 170)
            self._mode = "pyttsx3"
            print("[VoiceNarrator] pyttsx3 listo (fallback offline)")
            return eng
        except Exception as exc:
            print(f"[VoiceNarrator] pyttsx3 falló: {exc}")
        return None

    def _speak_edge(self, text: str, play: bool = True) -> None:
        import edge_tts
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    edge_tts.Communicate(text, EDGE_VOICE).save(tmp_path)
                )
            finally:
                loop.close()
            if play:
                self._play_mp3(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
