"""
TEST 03 — Pipeline completo con video (Detección + Análisis + Voz en tiempo real)
==================================================================================
Este es el test más completo: simula exactamente lo que verá el profesor.
Procesa un video frame a frame, LUCY habla cuando el riesgo cambia o se mantiene
alto, y la narrativa aparece subtitulada en la pantalla.

Controles:
    SPACE   →  pausar / continuar
    R       →  reiniciar desde el inicio
    Q / ESC →  salir

Uso:
    python scripts/test_03_video.py
    python scripts/test_03_video.py --video test_videos/video1.mp4
    python scripts/test_03_video.py --video test_videos/video1.mp4 --no-voice
    python scripts/test_03_video.py --video test_videos/video1.mp4 --step
       (--step procesa frame a frame con SPACE, útil para demo pausada)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import numpy as np
import time
from collections import deque

from engines.rtdetr_detector     import RTDetrDetector, TRAFFIC_CLASS_IDS
from engines.deep_traffic_engine import DeepTrafficEngine, RISK_COLORS, TL_COLORS_BGR, SEQ_LEN
from modules.voice_narrator      import VoiceNarrator, build_narrative

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_VIDEO  = "test_videos/video1.mp4"
WINDOW_NAME    = "LUCY — Test Pipeline Video"
FONT           = cv2.FONT_HERSHEY_SIMPLEX
PROCESS_EVERY  = 3          # analizar 1 de cada N frames (velocidad vs precisión)
MAX_WIDTH      = 1280       # redimensionar si el video es muy grande

LEVEL_LABELS   = {"SAFE": "SEGURO", "WARNING": "PRECAUCION", "CRITICAL": "PELIGRO"}
LEVEL_COLORS   = {"SAFE": (0,200,0), "WARNING": (0,165,255), "CRITICAL": (0,0,255)}

# ── Dibujo ─────────────────────────────────────────────────────────────────────

def _wrap(text: str, max_chars: int = 70) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


def _draw_frame(frame: np.ndarray, dets: list, result: dict,
                narrative: str, fps: float, frame_n: int, total: int) -> np.ndarray:
    out   = frame.copy()
    h, w  = out.shape[:2]
    level = result["risk_level"]
    color = LEVEL_COLORS.get(level, (0,200,0))

    # Bounding boxes
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        c     = (0,0,255) if det["class_id"] == 0 else color
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if det["class_id"] == 9 and det.get("tl_state"):
            tl_c = TL_COLORS_BGR.get(det["tl_state"], (128,128,128))
            cv2.circle(out, ((x1+x2)//2, max(y1-12,12)), 10, tl_c, -1)
            label = f"TL:{det['tl_state']}"
        cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.46, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), c, -1)
        cv2.putText(out, label, (x1+2, y1-4), FONT, 0.46, (255,255,255), 1, cv2.LINE_AA)

    # Banner superior
    cv2.rectangle(out, (0,0), (w, 62), color, -1)
    conf  = result.get("confidence", 0)
    title = f"LUCY: {LEVEL_LABELS.get(level, level)}  ({conf*100:.0f}%)"
    if result.get("tl_override"):
        title += "  [TL OVERRIDE]"
    cv2.putText(out, title, (10, 26), FONT, 0.78, (255,255,255), 2, cv2.LINE_AA)

    probs = result.get("all_probs", {})
    ptxt  = "  |  ".join(f"{k}: {v:.2f}" for k, v in probs.items())
    buf_r = result.get("buffer_ready", False)
    fi    = result.get("frames_in", 0)
    buf_t = f"GRU:{fi}/{SEQ_LEN}" if buf_r else f"warm:{fi}/{SEQ_LEN}"
    cv2.putText(out, f"{ptxt}   {buf_t}", (10, 52), FONT, 0.38, (255,255,255), 1, cv2.LINE_AA)

    # Info esquina superior derecha
    info = f"FPS:{fps:.1f}  frame:{frame_n}/{total}" if total > 0 else f"FPS:{fps:.1f}"
    (iw, _), _ = cv2.getTextSize(info, FONT, 0.38, 1)
    cv2.putText(out, info, (w-iw-8, 52), FONT, 0.38, (255,255,255), 1, cv2.LINE_AA)

    # Subtítulo / narrativa
    lines    = _wrap(narrative, max_chars=int(w / 9.5))
    box_h    = 28 + len(lines) * 26
    y_top    = h - box_h - 8
    overlay  = out.copy()
    cv2.rectangle(overlay, (0, y_top), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)
    cv2.putText(out, "LUCY dice:", (12, y_top+20), FONT, 0.44, (160,160,160), 1, cv2.LINE_AA)
    for j, line in enumerate(lines):
        cv2.putText(out, line, (12, y_top+42+j*26), FONT, 0.60, (255,255,255), 1, cv2.LINE_AA)

    return out


# ── Estadísticas de sesión ──────────────────────────────────────────────────────

class SessionStats:
    def __init__(self):
        self.counts     = {"SAFE": 0, "WARNING": 0, "CRITICAL": 0}
        self.fps_buf    = deque(maxlen=30)
        self.narratives = []

    def update(self, level: str, fps: float, narrative: str):
        self.counts[level] = self.counts.get(level, 0) + 1
        self.fps_buf.append(fps)
        if not self.narratives or self.narratives[-1] != narrative:
            self.narratives.append(narrative)

    def print_summary(self):
        total = sum(self.counts.values()) or 1
        print("\n" + "=" * 60)
        print("  LUCY — Resumen de sesión")
        print("=" * 60)
        for lvl, cnt in self.counts.items():
            pct = cnt / total * 100
            bar = "█" * int(pct / 5)
            print(f"  {lvl:<10} {cnt:>5} frames  ({pct:5.1f}%)  {bar}")
        avg_fps = sum(self.fps_buf) / len(self.fps_buf) if self.fps_buf else 0
        print(f"\n  FPS promedio : {avg_fps:.1f}")
        print(f"  Narrativas   : {len(self.narratives)} diferentes")
        print("\n  Narrativas generadas:")
        for i, n in enumerate(self.narratives, 1):
            print(f"    {i:02d}. {n}")
        print("=" * 60 + "\n")


# ── Main ────────────────────────────────────────────────────────────────────────

def run(video_path: str, no_voice: bool = False, step_mode: bool = False):
    if not os.path.exists(video_path):
        print(f"[!] Video no encontrado: {video_path}")
        sys.exit(1)

    print(f"\n[Test 03] Video       : {video_path}")
    print(f"[Test 03] Modo paso   : {'sí' if step_mode else 'no'}")
    print(f"[Test 03] Voz         : {'desactivada' if no_voice else 'activa'}")
    print("[Test 03] Cargando modelos...\n")

    detector = RTDetrDetector(
        model_path       = "rtdetr-l.onnx",
        conf_threshold   = 0.35,
        target_class_ids = TRAFFIC_CLASS_IDS,
    )
    engine   = DeepTrafficEngine()
    narrator = None if no_voice else VoiceNarrator()
    stats    = SessionStats()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] No se pudo abrir: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    print(f"[Test 03] {total_frames} frames @ {video_fps:.1f} fps")
    print("[Test 03] SPACE=pausa  R=reinicio  Q/ESC=salir\n")

    last_result    = None
    last_narrative = "Inicializando LUCY..."
    frame_count    = 0
    paused         = False
    t_prev         = time.time()

    while True:
        if not paused or step_mode:
            ret, frame = cap.read()
            if not ret:
                print("[Test 03] Fin del video.")
                break

            frame_count += 1

            # Redimensionar si es muy ancho
            fh, fw = frame.shape[:2]
            if fw > MAX_WIDTH:
                scale = MAX_WIDTH / fw
                frame = cv2.resize(frame, (MAX_WIDTH, int(fh * scale)))

            # FPS
            now   = time.time()
            fps   = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            # Análisis cada PROCESS_EVERY frames
            if frame_count % PROCESS_EVERY == 0 or last_result is None:
                dets         = detector.detect(frame)
                last_result  = engine.analyze(dets, frame.shape, image=frame, session_id="video_test")
                last_narrative = build_narrative(last_result)
                if narrator:
                    narrator.speak(last_result)

            stats.update(last_result["risk_level"], fps, last_narrative)

            display = _draw_frame(
                frame, dets if frame_count % PROCESS_EVERY == 0 else [],
                last_result, last_narrative, fps, frame_count, total_frames
            )

            # Indicador de pausa / paso
            if paused or step_mode:
                cv2.putText(display, "PAUSADO — SPACE para continuar",
                            (10, display.shape[0]//2),
                            FONT, 0.9, (0,0,200), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

        wait_ms = 0 if (paused or step_mode) else max(1, int(1000/video_fps) - 5)
        key     = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            if step_mode and paused:
                paused = False  # en step_mode SPACE siempre avanza
        elif key in (ord("r"), ord("R")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            engine.reset("video_test")
            frame_count = 0
            last_result = None
            last_narrative = "Reiniciando..."
            print("[Test 03] Video reiniciado.")

    cap.release()
    cv2.destroyAllWindows()
    stats.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help=f"Ruta al video (default: {DEFAULT_VIDEO})")
    parser.add_argument("--no-voice", action="store_true",
                        help="Desactivar voz")
    parser.add_argument("--step", action="store_true",
                        help="Modo paso a paso (SPACE para avanzar frame)")
    args = parser.parse_args()
    run(args.video, no_voice=args.no_voice, step_mode=args.step)
