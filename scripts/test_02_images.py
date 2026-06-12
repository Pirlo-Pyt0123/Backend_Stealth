"""
TEST 02 — Pipeline completo con imágenes (Detección + Análisis + Voz)
=====================================================================
Procesa cada imagen en test_images/ con el detector y el engine de tráfico,
muestra el resultado en pantalla y habla la narrativa contextual.

Controles:
    SPACE / Enter  →  siguiente imagen
    Q              →  salir

Uso:
    python scripts/test_02_images.py
    python scripts/test_02_images.py --images test_images/calle1.jpg test_images/calle2.jpg
    python scripts/test_02_images.py --no-voice   (solo visual, sin audio)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import cv2
import numpy as np
import time

from engines.rtdetr_detector     import RTDetrDetector, TRAFFIC_CLASS_IDS
from engines.deep_traffic_engine import DeepTrafficEngine, RISK_COLORS, TL_COLORS_BGR
from modules.voice_narrator      import VoiceNarrator, build_narrative

# ── Configuración ──────────────────────────────────────────────────────────────

IMAGES_DIR   = "test_images"
WINDOW_NAME  = "LUCY — Test Pipeline Imágenes"
FONT         = cv2.FONT_HERSHEY_SIMPLEX
LEVEL_LABELS = {"SAFE": "SEGURO", "WARNING": "PRECAUCION", "CRITICAL": "PELIGRO"}


def _draw_result(image: np.ndarray, detections: list, result: dict, narrative: str) -> np.ndarray:
    out   = image.copy()
    h, w  = out.shape[:2]
    level = result["risk_level"]
    color = tuple(RISK_COLORS.get(level, (0, 200, 0)))

    # Bounding boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        c     = (0, 0, 255) if det["class_id"] == 0 else color
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if det["class_id"] == 9 and det.get("tl_state"):
            tl_c  = TL_COLORS_BGR.get(det["tl_state"], (128, 128, 128))
            cv2.circle(out, ((x1+x2)//2, max(y1-12, 12)), 10, tl_c, -1)
            label = f"TL:{det['tl_state']}"
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.48, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), c, -1)
        cv2.putText(out, label, (x1+2, y1-4), FONT, 0.48, (255,255,255), 1, cv2.LINE_AA)

    # Banner superior
    cv2.rectangle(out, (0, 0), (w, 60), color, -1)
    conf  = result.get("confidence", 0.0)
    label = f"LUCY: {LEVEL_LABELS.get(level, level)}  ({conf*100:.0f}%)"
    if result.get("tl_override"):
        label += "  [TL OVERRIDE]"
    cv2.putText(out, label, (10, 24), FONT, 0.75, (255,255,255), 2, cv2.LINE_AA)
    probs = result.get("all_probs", {})
    ptxt  = "  |  ".join(f"{k}: {v:.2f}" for k, v in probs.items())
    cv2.putText(out, ptxt, (10, 50), FONT, 0.38, (255,255,255), 1, cv2.LINE_AA)

    # Caja de narrativa en la parte inferior
    lines      = _wrap_text(narrative, max_chars=72)
    box_h      = 28 + len(lines) * 26
    y_box_top  = h - box_h - 8
    overlay    = out.copy()
    cv2.rectangle(overlay, (0, y_box_top), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
    cv2.putText(out, "LUCY dice:", (12, y_box_top + 20),
                FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    for j, line in enumerate(lines):
        cv2.putText(out, line, (12, y_box_top + 42 + j * 26),
                    FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def _wrap_text(text: str, max_chars: int = 70) -> list:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def run(image_paths: list, no_voice: bool = False):
    print("\n[Test 02] Inicializando detector y engine...")
    detector = RTDetrDetector(
        model_path       = "rtdetr-l.onnx",
        conf_threshold   = 0.35,
        target_class_ids = TRAFFIC_CLASS_IDS,
    )
    engine = DeepTrafficEngine()
    narrator = None if no_voice else VoiceNarrator()

    print(f"[Test 02] {len(image_paths)} imágenes a procesar.")
    print("[Test 02] SPACE / Enter = siguiente | Q = salir\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    for idx, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"  [!] No encontrado: {path}")
            continue

        image = cv2.imread(path)
        if image is None:
            print(f"  [!] No se pudo leer: {path}")
            continue

        t0   = time.time()
        dets = detector.detect(image)
        res  = engine.analyze(dets, image.shape, image=image, session_id="test")
        dt   = (time.time() - t0) * 1000

        narrative = build_narrative(res)
        frame     = _draw_result(image, dets, res, narrative)

        # Consola
        fname = os.path.basename(path)
        print(f"[{idx+1:02d}/{len(image_paths):02d}] {fname}")
        print(f"         Nivel    : {res['risk_level']}  ({res['confidence']:.0%})")
        print(f"         Objetos  : personas={res['stats']['personas']}  "
              f"vehículos={res['stats']['vehiculos']}  "
              f"bicis={res['stats']['bicicletas']}")
        print(f"         LUCY dice: \"{narrative}\"")
        print(f"         Tiempo   : {dt:.1f} ms\n")

        if narrator:
            narrator.speak(res)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()
    print("[Test 02] Finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=None,
                        help="Imágenes específicas a procesar")
    parser.add_argument("--no-voice", action="store_true",
                        help="Desactivar voz (solo visual)")
    args = parser.parse_args()

    if args.images:
        paths = args.images
    else:
        exts  = ("*.jpg", "*.jpeg", "*.png")
        paths = []
        for ext in exts:
            paths += sorted(glob.glob(os.path.join(IMAGES_DIR, ext)))

    if not paths:
        print(f"[!] No se encontraron imágenes en {IMAGES_DIR}/")
        sys.exit(1)

    run(paths, no_voice=args.no_voice)
