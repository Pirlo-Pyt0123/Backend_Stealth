"""
LUCY — Servidor principal StealthVision
Reemplaza los motores de reglas por IA real:
  - DeepSecurityEngine  (BehaviorLSTM + PPO)
  - DeepTrafficEngine   (SpatialRiskTransformer + PPO)

Los endpoints mantienen la misma firma que antes
para que el Blueprint de UE5 no necesite cambios.

Uso:
    cd c:/Users/LENOVO/Documents/Stealth
    python main.py
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
import time
import io
import tempfile
import os
from typing import Dict

from engines.rtdetr_detector   import RTDetrDetector, TRAFFIC_CLASS_IDS
from engines.deep_traffic_engine   import DeepTrafficEngine
from engines.deep_security_engine  import DeepSecurityEngine

app = FastAPI(
    title="LUCY — StealthVision AI",
    description="IA real: BehaviorLSTM + SpatialRiskTransformer + PPO Agent",
    version="4.0.0",
)

# ── Detectores RT-DETR ────────────────────────────────────────────────────────
SECURITY_CLASS_IDS = {0, 43, 76}   # person, knife, scissors

traffic_detector = RTDetrDetector(
    model_path       = "rtdetr-l.onnx",
    conf_threshold   = 0.4,
    target_class_ids = TRAFFIC_CLASS_IDS,   # person, bike, car, moto, bus, truck, tl, stop
)

security_detector = RTDetrDetector(
    model_path       = "rtdetr-l.onnx",
    conf_threshold   = 0.4,
    target_class_ids = SECURITY_CLASS_IDS,
)

# ── Motores de IA (cargan modelos .pth al iniciar) ────────────────────────────
traffic_engine  = DeepTrafficEngine()
_security_engines: Dict[str, DeepSecurityEngine] = {}   # session_id → engine


def _get_security_engine(session_id: str) -> DeepSecurityEngine:
    if session_id not in _security_engines:
        _security_engines[session_id] = DeepSecurityEngine()
    return _security_engines[session_id]


# ── Utilidades ────────────────────────────────────────────────────────────────
def _load_image(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            image = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if image is None:
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

    return image


RISK_COLORS_BGR = {
    "SAFE":     (0, 200, 0),
    "WARNING":  (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

LEVEL_LABELS = {
    "SAFE":     "SEGURO",
    "WARNING":  "PRECAUCION",
    "CRITICAL": "PELIGRO",
}


def _draw_traffic(image: np.ndarray, detections, result: dict) -> np.ndarray:
    out   = image.copy()
    level = result["risk_level"]
    color = RISK_COLORS_BGR.get(level, (0, 200, 0))
    h, w  = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    banner_h = 60
    cv2.rectangle(out, (0, 0), (w, banner_h), color, -1)
    cv2.putText(out,
                f"[{LEVEL_LABELS[level]}] {result.get('titulo','')}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out,
                result.get("mensaje", ""),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)

    if result.get("is_dark"):
        txt = f"LUZ BAJA  {result['brightness']:.0f}/255"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (w-tw-12, 0), (w, 20), (20, 20, 100), -1)
        cv2.putText(out, txt, (w-tw-8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
    return out


def _draw_security(image: np.ndarray, detections, result: dict) -> np.ndarray:
    out   = image.copy()
    level = result["risk_level"]
    color = RISK_COLORS_BGR.get(level, (0, 200, 0))
    h, w  = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det_color = (0, 0, 255) if det["class_name"] in ("knife", "scissors") else color
        cv2.rectangle(out, (x1, y1), (x2, y2), det_color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), det_color, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    banner_h = 65
    cv2.rectangle(out, (0, 0), (w, banner_h), color, -1)
    behavior = result.get("behavior", "normal")
    cv2.putText(out,
                f"[{LEVEL_LABELS[level]}] LUCY: {behavior.upper()}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    conf   = result.get("behavior_conf", 0.0)
    buf    = result.get("buffer_ready", False)
    status = "LSTM activo" if buf else f"Calentando {result.get('frame', 0)}/30 frames"
    info   = f"Confianza: {conf*100:.0f}%  |  {status}"
    cv2.putText(out, info, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255,255,255), 1, cv2.LINE_AA)

    if result.get("is_dark"):
        txt = f"ZONA OSCURA  {result['brightness']:.0f}/255"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (w-tw-12, 0), (w, 20), (20, 20, 100), -1)
        cv2.putText(out, txt, (w-tw-8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,255), 1, cv2.LINE_AA)

    persons  = result.get("persons_found", 0)
    weapons  = result.get("weapons_found", 0)
    info_br  = f"Personas: {persons}  |  Armas: {weapons}  |  Frame: {result.get('frame',0)}"
    (tw, th), _ = cv2.getTextSize(info_br, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(out, (w-tw-10, h-th-14), (w, h), (40,40,40), -1)
    cv2.putText(out, info_br, (w-tw-6, h-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv2.LINE_AA)
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": "4.0.0 — LUCY AI",
        "models": {
            "traffic":  "SpatialRiskTransformer + PPO",
            "security": "BehaviorLSTM + PPO",
            "detector": "RT-DETR ONNX",
        },
        "endpoints": {
            "traffic":  "POST /analyze | /analyze_visual",
            "security": "POST /security | /security_visual",
        },
    }


# ── Tráfico ───────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Riesgo vial desde perspectiva peatón.
    SpatialRiskTransformer + PPO Agent.
    Mismo endpoint que antes — UE5 no necesita cambios.
    """
    try:
        image = _load_image(await file.read())
        t0    = time.time()
        dets  = traffic_detector.detect(image)
        result = traffic_engine.analyze(dets, image.shape, image=image)
        dt    = (time.time() - t0) * 1000.0

        return JSONResponse({
            # Campos que UE5 ya conoce
            "risk_level":      result["risk_level"],
            "titulo":          result.get("titulo", ""),
            "mensaje":         result.get("mensaje", ""),
            "leccion":         result.get("leccion", ""),
            "color_alerta":    result.get("color_alerta", [0,200,0]),
            # Campos nuevos de LUCY IA
            "lucy_action":     result.get("rl_action"),
            "transformer_conf":result.get("transformer_conf", 0.0),
            "attention":       result.get("attention", []),
            "is_dark":         result.get("is_dark", False),
            "brightness":      result.get("brightness", 128.0),
            "todas_detecciones": dets,
            "inference_time_ms": round(dt, 2),
            "image_shape": {"height": image.shape[0], "width": image.shape[1]},
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_visual")
async def analyze_visual(file: UploadFile = File(...)):
    """Igual que /analyze pero retorna imagen anotada con bboxes y banner."""
    try:
        image  = _load_image(await file.read())
        dets   = traffic_detector.detect(image)
        result = traffic_engine.analyze(dets, image.shape, image=image)
        out    = _draw_traffic(image, dets, result)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Seguridad ─────────────────────────────────────────────────────────────────

@app.post("/security")
async def security_analyze(
    file:       UploadFile = File(...),
    session_id: str        = Query(default="default"),
    timestamp:  float      = Query(default=None),
):
    """
    Seguridad ciudadana en tiempo real.
    BehaviorLSTM (30 frames) + PPO Agent.
    Enviar frames consecutivos con el mismo session_id.
    """
    try:
        image  = _load_image(await file.read())
        ts     = timestamp if timestamp is not None else time.time()
        t0     = time.time()
        dets   = security_detector.detect(image)
        engine = _get_security_engine(session_id)
        result = engine.analyze(dets, image.shape, image=image, timestamp=ts)
        dt     = (time.time() - t0) * 1000.0

        return JSONResponse({
            # Campos que UE5 ya conoce
            "risk_level":    result["risk_level"],
            "color_alerta":  list(result["color_alerta"]),
            # Campos nuevos de LUCY IA
            "behavior":      result["behavior"],
            "behavior_conf": round(result["behavior_conf"], 3),
            "all_behaviors": result["all_behaviors"],
            "rl_action":     result["rl_action"],
            "weapons_found": result["weapons_found"],
            "persons_found": result["persons_found"],
            "brightness":    result["brightness"],
            "is_dark":       result["is_dark"],
            "buffer_ready":  result["buffer_ready"],
            "frame":         result["frame"],
            "session_id":    session_id,
            "todas_detecciones": dets,
            "inference_time_ms": round(dt, 2),
            "image_shape":   {"height": image.shape[0], "width": image.shape[1]},
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security_visual")
async def security_visual(
    file:       UploadFile = File(...),
    session_id: str        = Query(default="default"),
    timestamp:  float      = Query(default=None),
):
    """Igual que /security pero retorna imagen anotada."""
    try:
        image  = _load_image(await file.read())
        ts     = timestamp if timestamp is not None else time.time()
        dets   = security_detector.detect(image)
        engine = _get_security_engine(session_id)
        result = engine.analyze(dets, image.shape, image=image, timestamp=ts)
        out    = _draw_security(image, dets, result)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security/reset")
async def security_reset(session_id: str = Query(default="default")):
    """Reinicia el buffer temporal de LUCY para una sesión (nueva escena en UE5)."""
    if session_id in _security_engines:
        _security_engines[session_id].reset()
    return {"status": "ok", "session_id": session_id}


@app.get("/security/sessions")
async def security_sessions():
    """Lista sesiones activas y su estado."""
    return {
        sid: {
            "frames_procesados": eng._frame_count,
            "buffer_ready":      len(eng._seq_buffer) >= 30,
            "buffer_frames":     len(eng._seq_buffer),
        }
        for sid, eng in _security_engines.items()
    }


# ── Legacy — mantiene compatibilidad con Blueprint antiguo ────────────────────
@app.post("/detect")
async def detect_legacy(file: UploadFile = File(...)):
    """Endpoint legacy — solo personas. Mantiene compatibilidad con UE5 anterior."""
    try:
        image = _load_image(await file.read())
        dets  = security_detector.detect(image)
        persons = [d for d in dets if d["class_id"] == 0]
        return JSONResponse({
            "detected":   len(persons) > 0,
            "count":      len(persons),
            "detections": persons,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
