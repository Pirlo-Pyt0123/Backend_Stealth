from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
import time
import io
import tempfile
import os
from typing import Dict

from rtdetr_detector import RTDetrDetector, TRAFFIC_CLASS_IDS
from traffic_risk_engine import TrafficRiskEngine
from suspicious_behavior_engine import SuspiciousBehaviorEngine, SECURITY_CLASSES

app = FastAPI(
    title="StealthVision — Security & Vial",
    description="Detección de riesgo vial y comportamiento sospechoso en tiempo real",
    version="3.0.0",
)

# --- Detectores ---
# Detector legacy (solo personas) — mantiene compatibilidad con UE anterior
detector = RTDetrDetector(
    model_path="rtdetr-l.onnx",
    conf_threshold=0.5,
    person_class_id=0,
)

# Detector de tráfico completo
traffic_detector = RTDetrDetector(
    model_path="rtdetr-l.onnx",
    conf_threshold=0.4,
    target_class_ids=TRAFFIC_CLASS_IDS,
)

# Detector de seguridad — personas + armas + objetos sospechosos
security_detector = RTDetrDetector(
    model_path="rtdetr-l.onnx",
    conf_threshold=0.4,
    target_class_ids=SECURITY_CLASSES,
)

# Motor de riesgo educativo (tráfico)
risk_engine = TrafficRiskEngine()

# Motores de seguridad — uno por sesión de video para mantener estado temporal
# session_id → SuspiciousBehaviorEngine
_security_engines: Dict[str, SuspiciousBehaviorEngine] = {}


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _load_image(data: bytes) -> np.ndarray:
    """Carga imagen desde bytes (JPEG/PNG/EXR). Lanza HTTPException si falla."""
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        suffix = ".exr"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
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


def _get_security_engine(session_id: str) -> SuspiciousBehaviorEngine:
    """Devuelve (o crea) el motor de seguridad para la sesión indicada."""
    if session_id not in _security_engines:
        _security_engines[session_id] = SuspiciousBehaviorEngine()
    return _security_engines[session_id]


RISK_COLORS = {
    "CRITICAL": (0, 0, 255),
    "WARNING":  (0, 165, 255),
    "SAFE":     (0, 200, 0),
}

RISK_LABELS_ES = {
    "CRITICAL": "PELIGRO",
    "WARNING":  "PRECAUCION",
    "SAFE":     "SEGURO",
}


def _draw_detections(image: np.ndarray, detections, risk_result: dict) -> np.ndarray:
    """Dibuja bboxes y overlay de riesgo sobre la imagen."""
    out = image.copy()
    color = risk_result.get("color_alerta", (0, 200, 0))
    h, w = out.shape[:2]

    # Bboxes de cada detección
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Banner de riesgo en la parte superior
    level = risk_result["risk_level"]
    banner_h = 60
    cv2.rectangle(out, (0, 0), (w, banner_h), color, -1)
    nivel_txt = RISK_LABELS_ES.get(level, level)
    titulo    = risk_result.get("titulo", "")
    cv2.putText(out, f"[{nivel_txt}] {titulo}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    consejo = risk_result.get("consejo_rapido", "")
    cv2.putText(out, consejo, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Indicador de iluminación (esquina superior derecha)
    lighting = risk_result.get("stats", {}).get("iluminacion")
    if lighting and lighting["is_dark"]:
        nivel_luz = lighting["nivel_luz"]
        brightness = lighting["brightness"]
        luz_txt = f"LUZ BAJA ({nivel_luz}) {brightness:.0f}/255"
        (tw, _), _ = cv2.getTextSize(luz_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (w - tw - 12, 0), (w, 22), (20, 20, 100), -1)
        cv2.putText(out, luz_txt, (w - tw - 8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    return out


SECURITY_LEVEL_COLORS = {
    "CRITICAL": (0, 0, 255),
    "WARNING":  (0, 140, 255),
    "SAFE":     (0, 200, 0),
}

SECURITY_LEVEL_LABELS = {
    "CRITICAL": "ALERTA",
    "WARNING":  "SOSPECHOSO",
    "SAFE":     "NORMAL",
}


def _draw_security(image: np.ndarray, detections, result: dict) -> np.ndarray:
    """Dibuja bboxes, tracks y banner de alerta de seguridad."""
    out   = image.copy()
    color = result.get("color_alerta", (0, 200, 0))
    h, w  = out.shape[:2]
    level = result["risk_level"]

    # Bboxes de detecciones crudas
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class_name"]
        # Armas en rojo aunque el nivel sea bajo
        det_color = (0, 0, 255) if cls in ("knife", "scissors") else color
        cv2.rectangle(out, (x1, y1), (x2, y2), det_color, 2)
        label = f"{cls} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), det_color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # IDs de tracks sobre personas
    for track in result.get("tracks", []):
        x1, y1, x2, y2 = track["bbox"]
        tid  = track["track_id"]
        age  = track["age_seconds"]
        loit = track["loitering"]
        t_color = (0, 0, 255) if loit else (200, 200, 0)
        label = f"ID:{tid} {age:.0f}s" + (" [MERODEANDO]" if loit else "")
        cv2.putText(out, label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_color, 1)

    # Banner superior
    banner_h = 65
    cv2.rectangle(out, (0, 0), (w, banner_h), color, -1)
    nivel_txt = SECURITY_LEVEL_LABELS.get(level, level)
    titulo    = result.get("titulo", "")
    cv2.putText(out, f"[{nivel_txt}] {titulo}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    consejo = result.get("consejo", "")
    cv2.putText(out, consejo, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)

    # Indicador de iluminación
    lighting = result.get("stats", {}).get("iluminacion")
    if lighting and lighting["is_dark"]:
        luz_txt = f"ZONA OSCURA ({lighting['nivel_luz']}) {lighting['brightness']:.0f}/255"
        (tw, _), _ = cv2.getTextSize(luz_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (w - tw - 12, 0), (w, 20), (20, 20, 100), -1)
        cv2.putText(out, luz_txt, (w - tw - 8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

    # Contador de personas + tracks activos (esquina inferior derecha)
    info = f"Personas: {result['stats']['personas']} | Tracks: {result['stats']['tracks_activos']}"
    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(out, (w - tw - 10, h - th - 14), (w, h), (40, 40, 40), -1)
    cv2.putText(out, info, (w - tw - 6, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "version":    "3.0.0",
        "model":      traffic_detector.model_path,
        "providers":  traffic_detector.session.get_providers(),
        "input_size": f"{traffic_detector.in_w}x{traffic_detector.in_h}",
        "modes": {
            "traffic":  "POST /analyze | /analyze_visual",
            "security": "POST /security | /security_visual (con ?session_id=)",
        },
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Endpoint legacy — detecta solo personas.
    Mantiene compatibilidad con la integración de Unreal Engine anterior.
    """
    try:
        data  = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        image = _load_image(data)
        t0    = time.time()
        dets  = detector.detect(image)
        dt    = (time.time() - t0) * 1000.0

        return JSONResponse({
            "detected":       len(dets) > 0,
            "count":          len(dets),
            "detections":     dets,
            "inference_time_ms": round(dt, 2),
            "image_shape":    {"height": image.shape[0], "width": image.shape[1]},
            "content_type":   file.content_type,
        })

    except HTTPException:
        raise
    except Exception as e:
        print("Backend error /detect:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Endpoint principal — detección vial + análisis de riesgo educativo.
    Devuelve objetos detectados + nivel de peligro + mensaje pedagógico.
    Diseñado para integración con Unreal Engine / simulador educativo.
    """
    try:
        data  = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        image = _load_image(data)
        t0    = time.time()
        dets  = traffic_detector.detect(image)
        risk  = risk_engine.analyze(dets, image.shape, image=image)
        dt    = (time.time() - t0) * 1000.0

        return JSONResponse({
            "risk_level":      risk["risk_level"],
            "scenario":        risk["scenario_key"],
            "titulo":          risk["titulo"],
            "explicacion":     risk["explicacion"],
            "leccion":         risk["leccion"],
            "consejo_rapido":  risk["consejo_rapido"],
            "stats":           risk["stats"],
            "detecciones_clave": risk["detecciones_clave"],
            "todas_detecciones": dets,
            "inference_time_ms": round(dt, 2),
            "image_shape":    {"height": image.shape[0], "width": image.shape[1]},
        })

    except HTTPException:
        raise
    except Exception as e:
        print("Backend error /analyze:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_visual")
async def analyze_visual(file: UploadFile = File(...)):
    """
    Igual que /analyze pero devuelve la imagen anotada con bboxes y banner de riesgo.
    Útil para visualización directa en Unreal Engine o demos.
    """
    try:
        data  = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        image = _load_image(data)
        dets  = traffic_detector.detect(image)
        risk  = risk_engine.analyze(dets, image.shape, image=image)
        out   = _draw_detections(image, dets, risk)

        _, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        print("Backend error /analyze_visual:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoints de SEGURIDAD (detección de comportamiento sospechoso en tiempo real)
# ---------------------------------------------------------------------------

@app.post("/security")
async def security_analyze(
    file: UploadFile = File(...),
    session_id: str = Query(default="default", description="ID de sesión de video. "
                            "Usar el mismo ID entre frames para mantener tracking temporal."),
    timestamp: float = Query(default=None, description="Timestamp UNIX del frame. "
                             "Si se omite, usa el tiempo del servidor."),
):
    """
    Endpoint principal de seguridad — análisis en tiempo real.

    Enviar frames consecutivos con el mismo `session_id` para activar:
    - Tracking de personas entre frames (quién es quién)
    - Detección de merodeo (persona quieta >5 segundos)
    - Detección de encercamiento, confrontación, armas

    Diseñado para integración con Unreal Engine en tiempo real.
    """
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        image  = _load_image(data)
        ts     = timestamp if timestamp is not None else time.time()
        t0     = time.time()
        dets   = security_detector.detect(image)
        engine = _get_security_engine(session_id)
        result = engine.analyze(dets, image.shape, image=image, timestamp=ts,
                                session_id=session_id)
        dt = (time.time() - t0) * 1000.0

        return JSONResponse({
            "risk_level":    result["risk_level"],
            "scenario_key":  result["scenario_key"],
            "titulo":        result["titulo"],
            "descripcion":   result["descripcion"],
            "accion":        result["accion"],
            "consejo":       result["consejo"],
            "involucrados":  result["involucrados"],
            "tracks":        result["tracks"],
            "stats":         result["stats"],
            "todas_detecciones": dets,
            "session_id":    session_id,
            "inference_time_ms": round(dt, 2),
            "image_shape":   {"height": image.shape[0], "width": image.shape[1]},
        })

    except HTTPException:
        raise
    except Exception as e:
        print("Backend error /security:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security_visual")
async def security_visual(
    file: UploadFile = File(...),
    session_id: str = Query(default="default"),
    timestamp: float = Query(default=None),
):
    """
    Igual que /security pero devuelve la imagen anotada con bboxes,
    track IDs, indicadores de merodeo y banner de alerta.
    Útil para visualización directa en Unreal Engine / monitor de seguridad.
    """
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        image  = _load_image(data)
        ts     = timestamp if timestamp is not None else time.time()
        dets   = security_detector.detect(image)
        engine = _get_security_engine(session_id)
        result = engine.analyze(dets, image.shape, image=image, timestamp=ts,
                                session_id=session_id)
        out    = _draw_security(image, dets, result)

        _, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        print("Backend error /security_visual:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security/reset")
async def security_reset(
    session_id: str = Query(default="default"),
):
    """
    Reinicia el estado temporal del motor de seguridad para una sesión.
    Llamar al inicio de cada nueva grabación o escena en Unreal Engine.
    """
    if session_id in _security_engines:
        _security_engines[session_id].reset()
    return {"status": "ok", "session_id": session_id, "message": "Estado reiniciado"}


@app.get("/security/sessions")
async def security_sessions():
    """Lista las sesiones de seguridad activas y su estado."""
    sessions = {}
    for sid, engine in _security_engines.items():
        tracks = engine.tracker._tracks
        sessions[sid] = {
            "frames_procesados": engine._frame_count,
            "tracks_activos":    len(tracks),
            "track_ids":         list(tracks.keys()),
        }
    return sessions


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
