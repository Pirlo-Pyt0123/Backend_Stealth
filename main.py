"""
LUCY -- StealthVision AI Server
Road safety from a pedestrian perspective.

TrafficRiskNet (SpatialRiskTransformer + BiGRU) trained on JAAD + synthetic data.
Session-aware: each UE5 client keeps its own GRU frame buffer via session_id.

Run:
    python main.py
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
import base64
import time
import io
import tempfile
import os

from engines.rtdetr_detector     import RTDetrDetector, TRAFFIC_CLASS_IDS
from engines.deep_traffic_engine import DeepTrafficEngine, RISK_COLORS, TL_COLORS_BGR, SEQ_LEN
from engines.fight_engine        import FightEngine
from modules.voice_narrator      import VoiceNarrator, build_narrative

app = FastAPI(
    title="LUCY -- StealthVision AI",
    description="TrafficRiskNet: SpatialRiskTransformer + BiGRU | JAAD + synthetic training",
    version="5.0.0",
)

# ── Detector + Engine ─────────────────────────────────────────────────────────
detector = RTDetrDetector(
    model_path       = "rtdetr-l.onnx",
    conf_threshold   = 0.35,
    target_class_ids = TRAFFIC_CLASS_IDS,
)

engine       = DeepTrafficEngine()
fight_engine = FightEngine()
narrator     = VoiceNarrator()
narrator.wait_until_ready(timeout=30)   # bloquea hasta que edge-tts esté listo

# ── Image loading ─────────────────────────────────────────────────────────────
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
        raise HTTPException(status_code=400, detail="Unsupported image format")

    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    return image


# ── Drawing ───────────────────────────────────────────────────────────────────
LEVEL_LABELS = {
    "SAFE":     "SEGURO",
    "WARNING":  "PRECAUCION",
    "CRITICAL": "PELIGRO",
}


def _draw_frame(image: np.ndarray, detections: list, result: dict) -> np.ndarray:
    out   = image.copy()
    h, w  = out.shape[:2]
    level = result["risk_level"]
    color = tuple(RISK_COLORS.get(level, (0, 200, 0)))

    # ── Bounding boxes — solo rectángulos, sin texto ni confianza ────────
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        c = (0, 0, 255) if det["class_id"] == 0 else color
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)

    # ── Fight bounding boxes — rectángulo naranja-rojo, sin texto ────────
    if result.get("fight_detected"):
        FIGHT_COLOR = (0, 60, 255)
        for (fx1, fy1, fx2, fy2) in result.get("fight_persons", []):
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), FIGHT_COLOR, 3)

    # ── Top banner — solo nivel de alerta ────────────────────────────────
    cv2.rectangle(out, (0, 0), (w, 36), color, -1)
    banner = "PELEA DETECTADA" if result.get("fight_detected") else LEVEL_LABELS.get(level, level)
    cv2.putText(out, f"LUCY: {banner}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def _encode_frame(image: np.ndarray, detections: list, result: dict) -> str:
    """Devuelve la imagen anotada como string base64 JPEG."""
    annotated = _draw_frame(image, detections, result)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": "5.0.0 -- LUCY TrafficRiskNet",
        "model":   engine._model_name,
        "device":  str(next(engine._model.parameters()).device) if engine._model else "cpu",
        "voice":  narrator.status,
        "endpoints": {
            "analyze":         "POST /analyze?session_id=default",
            "analyze_visual":  "POST /analyze_visual?session_id=default",
            "reset":           "POST /traffic/reset?session_id=default",
            "session_info":    "GET  /traffic/session?session_id=default",
            "legacy":          "POST /detect",
        },
    }


@app.post("/analyze")
async def analyze(
    file:       UploadFile = File(...),
    session_id: str        = Query(default="default"),
):
    """
    Road risk analysis -- pedestrian perspective.
    Send frames consecutively with the same session_id to activate the GRU.
    UE5 Blueprint: same endpoint as before, session_id is optional.
    """
    try:
        image  = _load_image(await file.read())
        t0     = time.time()
        dets   = detector.detect(image)
        result = engine.analyze(dets, image.shape, image=image, session_id=session_id)
        fight  = fight_engine.analyze(image, session_id)
        result.update(fight)
        if fight["fight_detected"]:
            result["risk_level"]   = "CRITICAL"
            result["color_alerta"] = list(RISK_COLORS["CRITICAL"])
        narrator.speak(result)
        dt        = (time.time() - t0) * 1000.0
        narrative = build_narrative(result)

        return JSONResponse({
            # UE5-compatible keys (same as v4)
            "risk_level":    result["risk_level"],
            "titulo":        result["titulo"],
            "mensaje":       result["mensaje"],
            "leccion":       result["leccion"],
            "color_alerta":  result["color_alerta"],
            # Extended LUCY output
            "confidence":        result["confidence"],
            "all_probs":         result["all_probs"],
            "tl_override":       result["tl_override"],
            "tl_states":         result["tl_states"],
            "buffer_ready":      result["buffer_ready"],
            "frames_in":         result["frames_in"],
            "seq_len":           result["seq_len"],
            "model_name":        result["model_name"],
            "stats":             result["stats"],
            "narrative":         narrative,
            # Fight detection
            "fight_detected":    fight["fight_detected"],
            "fight_confidence":  fight["fight_confidence"],
            "fight_ready":       fight["fight_ready"],
            "todas_detecciones": dets,
            "inference_time_ms": round(dt, 2),
            "image_shape":       {"height": image.shape[0], "width": image.shape[1]},
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_visual")
async def analyze_visual(
    file:       UploadFile = File(...),
    session_id: str        = Query(default="default"),
):
    """Same as /analyze but returns annotated JPEG image."""
    try:
        image  = _load_image(await file.read())
        dets   = detector.detect(image)
        result = engine.analyze(dets, image.shape, image=image, session_id=session_id)
        fight  = fight_engine.analyze(image, session_id)
        result.update(fight)
        narrator.speak(result)
        out    = _draw_frame(image, dets, result)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/traffic/reset")
async def traffic_reset(session_id: str = Query(default="default")):
    """Reset the GRU and fight frame buffers for a session (new scene in UE5)."""
    engine.reset(session_id)
    fight_engine.reset(session_id)
    return {"status": "ok", "session_id": session_id, "message": "Buffer cleared"}


@app.get("/traffic/session")
async def traffic_session(session_id: str = Query(default="default")):
    """Check GRU buffer state for a session."""
    return {"session_id": session_id, **engine.session_info(session_id)}


@app.get("/traffic/sessions")
async def traffic_sessions():
    """List all active sessions."""
    return {
        sid: engine.session_info(sid)
        for sid in engine._buffers
    }


# ── Legacy endpoints (UE5 Blueprint backwards compatibility) ─────────────────
@app.post("/security")
async def security_compat(
    file:       UploadFile = File(...),
    session_id: str        = Query(default="default"),
    timestamp:  float      = Query(default=None),
):
    """
    Compatibility stub for old UE5 Blueprints that call /security.
    Proxies to the traffic engine and maps the response to the old schema.
    """
    try:
        _ = timestamp  # accepted for Blueprint compatibility, not used
        image  = _load_image(await file.read())
        t0     = time.time()
        dets   = detector.detect(image)
        result = engine.analyze(dets, image.shape, image=image, session_id=session_id)
        fight  = fight_engine.analyze(image, session_id)
        result.update(fight)
        # Pelea fuerza CRITICAL para que Unreal suba el widget de alerta
        if fight["fight_detected"]:
            result["risk_level"]   = "CRITICAL"
            result["color_alerta"] = list(RISK_COLORS["CRITICAL"])
        narrator.speak(result)
        dt     = (time.time() - t0) * 1000.0
        stats  = result.get("stats", {})

        # Map traffic result to old security schema so Blueprint reads correctly
        return JSONResponse({
            "risk_level":    result["risk_level"],
            "color_alerta":  result["color_alerta"],
            "behavior":      result["risk_level"].lower(),
            "behavior_conf": result["confidence"],
            "all_behaviors": {k: v for k, v in result["all_probs"].items()},
            "rl_action":     result["risk_level"],
            "weapons_found": 0,
            "persons_found": stats.get("personas", 0),
            "buffer_ready":  result["buffer_ready"],
            "frame":         result["frames_in"],
            "session_id":    session_id,
            # Fight detection — para el widget de pelea en Unreal
            "fight_detected":   fight["fight_detected"],
            "fight_confidence": fight["fight_confidence"],
            "fight_ready":      fight["fight_ready"],
            # Imagen anotada con bounding boxes (base64 JPEG) para el overlay de Unreal
            "annotated_frame":   _encode_frame(image, dets, result),
            "todas_detecciones": dets,
            "inference_time_ms": round(dt, 2),
            "image_shape":   {"height": image.shape[0], "width": image.shape[1]},
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security_visual")
async def security_visual_compat(
    file:       UploadFile = File(...),
    session_id: str        = Query(default="default"),
    timestamp:  float      = Query(default=None),
):
    """Compatibility stub -- returns annotated image for old /security_visual calls."""
    try:
        _ = timestamp  # accepted for Blueprint compatibility, not used
        image  = _load_image(await file.read())
        dets   = detector.detect(image)
        result = engine.analyze(dets, image.shape, image=image, session_id=session_id)
        fight  = fight_engine.analyze(image, session_id)
        result.update(fight)
        narrator.speak(result)
        out    = _draw_frame(image, dets, result)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security/reset")
async def security_reset_compat(session_id: str = Query(default="default")):
    """Compatibility stub for /security/reset."""
    engine.reset(session_id)
    fight_engine.reset(session_id)
    return {"status": "ok", "session_id": session_id}


@app.post("/fight/reset")
async def fight_reset(session_id: str = Query(default="default")):
    """Reset the fight detection frame buffer for a session."""
    fight_engine.reset(session_id)
    return {"status": "ok", "session_id": session_id}


@app.post("/detect")
async def detect_legacy(file: UploadFile = File(...)):
    """Legacy endpoint -- person detection only. Keeps old UE5 Blueprints working."""
    try:
        image   = _load_image(await file.read())
        dets    = detector.detect(image)
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
