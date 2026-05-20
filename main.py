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
import time
import io
import tempfile
import os

from engines.rtdetr_detector   import RTDetrDetector, TRAFFIC_CLASS_IDS
from engines.deep_traffic_engine import DeepTrafficEngine, RISK_COLORS, TL_COLORS_BGR, SEQ_LEN

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

engine = DeepTrafficEngine()

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

    # ── Bounding boxes ────────────────────────────────────────────────────
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        c     = (0, 0, 255) if det["class_id"] == 0 else color
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # TL colored circle
        if det["class_id"] == 9 and det.get("tl_state"):
            tl_col = TL_COLORS_BGR.get(det["tl_state"], (128, 128, 128))
            cx_tl  = (x1 + x2) // 2
            cy_tl  = max(y1 - 12, 12)
            cv2.circle(out, (cx_tl, cy_tl), 10, tl_col, -1)
            cv2.circle(out, (cx_tl, cy_tl), 10, (255, 255, 255), 1)
            label = f"TL:{det['tl_state']} {det['confidence']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), c, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Top banner ────────────────────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (w, 58), color, -1)
    conf = result.get("confidence", 0.0)
    frames_in = result.get("frames_in", 0)
    ready     = result.get("buffer_ready", False)
    buf_txt   = f"GRU: {frames_in}/{SEQ_LEN}" if ready else f"warming {frames_in}/{SEQ_LEN}"
    banner    = f"LUCY: {LEVEL_LABELS.get(level, level)}  ({conf*100:.0f}%)"
    if result.get("tl_override"):
        banner += "  [TL OVERRIDE]"
    cv2.putText(out, banner,
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    probs = result.get("all_probs", {})
    probs_txt = "  |  ".join(f"{k}: {v:.2f}" for k, v in probs.items())
    cv2.putText(out, probs_txt + "  " + buf_txt,
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Low-light indicator ───────────────────────────────────────────────
    stats = result.get("stats", {})
    if stats.get("is_dark"):
        txt = f"LOW LIGHT  {stats.get('brightness', 0):.0f}/255"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (w-tw-12, 0), (w, 20), (20, 20, 100), -1)
        cv2.putText(out, txt, (w-tw-8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1, cv2.LINE_AA)

    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": "5.0.0 -- LUCY TrafficRiskNet",
        "model":   engine._model_name,
        "device":  str(next(engine._model.parameters()).device) if engine._model else "cpu",
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
        dt     = (time.time() - t0) * 1000.0

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
        out    = _draw_frame(image, dets, result)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/traffic/reset")
async def traffic_reset(session_id: str = Query(default="default")):
    """Reset the GRU frame buffer for a session (new scene in UE5)."""
    engine.reset(session_id)
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
            "brightness":    stats.get("brightness", 128.0),
            "is_dark":       stats.get("is_dark", False),
            "buffer_ready":  result["buffer_ready"],
            "frame":         result["frames_in"],
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
