from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from rtdetr_detector import RTDetrDetector
import numpy as np
import cv2
import time

app = FastAPI(title="StealthVision RT-DETR Backend", version="0.1.0")

detector = RTDetrDetector(
    model_path="rtdetr-l.onnx",   # ajusta si está en otra ruta
    conf_threshold=0.5,
    person_class_id=0,
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": detector.model_path,
        "providers": detector.session.get_providers(),
        "input_size": {"h": detector.in_h, "w": detector.in_w},
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Recibe una imagen (JPEG/PNG) enviada por Unreal (o cualquier cliente)
    y devuelve detecciones en JSON.
    """
    try:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")

        t0 = time.time()
        detections = detector.detect(image)
        dt = (time.time() - t0) * 1000  # ms

        return JSONResponse(
            {
                "detected": len(detections) > 0,
                "count": len(detections),
                "detections": detections,
                "inference_time_ms": round(dt, 2),
                "image_shape": {
                    "height": image.shape[0],
                    "width": image.shape[1],
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
