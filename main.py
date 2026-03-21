from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from rtdetr_detector import RTDetrDetector
import numpy as np
import cv2
import time
import tempfile
import os

app = FastAPI(title="StealthVision RT-DETR Backend", version="0.1.0")

detector = RTDetrDetector(
    model_path="rtdetr-l.onnx",
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
    Recibe una imagen enviada por Unreal (PNG/JPEG o EXR)
    y devuelve detecciones en JSON.
    """
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        # 1) Intentar como PNG/JPEG normal
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2) Si falla, intentar como EXR guardándolo temporalmente
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
            raise HTTPException(status_code=400, detail="Imagen inválida o formato no soportado")

        # Asegurar BGR uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

        t0 = time.time()
        detections = detector.detect(image)
        dt = (time.time() - t0) * 1000.0

        return JSONResponse(
            {
                "detected": len(detections) > 0,
                "count": len(detections),
                "detections": detections,
                "inference_time_ms": round(dt, 2),
                "image_shape": {
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                },
                "content_type": file.content_type,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        # log detallado en consola para debug
        print(" Backend error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
