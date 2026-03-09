import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any


class RTDetrDetector:
    """
    Wrapper para modelo RT-DETR ONNX exportado con Ultralytics.
    Asume salida (1, 300, 84): 4 coords + 80 scores COCO.
    """

    def __init__(
        self,
        model_path: str = "rtdetr-l.onnx",
        conf_threshold: float = 0.5,
        person_class_id: int = 0,
        providers=None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.in_h, self.in_w = self.session.get_inputs()[0].shape

        print("✅ RTDetrDetector inicializado")
        print("   Modelo:", self.model_path)
        print("   Input esperado:", self.in_w, "x", self.in_h)
        print("   Providers:", self.session.get_providers())

    def preprocess(self, image: np.ndarray):
        """Resize + padding + normalización. Devuelve tensor, escala y tamaño original."""
        orig_h, orig_w = image.shape[:2]

        scale = min(self.in_w / orig_w, self.in_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # [1,3,H,W]

        return img, scale, (orig_h, orig_w)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        orig_shape,
    ) -> List[Dict[str, Any]]:
        """Convierte salida (1,300,84) a lista de detecciones en coords de la imagen original."""
        preds = outputs[0][0]              # (300, 84)
        boxes = preds[:, 0:4]              # xc, yc, w, h (normalizados)
        class_scores = preds[:, 4:]        # (300, 80)

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        detections = []
        orig_h, orig_w = orig_shape

        for i in range(preds.shape[0]):
            score = float(scores[i])
            cls = int(class_ids[i])

            if score < self.conf_threshold:
                continue

            # Si quieres detectar solo personas:
            if cls != self.person_class_id:
                continue

            xc, yc, w, h = boxes[i]

            # Des-normalizar al tamaño del canvas
            xc *= self.in_w
            yc *= self.in_h
            w *= self.in_w
            h *= self.in_h

            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)

            # Reescalar a tamaño original
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            # Clampear
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": score,
                    "class_id": cls,
                    "class_name": "person",  # COCO id 0
                }
            )

        return detections

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta personas en una imagen BGR."""
        inp, scale, orig_shape = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: inp})
        detections = self.postprocess(outputs, scale, orig_shape)
        return detections
