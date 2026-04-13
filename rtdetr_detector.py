import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Set

# Todas las clases COCO80
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
    52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

# Clases relevantes para educacion vial
TRAFFIC_CLASS_IDS = {0, 1, 2, 3, 5, 7, 9, 11}
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic light, 11=stop sign


class RTDetrDetector:
    """
    Wrapper para modelo RT-DETR ONNX (Ultralytics).
    Soporta deteccion multi-clase configurable.
    Salida esperada del modelo: (1, 300, 84) — 4 coords + 80 scores COCO.
    """

    def __init__(
        self,
        model_path: str = "rtdetr-l.onnx",
        conf_threshold: float = 0.4,
        target_class_ids: Optional[Set[int]] = None,
        person_class_id: int = 0,   # parametro legacy, se mantiene por compatibilidad
        providers=None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # Si se especifican clases, usarlas; si no, usar solo persona (legacy)
        if target_class_ids is not None:
            self.target_class_ids = set(target_class_ids)
        else:
            self.target_class_ids = {person_class_id}

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.in_h, self.in_w = self.session.get_inputs()[0].shape

        class_names = [COCO_CLASSES.get(c, f"class_{c}") for c in sorted(self.target_class_ids)]
        print("RTDetrDetector listo")
        print(f"  Modelo:   {self.model_path}")
        print(f"  Input:    {self.in_w}x{self.in_h}")
        print(f"  Clases:   {class_names}")
        print(f"  Umbral:   {self.conf_threshold}")
        print(f"  Backend:  {self.session.get_providers()}")

    # ------------------------------------------------------------------
    def preprocess(self, image: np.ndarray):
        """Letterbox resize + normalización. Devuelve tensor, escala, forma original."""
        orig_h, orig_w = image.shape[:2]
        scale = min(self.in_w / orig_w, self.in_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.expand_dims(img.transpose(2, 0, 1), 0)   # [1, 3, H, W]
        return img, scale, (orig_h, orig_w)

    # ------------------------------------------------------------------
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        orig_shape,
    ) -> List[Dict[str, Any]]:
        """Convierte salida (1,300,84) → lista de detecciones en coords originales."""
        preds = outputs[0][0]          # (300, 84)
        boxes = preds[:, :4]           # xc, yc, w, h  (normalizados 0-1)
        class_scores = preds[:, 4:]    # (300, 80)

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        orig_h, orig_w = orig_shape
        detections = []

        for i in range(len(preds)):
            score = float(scores[i])
            cls = int(class_ids[i])

            if score < self.conf_threshold:
                continue
            if cls not in self.target_class_ids:
                continue

            xc, yc, w, h = boxes[i]
            xc *= self.in_w;  yc *= self.in_h
            w  *= self.in_w;  h  *= self.in_h

            x1 = max(0, min(int((xc - w / 2) / scale), orig_w - 1))
            y1 = max(0, min(int((yc - h / 2) / scale), orig_h - 1))
            x2 = max(0, min(int((xc + w / 2) / scale), orig_w - 1))
            y2 = max(0, min(int((yc + h / 2) / scale), orig_h - 1))

            detections.append({
                "bbox":        [x1, y1, x2, y2],
                "confidence":  round(score, 4),
                "class_id":    cls,
                "class_name":  COCO_CLASSES.get(cls, f"class_{cls}"),
            })

        return detections

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta objetos en una imagen BGR."""
        inp, scale, orig_shape = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: inp})
        return self.postprocess(outputs, scale, orig_shape)

    def detect_all_traffic(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta todas las clases de trafico de una vez, sin cambiar la config."""
        saved = self.target_class_ids
        self.target_class_ids = TRAFFIC_CLASS_IDS
        result = self.detect(image)
        self.target_class_ids = saved
        return result
