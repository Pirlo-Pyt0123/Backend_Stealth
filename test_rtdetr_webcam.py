import cv2
import numpy as np
import onnxruntime as ort
import time
import os

MODEL_PATH = "rtdetr-l.onnx"
CONF_THRESHOLD = 0.5
PERSON_CLASS_ID = 0

def main():
    os.makedirs("output", exist_ok=True)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)

    input_name = session.get_inputs()[0].name
    _, _, in_h, in_w = session.get_inputs()[0].shape

    print("✅ Modelo cargado:", MODEL_PATH)
    print("   Input esperado:", in_w, "x", in_h)
    print("   Providers:", session.get_providers())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer frame de la cámara")
            break

        frame_idx += 1
        orig_h, orig_w = frame.shape[:2]

        # Preprocesar
        scale = min(in_w / orig_w, in_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        t0 = time.time()
        outputs = session.run(None, {input_name: img})
        dt = (time.time() - t0) * 1000

        preds = outputs[0][0]          # (300, 84)
        boxes = preds[:, 0:4]          # xc, yc, w, h (0..1)
        class_scores = preds[:, 4:]    # 80 clases

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        detections = []

        for i in range(preds.shape[0]):
            score = float(scores[i])
            cls = int(class_ids[i])

            if score < CONF_THRESHOLD or cls != PERSON_CLASS_ID:
                continue

            xc, yc, w, h = boxes[i]
            xc *= in_w
            yc *= in_h
            w *= in_w
            h *= in_h

            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            detections.append((x1, y1, x2, y2, score))

        # Dibujar y guardar frame
        for (x1, y1, x2, y2, score) in detections:
            label = f"person {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        fps_text = f"{1000.0/dt:.1f} FPS" if dt > 0 else "inf FPS"
        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out_path = f"output/frame_{frame_idx:03d}.jpg"
        cv2.imwrite(out_path, frame)
        print(f"[Frame {frame_idx}] personas: {len(detections)}, "
              f"tiempo: {dt:.2f} ms, guardado: {out_path}")

        if frame_idx >= 10:
            break

    cap.release()
    print("✅ Captura finalizada. Revisa la carpeta 'output/'.")


if __name__ == "__main__":
    main()
