from ultralytics import RTDETR


def main():
    # Puedes cambiar 'rtdetr-l.pt' por 'rtdetr-x.pt', 'rtdetr-s.pt', etc.
    model_name = "rtdetr-l.pt"

    print("=" * 60)
    print(f"🚀 Exportando {model_name} a ONNX con Ultralytics RT-DETR")
    print("=" * 60)

    # Cargar modelo preentrenado
    model = RTDETR(model_name)

    # Exportar a ONNX
    # imgsz puede ser un int o tupla (h, w); 640 es estándar
    model.export(
        format="onnx",
        imgsz=640,
        half=False,
        simplify=True,
        dynamic=False,
        opset=16,  # opset estable para ORT moderno
        nms=True,  # incluye NMS dentro del ONNX
    )

    print("✅ Exportación iniciada, revisa el archivo generado (rtdetr-l.onnx).")


if __name__ == "__main__":
    main()
