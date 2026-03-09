# StealthVision Backend

Backend de inferencia de detección de personas usando **RF-DETR** (Roboflow Detection Transformer) exportado a ONNX, ejecutado con ONNX Runtime (GPU si hay CUDA, CPU si no), y expuesto vía FastAPI.

## 🚀 Características

- ✅ Detección de personas con **RF-DETR** (Estado del Arte - SOTA)
- ✅ Inferencia optimizada con **ONNX Runtime** (GPU/CPU)
- ✅ API REST con **FastAPI**
- ✅ Soporte para CUDA (GPU aceleration)
- ✅ Preprocesamiento optimizado con aspect ratio preservado
- ✅ Endpoints JSON y visualización con bounding boxes
- ✅ Exportación automatizada desde RF-DETR de Roboflow

## 📁 Estructura del Proyecto

```
StealthVision/
├── models/                    # Modelos ONNX
│   └── inference_model.onnx  # Modelo RF-DETR exportado
├── utils/                    # Utilidades adicionales
├── test_images/              # Imágenes de prueba
├── output/                   # Resultados de detección
├── main.py                   # API FastAPI principal
├── rtdetr_detector.py        # Clase detector ONNX
├── export_rfdetr_onnx.py     # Script de exportación RF-DETR → ONNX
├── test_onnx_model.py        # Script de validación de modelo
├── setup_rfdetr.py           # Setup automatizado
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
└── EXPORT_MODEL_GUIDE.md     # Guía detallada de exportación
```

## 🔧 Instalación

### Opción 1: Setup Automatizado (Recomendado)

```bash
# 1. Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# 2. Ejecutar setup completo
python setup_rfdetr.py
```

Este script instala automáticamente:
- PyTorch y TorchVision
- RF-DETR de Roboflow
- ONNX y ONNX Simplifier
- Exporta el modelo RF-DETR Medium a ONNX

### Opción 2: Setup Manual

#### 1. Activar el entorno virtual

```bash
.\.venv\Scripts\Activate.ps1
```

#### 2. Instalar dependencias básicas

```bash
pip install -r requirements.txt
```

#### 3. Instalar RF-DETR y dependencias de exportación

```bash
# PyTorch (ajusta según tu CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# RF-DETR (desde GitHub)
pip install git+https://github.com/roboflow/rf-detr.git

# Herramientas ONNX
pip install onnx onnxsim
```

#### 4. Exportar modelo RF-DETR a ONNX

```bash
# Exportar modelo Medium (recomendado)
python export_rfdetr_onnx.py --size medium

# O elegir otro tamaño
python export_rfdetr_onnx.py --size small    # Más rápido
python export_rfdetr_onnx.py --size large    # Más preciso
```

Esto generará `models/inference_model.onnx`.

#### 5. Verificar el modelo exportado

```bash
python test_onnx_model.py models/inference_model.onnx
```

### 📚 Documentación de Exportación

Para opciones avanzadas de exportación (checkpoints personalizados, tamaños de entrada, etc.), consulta:

**[📖 EXPORT_MODEL_GUIDE.md](EXPORT_MODEL_GUIDE.md)**

## 🏃 Uso

### Iniciar el servidor API

**Opción 1: Directamente con Python**

```bash
python main.py
```

**Opción 2: Con uvicorn (recomendado para desarrollo)**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en: `http://localhost:8000`

### Documentación interactiva

FastAPI proporciona documentación automática:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔌 Endpoints de la API

### 1. Health Check

Verifica el estado del servicio y configuración del modelo.

```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "model": "RT-DETR ONNX",
  "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "conf_threshold": 0.5,
  "input_size": "640x640"
}
```

### 2. Detección (JSON)

Detecta personas en una imagen y devuelve JSON con las coordenadas y confianza.

**Endpoint:** `POST /detect`

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/person.jpg"
```

**Respuesta:**
```json
{
  "detected": true,
  "count": 2,
  "detections": [
    {
      "bbox": [123.45, 67.89, 456.78, 789.01],
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person"
    },
    {
      "bbox": [567.89, 123.45, 890.12, 345.67],
      "confidence": 0.87,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "inference_time_ms": 25.34,
  "shape": {
    "h": 1080,
    "w": 1920
  }
}
```

### 3. Detección Visual

Detecta personas y devuelve la imagen con bounding boxes dibujados.

**Endpoint:** `POST /detect_visual`

```bash
curl -X POST "http://localhost:8000/detect_visual" \
  -H "accept: image/jpeg" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/person.jpg" \
  --output output/result.jpg
```

La imagen resultante se guarda en `output/result.jpg` con los bounding boxes verde y etiquetas.

## 🧪 Ejemplos de Prueba

### Probar con imagen local

```bash
# 1. Coloca una imagen en test_images/
# 2. Ejecuta:
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_images/person.jpg"
```

### Probar con Python

```python
import requests

# Detectar personas
with open("test_images/person.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": f}
    )
    result = response.json()
    print(f"Detectadas {result['count']} personas")
    print(f"Tiempo: {result['inference_time_ms']:.2f} ms")

# Obtener imagen visual
with open("test_images/person.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect_visual",
        files={"file": f}
    )
    with open("output/result.jpg", "wb") as out:
        out.write(response.content)
```

## 🎮 Integración con Unreal Engine

Este backend está diseñado para integrarse con Unreal Engine mediante HTTP requests.

### Ejemplo Blueprint (Unreal Engine)

1. **Capturar frame de cámara** → Guardar como JPEG temporal
2. **HTTP Request** → `POST http://localhost:8000/detect`
3. **Parsear JSON** → Obtener detecciones
4. **Renderizar** → Dibujar bounding boxes o activar eventos

### Ejemplo C++ (Unreal Engine)

```cpp
// Enviar imagen al backend
FHttpModule* Http = &FHttpModule::Get();
TSharedRef<IHttpRequest> Request = Http->CreateRequest();
Request->SetURL("http://localhost:8000/detect");
Request->SetVerb("POST");
Request->SetHeader("Content-Type", "multipart/form-data");

// Agregar imagen como archivo
// ... (código de serialización)

Request->ProcessRequest();
```

## ⚙️ Configuración Avanzada

### Ajustar umbral de confianza

Edita [main.py](main.py) línea donde se crea el detector:

```python
detector = RTDetrDetector(
    model_path="models/rtdetr.onnx",
    conf_threshold=0.7,  # Aumentar para menos falsos positivos
    person_class_id=0
)
```

### Cambiar clase objetivo

Si tu modelo detecta otras clases además de personas, ajusta `person_class_id`:

```python
# Ejemplo: COCO dataset
# 0 = person, 1 = bicycle, 2 = car, etc.
detector = RTDetrDetector(
    model_path="models/rtdetr.onnx",
    conf_threshold=0.5,
    person_class_id=0  # Cambiar según tu modelo
)
```

## 📊 Rendimiento

Con CUDA habilitado (RTX 3060 ejemplo):
- Inferencia: ~20-30ms por imagen (1920x1080)
- Preprocesamiento: ~5ms
- Postprocesamiento: ~2ms

Con CPU (Intel i7 ejemplo):
- Inferencia: ~100-150ms por imagen

## 🐛 Troubleshooting

### Error: "CUDA not available"

```bash
# Verificar CUDA con PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Si es False, instala CUDA toolkit y reinstala onnxruntime-gpu
pip uninstall onnxruntime-gpu
pip install onnxruntime-gpu
```

### Error: "Cannot find model file"

```bash
# Verificar que el modelo existe
ls models/rtdetr.onnx

# Si no existe, coloca tu modelo ONNX en esa ubicación
```

### Inferencia lenta

1. Verifica que CUDA esté activo: revisar logs al iniciar
2. Reduce la resolución de entrada del modelo
3. Ajusta `conf_threshold` para filtrar más detecciones

## 📝 Licencia

Proyecto educativo para StealthVision.

## 🤝 Contribuciones

Este es un backend base. Se puede extender con:
- [ ] Tracking multi-objeto
- [ ] WebSocket para streaming en vivo
- [ ] Batch processing
- [ ] Métricas de rendimiento con Prometheus
- [ ] Rate limiting
- [ ] Autenticación JWT

---

**StealthVision** - Sistema de detección de personas en tiempo real 🎯
