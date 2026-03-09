"""
Script de prueba para verificar modelo RF-DETR exportado a ONNX
Valida que el modelo carga correctamente y muestra información de entrada/salida
"""
import os
import sys
import argparse
import numpy as np


def test_onnx_model(model_path: str, detailed: bool = False):
    """
    Prueba un modelo ONNX y muestra su información
    
    Args:
        model_path: Ruta al archivo .onnx
        detailed: Si mostrar información detallada
    """
    print(f"\n{'='*70}")
    print(f"🧪 Probando modelo ONNX: {model_path}")
    print(f"{'='*70}\n")
    
    # Verificar que el archivo existe
    if not os.path.exists(model_path):
        print(f"❌ Error: El archivo no existe: {model_path}")
        return False
    
    # Obtener tamaño del archivo
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📁 Tamaño del archivo: {file_size:.2f} MB\n")
    
    # Intentar cargar con ONNX (opcional)
    try:
        import onnx
        print("📦 Cargando con ONNX...")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("   ✓ Modelo ONNX válido\n")
        
        if detailed:
            print("📊 Metadatos del modelo:")
            print(f"   ├─ IR Version: {model.ir_version}")
            print(f"   ├─ Producer: {model.producer_name}")
            print(f"   └─ Opset: {model.opset_import[0].version if model.opset_import else 'N/A'}\n")
    
    except ImportError:
        print("⚠️  Paquete 'onnx' no instalado (opcional)")
        print("   Instalar con: pip install onnx\n")
    except Exception as e:
        print(f"⚠️  Advertencia al cargar con ONNX: {str(e)}\n")
    
    # Cargar con ONNX Runtime (principal)
    try:
        import onnxruntime as ort
        print("🚀 Cargando con ONNX Runtime...")
        
        # Configurar providers
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Mostrar providers activos
        active_providers = session.get_providers()
        print(f"   ├─ Providers disponibles: {providers}")
        print(f"   └─ Providers activos: {active_providers}")
        
        if "CUDAExecutionProvider" in active_providers:
            print("   ✓ GPU (CUDA) activada\n")
        else:
            print("   ⚠️  Ejecución en CPU (CUDA no disponible)\n")
        
        # Información de entradas
        print("📥 Entradas del modelo:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"   [{i}] {input_info.name}")
            print(f"       ├─ Forma: {input_info.shape}")
            print(f"       └─ Tipo: {input_info.type}")
        print()
        
        # Información de salidas
        print("📤 Salidas del modelo:")
        for i, output_info in enumerate(session.get_outputs()):
            print(f"   [{i}] {output_info.name}")
            print(f"       ├─ Forma: {output_info.shape}")
            print(f"       └─ Tipo: {output_info.type}")
        print()
        
        # Prueba de inferencia dummy
        print("🔄 Ejecutando inferencia de prueba...")
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Manejar dimensiones dinámicas
        batch = 1
        channels = input_shape[1] if input_shape[1] > 0 else 3
        height = input_shape[2] if input_shape[2] > 0 else 640
        width = input_shape[3] if input_shape[3] > 0 else 640
        
        # Crear entrada dummy
        dummy_input = np.random.randn(batch, channels, height, width).astype(np.float32)
        print(f"   ├─ Input: {input_name}")
        print(f"   └─ Forma: {dummy_input.shape}")
        
        # Ejecutar inferencia
        try:
            outputs = session.run(None, {input_name: dummy_input})
            print(f"\n   ✓ Inferencia exitosa!")
            print(f"   └─ Número de salidas: {len(outputs)}\n")
            
            # Mostrar formas de salida
            print("📊 Formas de salida:")
            for i, output in enumerate(outputs):
                print(f"   [{i}] {output.shape} - {output.dtype}")
            print()
            
            # Intentar determinar el formato
            print("🔍 Análisis del formato de salida:")
            if len(outputs) >= 3:
                print("   ✓ Formato detectado: RF-DETR estándar")
                print(f"      ├─ Salida 0 (labels): {outputs[0].shape}")
                print(f"      ├─ Salida 1 (boxes): {outputs[1].shape}")
                print(f"      └─ Salida 2 (scores): {outputs[2].shape}")
                
                # Estadísticas de las salidas
                if outputs[0].size > 0:
                    num_detections = outputs[0].shape[-1] if len(outputs[0].shape) > 1 else len(outputs[0])
                    print(f"\n   📈 Estadísticas de la inferencia dummy:")
                    print(f"      ├─ Detecciones: {num_detections}")
                    if outputs[2].size > 0:
                        max_score = np.max(outputs[2])
                        min_score = np.min(outputs[2])
                        print(f"      ├─ Score máximo: {max_score:.4f}")
                        print(f"      └─ Score mínimo: {min_score:.4f}")
            else:
                print("   ⚠️  Formato no estándar detectado")
                print("      Puede requerir ajustes en rtdetr_detector.py")
            print()
            
        except Exception as e:
            print(f"\n   ❌ Error durante inferencia: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # Resumen final
        print(f"{'='*70}")
        print("✅ Modelo ONNX cargado y probado exitosamente")
        print(f"{'='*70}\n")
        
        print("📝 Siguientes pasos:")
        print("   1. Actualiza rtdetr_detector.py con:")
        print(f"      model_path=\"{model_path}\"")
        print("   2. Ejecuta la API: python main.py")
        print("   3. Visita la documentación: http://localhost:8000/docs\n")
        
        return True
        
    except ImportError:
        print("❌ Error: onnxruntime no está instalado")
        print("   Instalar con: pip install onnxruntime-gpu")
        return False
    except Exception as e:
        print(f"❌ Error al cargar modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal con CLI"""
    parser = argparse.ArgumentParser(
        description="Prueba un modelo RF-DETR exportado a ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Prueba básica
  python test_onnx_model.py models/inference_model.onnx
  
  # Prueba con información detallada
  python test_onnx_model.py models/inference_model.onnx --detailed
        """
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Ruta al archivo .onnx a probar"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Mostrar información detallada del modelo"
    )
    
    args = parser.parse_args()
    
    # Ejecutar prueba
    success = test_onnx_model(args.model_path, args.detailed)
    
    # Salir con código apropiado
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
