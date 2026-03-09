"""
Script de setup paso a paso para instalar RF-DETR y exportar el modelo
"""
import subprocess
import sys
import os


def print_step(step_num, total, title):
    """Imprime encabezado de paso"""
    print(f"\n{'='*70}")
    print(f"📋 Paso {step_num}/{total}: {title}")
    print(f"{'='*70}\n")


def run_command(command, description):
    """Ejecuta un comando y muestra resultado"""
    print(f"🔄 {description}...")
    print(f"💻 Comando: {command}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        print(f"✓ {description} completado")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        print(f"Código de salida: {e.returncode}")
        if e.stdout:
            print("Salida estándar:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False


def check_venv():
    """Verifica si estamos en un entorno virtual"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("⚠️  No estás en un entorno virtual")
        print("   Recomendado: activa el entorno virtual primero")
        print("   Ejecuta: .venv\\Scripts\\Activate.ps1")
        
        response = input("\n¿Continuar de todos modos? (s/n): ")
        if response.lower() != 's':
            print("Configuración cancelada")
            return False
    else:
        print("✓ Entorno virtual activo")
    
    return True


def main():
    """Función principal de setup"""
    print("\n" + "="*70)
    print("🚀 Setup de RF-DETR para StealthVision")
    print("="*70)
    
    # Verificar entorno virtual
    if not check_venv():
        return
    
    # Paso 1: Instalar dependencias básicas
    print_step(1, 4, "Instalar Dependencias Básicas")
    success = run_command(
        "pip install --upgrade pip",
        "Actualizar pip"
    )
    if not success:
        print("\n⚠️  No se pudo actualizar pip, continuando...")
    
    success = run_command(
        "pip install onnx onnxsim numpy opencv-python",
        "Instalar dependencias básicas"
    )
    if not success:
        print("\n❌ Error instalando dependencias básicas")
        return
    
    # Paso 2: Instalar PyTorch
    print_step(2, 4, "Instalar PyTorch")
    print("ℹ️  Detectando configuración de GPU...\n")
    
    # Detectar si hay CUDA disponible
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            print(f"✓ CUDA detectado: {torch.cuda.get_device_name(0)}")
            print("   Usando PyTorch con soporte CUDA existente")
        else:
            print("⚠️  CUDA no disponible, instalando PyTorch CPU")
            success = run_command(
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
                "Instalar PyTorch CPU"
            )
            if not success:
                print("\n❌ Error instalando PyTorch")
                return
    except ImportError:
        print("PyTorch no instalado, instalando versión CPU...")
        success = run_command(
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            "Instalar PyTorch CPU"
        )
        if not success:
            print("\n❌ Error instalando PyTorch")
            return
    
    # Paso 3: Instalar RF-DETR
    print_step(3, 4, "Instalar RF-DETR")
    success = run_command(
        "pip install git+https://github.com/roboflow/rf-detr.git",
        "Instalar RF-DETR de Roboflow"
    )
    if not success:
        print("\n❌ Error instalando RF-DETR")
        print("ℹ️  Verifica tu conexión a internet y git")
        return
    
    # Paso 4: Exportar modelo
    print_step(4, 4, "Exportar Modelo a ONNX")
    
    # Verificar si ya existe el modelo
    model_path = "models/inference_model.onnx"
    if os.path.exists(model_path):
        print(f"⚠️  Ya existe un modelo en: {model_path}")
        response = input("¿Exportar nuevo modelo y sobrescribir? (s/n): ")
        if response.lower() != 's':
            print("Exportación omitida")
        else:
            success = run_command(
                "python export_rfdetr_onnx.py --size medium",
                "Exportar RF-DETR Medium a ONNX"
            )
            if not success:
                print("\n❌ Error exportando modelo")
                return
    else:
        print("No se encontró modelo existente, exportando...")
        success = run_command(
            "python export_rfdetr_onnx.py --size medium",
            "Exportar RF-DETR Medium a ONNX"
        )
        if not success:
            print("\n❌ Error exportando modelo")
            print("\nℹ️  Puedes exportar manualmente con:")
            print("   python export_rfdetr_onnx.py --size medium")
            return
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ Setup completado exitosamente!")
    print("="*70)
    
    print("\n📝 Qué se instaló:")
    print("   ├─ ONNX y ONNX Simplifier")
    print("   ├─ PyTorch y TorchVision")
    print("   ├─ RF-DETR de Roboflow")
    print("   └─ Modelo RF-DETR Medium exportado a ONNX")
    
    print("\n🚀 Siguientes pasos:")
    print("   1. Probar el modelo exportado:")
    print("      python test_onnx_model.py models/inference_model.onnx")
    print("\n   2. Instalar dependencias de la API:")
    print("      pip install -r requirements.txt")
    print("\n   3. Ejecutar la API:")
    print("      python main.py")
    print("\n   4. O probar detección en imagen específica (si se implementa CLI):")
    print("      python -c 'from main import detector; detector.detect(...)'")
    
    print("\n📚 Documentación adicional:")
    print("   - Ver EXPORT_MODEL_GUIDE.md para opciones avanzadas")
    print("   - Ver README.md para documentación completa")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
