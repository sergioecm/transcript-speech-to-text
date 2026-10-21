# Transcriptor local: dos flujos independientes

**Flujo A:** MP3/WebM/WAV/MP4/MKV -> FFmpeg -> Faster-Whisper (o ASR API explícita) -> TXT/JSON/SRT -> FIN.

**Flujo B:** TXT/JSON/SRT -> corrección opcional Ollama -> texto corregido independiente -> XTTS-v2 opcional con voz autorizada -> WAV -> FIN.

## Instalación (Ubuntu 24.04 / Python 3.12)

```bash
sudo apt update && sudo apt install -y ffmpeg python3-venv
cd transcriptor_local
python3 -m venv .venv-asr
source .venv-asr/bin/activate
python -m pip install -U pip
pip install faster-whisper 'av<20' ollama openai nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'
# Necesario para las bibliotecas CUDA cargadas por CTranslate2 (antes de iniciar Python):
export LD_LIBRARY_PATH="$(python -c 'import os,nvidia.cublas.lib,nvidia.cudnn.lib;print(os.path.dirname(nvidia.cublas.lib.__file__)+":"+os.path.dirname(nvidia.cudnn.lib.__file__))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ollama pull qwen3:4b-instruct
```

Para XTTS, instalar un entorno **separado**, preferiblemente seleccionando un backend PyTorch CUDA compatible según https://pytorch.org/get-started/locally/:

```bash
python3 -m venv .venv-tts
source .venv-tts/bin/activate
python -m pip install -U pip
# Instale torch, torchaudio (y torchcodec si su versión lo requiere) según la guía oficial.
pip install coqui-tts
```

**Importante:** Para el comando `process --voice`, active el entorno `.venv-tts`, e instale también `ollama` en ese entorno si usará `--correct`. Para el comando `transcribe`, use `.venv-asr`. La aplicación necesita que las dependencias de cada flujo estén disponibles en el entorno activo. La descarga del modelo XTTS y la aceptación de su licencia pueden requerir una ejecución interactiva inicial.

```bash
cp .env.example .env
# Opcional: edite .env con sus credenciales, sin almacenarlas en código ni en control de versiones.
set -a; source .env; set +a

python3.12 -m app.main transcribe /home/{USER}/Vídeos/Kooha/parte_exacta_000.mp3 --output /home/{USER}/Vídeos/Kooha/outputs/transcripcion1

python -m app.main transcribe entrada.mp4 --output outputs/transcripcion1
python -m app.main transcribe entrada.webm --output outputs/transcripcion2 --engine groq --allow-cloud
python -m app.main process outputs/transcripcion1/transcripcion.srt --output outputs/texto1 --correct
python -m app.main process original.txt --output outputs/voz1 --voice --reference mi_voz_autorizada.wav --authorized
python -m app.main process original.txt --output outputs/voz2 --correct --voice --reference mi_voz_autorizada.wav --authorized
python -m unittest discover -s tests -v
```
---



## Instalación exclusiva para macOS M1 "Tahoe" (Versión 26)

> **Nota de hipótesis/entorno:** Esta sección está diseñada específicamente para la arquitectura Apple Silicon (M1) en el sistema operativo macOS "Tahoe" (Versión 26), según lo solicitado. Utiliza aceleración por hardware a través de Metal Performance Shaders (MPS) y CoreML, reemplazando por completo las dependencias de NVIDIA/CUDA.

### 1. Prerrequisitos del sistema
Asegúrese de tener [Homebrew](https://brew.sh/) instalado. Luego, instale las dependencias nativas del sistema:
```bash
brew update
brew install ffmpeg python@3.12
```
*Nota: Asegúrese también de tener la aplicación nativa de [Ollama para macOS](https://ollama.com/download) instalada y ejecutándose en segundo plano.*

### 2. Entorno para Flujo A (ASR / Transcripción)
```bash
cd transcriptor_local
python3.12 -m venv .venv-asr
source .venv-asr/bin/activate
python -m pip install -U pip

# Instalar dependencias optimizadas para Apple Silicon (sin paquetes NVIDIA)
pip install faster-whisper 'av<20' ollama openai

# Habilitar fallback a MPS para operaciones no soportadas nativamente en Metal
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Descargar el modelo (Ollama debe estar activo en el sistema)
ollama pull qwen3:4b-instruct
```

### 3. Entorno para Flujo B (Corrección y XTTS-v2)
Para XTTS, se requiere un entorno **separado** con PyTorch configurado para soportar MPS (Apple Silicon):
```bash
python3.12 -m venv .venv-tts
source .venv-tts/bin/activate
python -m pip install -U pip

# Instalar PyTorch con soporte nativo para macOS (MPS)
pip install torch torchaudio
pip install coqui-tts

# Habilitar fallback a MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 4. Configuración y Ejecución en macOS
```bash
cp .env.example .env
# Opcional: edite .env con sus credenciales, sin almacenarlas en código ni en control de versiones.
set -a; source .env; set +a

# Ejemplo de transcripción (usando .venv-asr)
# Nota: Las rutas en macOS utilizan /Users/ en lugar de /home/
python3.12 -m app.main transcribe /Users/{USER}/Vídeos/Kooha/parte_exacta_000.mp3 --output /Users/{USER}/Vídeos/Kooha/outputs/transcripcion1

# Ejemplo de procesamiento y síntesis de voz (usando .venv-tts)
python3.12 -m app.main process outputs/transcripcion1/transcripcion.srt --output outputs/texto1 --correct
python3.12 -m app.main process original.txt --output outputs/voz1 --voice --reference mi_voz_autorizada.wav --authorized
```

### Consideraciones específicas para macOS M1:
1. **Sin CUDA:** No se requieren ni funcionan las variables de entorno `LD_LIBRARY_PATH` de NVIDIA, ni los paquetes `nvidia-cublas-cu12` o `nvidia-cudnn-cu12`. Su inclusión en este entorno provocará errores de instalación.
2. **Aceleración:** `faster-whisper` (a través de CTranslate2) detectará y utilizará automáticamente la aceleración CoreML/MPS en chips M1 si la versión instalada lo soporta.
3. **Ollama:** En macOS, el comando `ollama` en la terminal se comunica con el demonio de la aplicación de escritorio de Ollama. No es necesario (ni recomendado) instalar el servidor de Ollama vía `pip` dentro del entorno virtual.
4. **Rutas:** Se han ajustado los ejemplos de ruta de `/home/{USER}/...` a `/Users/{USER}/...` para cumplir con la estructura de directorios estándar de macOS.

---

### Contratos y limitaciones
- `transcripcion.json`: `{schema_version:1,language:'es',segments:[{start,end,text}],metadata:{...}}`.
- Corrección de JSON/SRT: exporta `corregido.txt`; **no** modifica las marcas temporales ni reescribe el SRT original.
- XTTS requiere autorización de voz; `--authorized` constituye una declaración del operador, no una verificación biométrica ni legal.
- Las salidas se escriben al directorio indicado; use directorios de salida diferentes entre ejecuciones para evitar sobrescribir resultados.
- Groq y OpenAI aplican límites de tamaño/duración: un archivo WAV largo puede excederlos. Falta añadir partición inteligente en producción.
- El fallback hacia la nube **no es automático**: se debe elegir explícitamente `--engine groq|openai --allow-cloud`.
- Ollama recibe el texto fragmentado de manera aproximada; la corrección no garantiza fidelidad absoluta y debe ser revisada.
- Para textos extensos, XTTS puede necesitar troceado, unión y normalización de audio a nivel de producción.
- Este proyecto no configura ni prueba un motor ASR de audio multimodal directamente bajo Ollama: requiere un adaptador independiente para un modelo con API de audio verificada.
