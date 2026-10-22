# Transcriptor local: dos flujos independientes

**Flujo A:** MP3/WebM/WAV/MP4/MKV -> FFmpeg -> Faster-Whisper (o ASR API explícita) -> TXT/JSON/SRT -> FIN.

**Flujo B:** TXT/JSON/SRT -> corrección opcional Ollama -> texto corregido independiente -> XTTS-v2 opcional con voz autorizada -> WAV -> FIN.

---

## 1. Instalación en Ubuntu 24.04 / Python 3.12

### 1.1. Dependencias base

```bash
sudo apt update && sudo apt install -y ffmpeg python3-venv
```

Sitúese en el directorio del proyecto:

```bash
cd transcriptor_local
```

### 1.2. Entorno ASR

Cree un entorno virtual independiente para transcripción:

```bash
python3 -m venv .venv-asr
source .venv-asr/bin/activate

python -m pip install -U pip
pip install faster-whisper 'av<20' ollama openai \
    nvidia-cublas-cu12 \
    'nvidia-cudnn-cu12==9.*'
```

Las bibliotecas NVIDIA son necesarias únicamente cuando `faster-whisper` se ejecuta con CTranslate2 sobre una GPU NVIDIA compatible.

Antes de iniciar Python, configure las bibliotecas CUDA cargadas por CTranslate2:

```bash
export LD_LIBRARY_PATH="$(
python -c 'import os,nvidia.cublas.lib,nvidia.cudnn.lib;print(os.path.dirname(nvidia.cublas.lib.__file__)+":"+os.path.dirname(nvidia.cudnn.lib.__file__))'
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Instale el modelo utilizado para corrección de texto:

```bash
ollama pull qwen3:4b-instruct
```

### 1.3. Entorno XTTS

Para XTTS cree un entorno independiente:

```bash
deactivate

python3 -m venv .venv-tts
source .venv-tts/bin/activate

python -m pip install -U pip
```

Instale `torch`, `torchaudio` y, cuando la versión de PyTorch lo requiera, `torchcodec`, seleccionando un backend CUDA compatible según la documentación oficial de PyTorch.

Posteriormente instale Coqui TTS:

```bash
pip install coqui-tts
```

Si utilizará simultáneamente la opción `--correct` dentro del flujo de síntesis, instale también el cliente Python de Ollama:

```bash
pip install ollama
```

---

# 2. Instalación exclusiva para macOS Tahoe 26 / Apple Silicon M1

Esta instalación es específica para:

- **Sistema operativo:** macOS Tahoe 26.
- **Arquitectura:** Apple Silicon ARM64.
- **Procesador:** Apple M1.
- **Python recomendado para este proyecto:** Python 3.12 ARM64.
- **Aceleración disponible:** Apple Metal/MPS para componentes PyTorch que lo soporten.
- **CUDA:** no aplicable.
- **cuDNN:** no aplicable.
- **NVIDIA:** no aplicable.

En Apple Silicon no deben instalarse:

```text
nvidia-cublas-cu12
nvidia-cudnn-cu12
```

Tampoco debe configurarse:

```text
LD_LIBRARY_PATH
```

con bibliotecas NVIDIA.

`faster-whisper` utiliza CTranslate2. Los binarios oficiales de CTranslate2 soportan procesadores AArch64/ARM64, incluidos los Apple Silicon, pero su backend GPU oficial corresponde a NVIDIA/CUDA; por ello, en M1 la configuración convencional de `faster-whisper` debe considerarse una ejecución **CPU ARM64**. Para ARM64, CTranslate2 admite cuantización `int8`, que resulta especialmente conveniente para este escenario.

Por otra parte, PyTorch sí dispone de un backend `mps` que utiliza Metal Performance Shaders en macOS, por lo que componentes compatibles de XTTS/PyTorch pueden aprovechar la GPU integrada del M1.

---

## 2.1. Verificar que la terminal se ejecuta como ARM64

Antes de instalar dependencias ejecute:

```bash
uname -m
```

El resultado esperado es:

```text
arm64
```

Compruebe también:

```bash
arch
```

El resultado esperado debe ser:

```text
arm64
```

Si aparece:

```text
x86_64
```

la terminal, Python o alguna herramienta puede estar ejecutándose mediante Rosetta 2. Para este proyecto se recomienda utilizar versiones ARM64 nativas.

---

## 2.2. Instalar las herramientas de desarrollo de Apple

Instale las Command Line Tools de Xcode:

```bash
xcode-select --install
```

Después puede comprobar la instalación con:

```bash
xcode-select -p
```

Normalmente devolverá una ruta equivalente a:

```text
/Library/Developer/CommandLineTools
```

Verifique también Clang:

```bash
clang --version
```

---

## 2.3. Instalar Homebrew para Apple Silicon

Si Homebrew todavía no está instalado:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

En Apple Silicon, Homebrew normalmente se instala bajo:

```text
/opt/homebrew
```

Si el instalador lo solicita, incorpore Homebrew al entorno:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Compruebe:

```bash
brew --version
```

y:

```bash
which brew
```

El segundo comando debería devolver normalmente:

```text
/opt/homebrew/bin/brew
```

---

## 2.4. Instalar FFmpeg y Python 3.12

Instale las dependencias base:

```bash
brew update
brew install ffmpeg python@3.12
```

Verifique FFmpeg:

```bash
ffmpeg -version
```

Verifique Python:

```bash
/opt/homebrew/bin/python3.12 --version
```

Deberá obtener una versión de Python 3.12.

También conviene comprobar que Python es ARM64:

```bash
/opt/homebrew/bin/python3.12 -c \
"import platform; print(platform.machine())"
```

Resultado esperado:

```text
arm64
```

Aunque `faster-whisper` utiliza PyAV y no depende obligatoriamente de un ejecutable FFmpeg del sistema para la decodificación básica, este proyecto mantiene FFmpeg como dependencia explícita porque los flujos de normalización, conversión, extracción y preparación de audio pueden utilizarlo.

---

## 2.5. Crear el entorno ASR para macOS M1

Sitúese en el proyecto:

```bash
cd transcriptor_local
```

Cree el entorno:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-asr
source .venv-asr/bin/activate
```

Verifique:

```bash
python --version
python -c "import platform; print(platform.machine())"
```

Los resultados esperados son equivalentes a:

```text
Python 3.12.x
arm64
```

Actualice las herramientas de instalación:

```bash
python -m pip install -U pip setuptools wheel
```

Instale las dependencias del flujo ASR:

```bash
pip install faster-whisper 'av<20' ollama openai
```

No instale los paquetes NVIDIA utilizados por Ubuntu:

```text
nvidia-cublas-cu12
nvidia-cudnn-cu12
```

---

## 2.6. Verificar CTranslate2 en Apple Silicon

Compruebe que CTranslate2 puede cargarse:

```bash
python -c "
import ctranslate2
print('CTranslate2:', ctranslate2.__version__)
print('CPU compute types:', ctranslate2.get_supported_compute_types('cpu'))
"
```

En Apple Silicon el conjunto concreto puede depender de la versión instalada, pero `int8` es un modo apropiado para ARM64.

Para obtener información adicional de diagnóstico:

```bash
export CT2_VERBOSE=1
```

Después puede ejecutar la aplicación normalmente.

Para eliminar posteriormente el diagnóstico:

```bash
unset CT2_VERBOSE
```

### Configuración recomendada de Faster-Whisper en M1

La configuración recomendada es conceptualmente:

```python
WhisperModel(
    model_size,
    device="cpu",
    compute_type="int8",
)
```

Es decir:

```text
device = cpu
compute_type = int8
```

No debe configurarse:

```text
device = cuda
```

porque Apple M1 no dispone de CUDA.

Tampoco debe asumirse que:

```text
device = mps
```

funcione con Faster-Whisper/CTranslate2, ya que MPS pertenece al backend de PyTorch y no constituye el backend GPU oficial de CTranslate2.

---

## 2.7. Instalar Ollama nativamente en macOS M1

Ollama puede instalarse en macOS mediante su instalador oficial.

Desde terminal:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Compruebe:

```bash
ollama --version
```

Después descargue el modelo utilizado por el proyecto:

```bash
ollama pull qwen3:4b-instruct
```

Compruebe que está disponible:

```bash
ollama list
```

Puede realizar una prueba independiente:

```bash
ollama run qwen3:4b-instruct
```

y finalizar la sesión con:

```text
/bye
```

En Apple Silicon, Ollama dispone de optimizaciones específicas para la plataforma de Apple y las versiones recientes incorporan soporte basado en MLX para aprovechar la arquitectura de memoria unificada.

Por ello, **no debe configurarse CUDA para Ollama en un Mac M1**.

---

## 2.8. Crear un entorno separado para XTTS-v2 en macOS

Salga primero del entorno ASR:

```bash
deactivate
```

Cree el entorno TTS:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-tts
source .venv-tts/bin/activate
```

Actualice pip:

```bash
python -m pip install -U pip setuptools wheel
```

Instale PyTorch y las dependencias de audio:

```bash
pip install torch torchaudio
```

Cuando la versión de PyTorch instalada requiera `torchcodec`, añádalo:

```bash
pip install torchcodec
```

Después instale Coqui TTS:

```bash
pip install coqui-tts
```

Si utilizará también:

```text
--correct
```

desde este mismo entorno, instale:

```bash
pip install ollama
```

---

## 2.9. Verificar PyTorch y MPS en el Mac M1

Ejecute:

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS compilado:', torch.backends.mps.is_built())
print('MPS disponible:', torch.backends.mps.is_available())
"
```

En una instalación nativa ARM64 compatible debe esperarse:

```text
MPS compilado: True
MPS disponible: True
```

Puede realizar además una prueba elemental en la GPU Apple:

```bash
python - <<'PY'
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("MPS no está disponible")

device = torch.device("mps")
x = torch.ones(5, device=device)
y = x * 2

print("Dispositivo:", y.device)
print("Resultado:", y)
PY
```

El dispositivo esperado será:

```text
mps:0
```

### Importante sobre XTTS y MPS

Que PyTorch detecte correctamente:

```text
mps
```

no implica que absolutamente todas las operaciones utilizadas por XTTS sean compatibles o estén optimizadas para MPS.

Por lo tanto, el proyecto debe permitir:

1. utilizar `mps` cuando el modelo y sus operaciones sean compatibles;
2. volver a CPU cuando una operación no tenga implementación compatible;
3. no asumir que el comportamiento de CUDA y MPS es idéntico.

Durante la instalación inicial conviene validar el modelo XTTS completo con una síntesis pequeña antes de procesar textos extensos.

---

## 2.10. Fallback de operaciones MPS

Si una versión concreta de XTTS/PyTorch utiliza operaciones que todavía no estén implementadas en MPS, puede resultar útil probar:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Esto permite que determinadas operaciones no implementadas en MPS se ejecuten mediante CPU.

Después inicie la aplicación desde la misma terminal.

Para volver al comportamiento normal:

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
```

Esta variable debe considerarse una medida de compatibilidad y no una garantía de aceleración integral del modelo.

---

## 2.11. Verificación completa del entorno ASR en macOS

Active:

```bash
source .venv-asr/bin/activate
```

y ejecute:

```bash
python - <<'PY'
import platform
import av
import ctranslate2
import faster_whisper

print("Arquitectura:", platform.machine())
print("CTranslate2:", ctranslate2.__version__)
print("CPU:", ctranslate2.get_supported_compute_types("cpu"))
print("PyAV:", av.__version__)
print("faster-whisper disponible")
PY
```

Debe confirmarse como mínimo:

```text
Arquitectura: arm64
```

y que no se produzcan errores importando:

```text
av
ctranslate2
faster_whisper
```

---

## 2.12. Verificación completa del entorno TTS en macOS

Active:

```bash
deactivate 2>/dev/null || true
source .venv-tts/bin/activate
```

y ejecute:

```bash
python - <<'PY'
import platform
import torch

print("Arquitectura:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS compilado:", torch.backends.mps.is_built())
print("MPS disponible:", torch.backends.mps.is_available())
PY
```

La arquitectura debe ser:

```text
arm64
```

---

## 2.13. Configuración del proyecto en macOS

Desde la raíz:

```bash
cp .env.example .env
```

Edite el archivo si necesita utilizar servicios externos:

```bash
nano .env
```

No almacene credenciales dentro del código ni incorpore `.env` al control de versiones.

Cargue las variables:

```bash
set -a
source .env
set +a
```

---

## 2.14. Ejecución del Flujo A en macOS M1

Active el entorno ASR:

```bash
source .venv-asr/bin/activate
```

Transcripción local:

```bash
python -m app.main transcribe entrada.mp4 \
    --output outputs/transcripcion1
```

Otro ejemplo:

```bash
python -m app.main transcribe entrada.webm \
    --output outputs/transcripcion2
```

Si se utiliza explícitamente un proveedor externo:

```bash
python -m app.main transcribe entrada.webm \
    --output outputs/transcripcion2 \
    --engine groq \
    --allow-cloud
```

Al terminar:

```bash
deactivate
```

---

## 2.15. Ejecución del Flujo B en macOS M1

Active el entorno TTS:

```bash
source .venv-tts/bin/activate
```

### Solamente corrección

```bash
python -m app.main process \
    outputs/transcripcion1/transcripcion.srt \
    --output outputs/texto1 \
    --correct
```

### Solamente generación de voz

```bash
python -m app.main process \
    original.txt \
    --output outputs/voz1 \
    --voice \
    --reference mi_voz_autorizada.wav \
    --authorized
```

### Corrección y posteriormente generación de voz

```bash
python -m app.main process \
    original.txt \
    --output outputs/voz2 \
    --correct \
    --voice \
    --reference mi_voz_autorizada.wav \
    --authorized
```

---

## 2.16. Ejecución de pruebas en macOS

Utilice el entorno correspondiente a las dependencias verificadas por las pruebas.

Por ejemplo:

```bash
source .venv-asr/bin/activate
python -m unittest discover -s tests -v
```

Si existen pruebas específicas para XTTS:

```bash
deactivate
source .venv-tts/bin/activate
python -m unittest discover -s tests -v
```

---

# 3. Uso de los entornos virtuales

La aplicación mantiene separados los componentes ASR y TTS para evitar conflictos entre dependencias.

## Para `transcribe`

### Ubuntu

```bash
source .venv-asr/bin/activate
```

### macOS Tahoe 26 / M1

```bash
source .venv-asr/bin/activate
```

Después:

```bash
python -m app.main transcribe entrada.mp4 \
    --output outputs/transcripcion1
```

---

## Para `process --voice`

### Ubuntu o macOS

```bash
source .venv-tts/bin/activate
```

Después:

```bash
python -m app.main process original.txt \
    --output outputs/voz1 \
    --voice \
    --reference mi_voz_autorizada.wav \
    --authorized
```

Si también se utilizará:

```text
--correct
```

el paquete Python:

```text
ollama
```

debe encontrarse instalado en `.venv-tts`.

---

# 4. Diferencias importantes entre Ubuntu/NVIDIA y macOS M1

| Característica | Ubuntu + NVIDIA | macOS Tahoe 26 + M1 |
|---|---|---|
| Arquitectura habitual | x86_64 | ARM64 |
| CUDA | Sí | No |
| cuDNN | Sí | No |
| `nvidia-cublas-cu12` | Sí | No |
| `nvidia-cudnn-cu12` | Sí | No |
| `LD_LIBRARY_PATH` CUDA | Sí | No |
| Faster-Whisper | CTranslate2 CPU/CUDA | CTranslate2 CPU ARM64 |
| Faster-Whisper con MPS | No | No mediante backend oficial CTranslate2 |
| Compute recomendado Faster-Whisper | `float16`/`int8_float16` en CUDA según GPU | `int8` CPU |
| PyTorch GPU | CUDA | MPS/Metal |
| XTTS-v2 | PyTorch CUDA/CPU | PyTorch MPS/CPU según compatibilidad |
| Ollama | CPU/NVIDIA según equipo | Apple Silicon optimizado |
| FFmpeg | `apt` | Homebrew |
| Python 3.12 | sistema/apt | Homebrew ARM64 |

Esta diferencia es fundamental: **no deben copiarse las configuraciones CUDA de Ubuntu al entorno macOS M1**.

---

# 5. Configuración del proyecto

Copie el archivo de ejemplo:

```bash
cp .env.example .env
```

Opcionalmente edite `.env` con las credenciales requeridas:

```bash
nano .env
```

Cargue las variables:

```bash
set -a
source .env
set +a
```

No almacene claves, tokens ni credenciales directamente en código fuente ni las incorpore al repositorio.

---

# 6. Ejemplos de ejecución

## Ubuntu

Ejemplo con una ruta absoluta:

```bash
python3.12 -m app.main transcribe \
    /home/sergio/Vídeos/Kooha/parte_exacta_000.mp3 \
    --output \
    /home/sergio/Vídeos/Kooha/outputs/transcripcion1
```

Ejemplos genéricos:

```bash
python -m app.main transcribe entrada.mp4 \
    --output outputs/transcripcion1

python -m app.main transcribe entrada.webm \
    --output outputs/transcripcion2 \
    --engine groq \
    --allow-cloud

python -m app.main process \
    outputs/transcripcion1/transcripcion.srt \
    --output outputs/texto1 \
    --correct

python -m app.main process original.txt \
    --output outputs/voz1 \
    --voice \
    --reference mi_voz_autorizada.wav \
    --authorized

python -m app.main process original.txt \
    --output outputs/voz2 \
    --correct \
    --voice \
    --reference mi_voz_autorizada.wav \
    --authorized
```

## macOS Tahoe 26 / M1

Los comandos funcionales son los mismos; cambia principalmente la instalación y el backend utilizado.

```bash
python -m app.main transcribe entrada.mp4 \
    --output outputs/transcripcion1

python -m app.main process \
    outputs/transcripcion1/transcripcion.srt \
    --output outputs/texto1 \
    --correct

python -m app.main process original.txt \
    --output outputs/voz1 \
    --voice \
    --reference mi_voz_autorizada.wav \
    --authorized
```

Ejecute finalmente las pruebas:

```bash
python -m unittest discover -s tests -v
```

---

# 7. Contratos y limitaciones

- `transcripcion.json` utiliza el contrato:

```text
{
  schema_version: 1,
  language: 'es',
  segments: [
    {
      start,
      end,
      text
    }
  ],
  metadata: {
    ...
  }
}
```

- La corrección de JSON/SRT exporta `corregido.txt`; **no modifica las marcas temporales ni reescribe el SRT original**.

- XTTS requiere autorización de voz. La opción `--authorized` constituye una declaración del operador y no una verificación biométrica, contractual ni legal.

- Las salidas se escriben al directorio indicado. Utilice directorios diferentes entre ejecuciones cuando deba preservar resultados previos y evitar sobrescrituras.

- Groq y OpenAI pueden aplicar límites de tamaño, duración y formato. Un archivo WAV de larga duración puede superar dichos límites. La partición inteligente de audio debe incorporarse antes de considerar el flujo cloud listo para producción.

- El fallback hacia la nube **no es automático**. Debe seleccionarse explícitamente:

```text
--engine groq --allow-cloud
```

o:

```text
--engine openai --allow-cloud
```

- Ollama recibe el texto fragmentado aproximadamente. La corrección lingüística no garantiza fidelidad absoluta respecto del texto transcrito y el resultado debe revisarse cuando su contenido sea crítico.

- Para textos extensos, XTTS puede requerir segmentación, generación por bloques, unión, normalización y posprocesamiento de audio antes de utilizarse en producción.

- Este proyecto no configura ni prueba un motor ASR de audio multimodal directamente bajo Ollama. Un modelo de ese tipo requiere un adaptador independiente y una API de entrada de audio explícitamente verificada.

- En macOS M1, `faster-whisper` no debe configurarse como si MPS fuese un dispositivo CTranslate2. El backend convencional debe utilizar:

```text
device="cpu"
compute_type="int8"
```

- En macOS M1, PyTorch puede utilizar:

```text
device="mps"
```

cuando el modelo y las operaciones involucradas lo admitan.

- La disponibilidad de MPS debe verificarse en tiempo de ejecución mediante:

```python
torch.backends.mps.is_available()
```

y no inferirse únicamente porque el equipo utilice un procesador Apple Silicon.

- La descarga inicial del modelo XTTS y la aceptación de su licencia pueden requerir una ejecución interactiva.

---

# 8. Diagnóstico rápido de macOS M1

Si la instalación en macOS presenta problemas, ejecute estas comprobaciones antes de modificar dependencias.

## Arquitectura

```bash
uname -m
which python
python -c "import platform; print(platform.machine())"
```

Debe aparecer:

```text
arm64
```

## Homebrew

```bash
which brew
brew --prefix
```

Normalmente:

```text
/opt/homebrew/bin/brew
/opt/homebrew
```

## FFmpeg

```bash
which ffmpeg
ffmpeg -version
```

## Faster-Whisper

```bash
source .venv-asr/bin/activate

python -c "
import ctranslate2
import faster_whisper
print('CTranslate2:', ctranslate2.__version__)
print(ctranslate2.get_supported_compute_types('cpu'))
"
```

## Ollama

```bash
ollama --version
ollama list
```

## PyTorch/MPS

```bash
source .venv-tts/bin/activate

python -c "
import torch
print(torch.__version__)
print('MPS:', torch.backends.mps.is_available())
"
```

Esta separación permite determinar si el fallo pertenece a la arquitectura ARM64, Faster-Whisper/CTranslate2, Ollama, PyTorch/MPS o XTTS antes de modificar la aplicación.

Hay tres cambios técnicos especialmente importantes respecto del texto original. Primero, en **M1 no se instalan CUDA, cuBLAS ni cuDNN**; CTranslate2 reconoce ARM64 y su documentación oficial contempla `int8` para AArch64/Apple, mientras que su soporte GPU precompilado está orientado a NVIDIA. [opennmt.net](https://opennmt.net/CTranslate2/hardware_support.html?utm_source=chatgpt.com) Segundo, **MPS corresponde a PyTorch**, por lo que puede beneficiar a XTTS cuando sus operaciones sean compatibles, pero no debe utilizarse como sustituto ficticio de `cuda` en Faster-Whisper. [docs.pytorch.org](https://docs.pytorch.org/docs/main/notes/mps.html?utm_source=chatgpt.com) Tercero, Coqui TTS actualmente documenta Python `>=3.10,<3.15`, PyTorch 2.2+ y compatibilidad esperada con macOS, de modo que mantener **Python 3.12** resulta razonable para compartir una base homogénea entre Ubuntu y macOS. [coqui-tts.readthedocs.io](https://coqui-tts.readthedocs.io/en/latest/installation.html?utm_source=chatgpt.com)

Además, Ollama dispone actualmente de soporte específico para Apple Silicon e introdujo en 2026 una implementación basada en **MLX**, por lo que el M1 tiene un camino de aceleración nativo para el flujo de corrección sin requerir CUDA. [Ollama](https://ollama.com/blog/mlx?utm_source=chatgpt.com)
