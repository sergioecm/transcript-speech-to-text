"""Dos flujos independientes: media->TXT/JSON/SRT y texto->(Ollama)->(XTTS)."""
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MEDIA = {'.mp3', '.webm', '.wav', '.mp4', '.mkv'}
TEXT = {'.txt', '.json', '.srt'}


def require_file(path: Path, extensions: set[str]):
    if not path.is_file() or path.suffix.lower() not in extensions:
        raise ValueError(f'Archivo inexistente o extensión no permitida: {path}')


def run(cmd: list[str], env=None):
    subprocess.run(cmd, check=True, env=env)


def asr_worker_env():
    """Prepara las bibliotecas CUDA instaladas por pip antes de crear el worker."""
    env = os.environ.copy()
    if env.get('ASR_DEVICE', 'cuda') != 'cuda':
        return env
    libraries = []
    required = (
        ('nvidia.cublas.lib', 'libcublas.so.12'),
        ('nvidia.cudnn.lib', 'libcudnn.so.9'),
    )
    for module_name, filename in required:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ValueError(
                f'Falta {module_name}; reinstale las dependencias de requirements.txt'
            ) from exc
        directories = [Path(path).resolve() for path in module.__path__]
        directory = next((path for path in directories if (path / filename).is_file()), None)
        if directory is None:
            searched = ', '.join(str(path / filename) for path in directories)
            raise ValueError(f'No se encontró la biblioteca CUDA requerida; se buscó en: {searched}')
        libraries.append(str(directory))
    current = env.get('LD_LIBRARY_PATH')
    if current:
        libraries.append(current)
    env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(libraries))
    return env


def normalize(source: Path, wav: Path):
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
         '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)])


def srt_time(sec: float):
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'


def validate_segments(segments):
    output = []
    for seg in segments:
        start, end = float(seg['start']), float(seg['end'])
        if start < 0 or end < start:
            raise ValueError('Segmento con tiempo inválido')
        output.append({'start': start, 'end': end, 'text': str(seg['text']).strip()})
    return output


def export_transcript(segments, destination: Path, metadata=None):
    segments = validate_segments(segments)
    destination.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(x['text'] for x in segments)
    (destination / 'transcripcion.txt').write_text(text + ('\n' if text else ''), encoding='utf-8')
    data = {'schema_version': 1, 'language': 'es', 'segments': segments,
            'metadata': metadata or {}}
    (destination / 'transcripcion.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    blocks = [f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}"
              for i, s in enumerate(segments, 1)]
    (destination / 'transcripcion.srt').write_text('\n\n'.join(blocks) + ('\n' if blocks else ''), encoding='utf-8')


def read_text(path: Path):
    require_file(path, TEXT)
    content = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.txt':
        return content.strip()
    if path.suffix.lower() == '.json':
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get('segments'), list):
            return '\n'.join(str(s['text']) for s in obj['segments'] if isinstance(s, dict) and 'text' in s)
        if isinstance(obj, dict) and isinstance(obj.get('text'), str):
            return obj['text'].strip()
        raise ValueError('JSON no compatible: requiere segments[].text o text')
    # SRT: reconoce encabezado temporal; evita borrar texto legítimo numérico.
    blocks = re.split(r'\n\s*\n', content.replace('\r\n', '\n').strip())
    extracted = []
    for block in blocks:
        lines = block.strip().splitlines()
        marker = next((i for i, ln in enumerate(lines) if '-->' in ln), None)
        if marker is None:
            raise ValueError('Bloque SRT sin marca temporal')
        extracted.extend(lines[marker + 1:])
    return '\n'.join(extracted).strip()


def ollama_correct(text, model: str):
    from ollama import Client
    client = Client(host=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'))
    instructions = ('Corrige exclusivamente ortografía, puntuación y mayúsculas. '
                    'Conserva nombres, números, sentido, palabras, párrafos e idioma. '
                    'No inventes, resumas ni agregues contenido. Devuelve solo el texto.')
    # Troceado por párrafos, con límite aproximado de caracteres; no sustituye tokenización.
    chunks = []
    current = ''
    for paragraph in text.splitlines(keepends=True):
        if current and len(current) + len(paragraph) > 3500:
            chunks.append(current)
            current = ''
        if len(paragraph) > 3500:
            for j in range(0, len(paragraph), 3500):
                piece = paragraph[j:j+3500]
                if current:
                    chunks.append(current)
                    current = ''
                chunks.append(piece)
        else:
            current += paragraph
    if current:
        chunks.append(current)
    results = []
    try:
        for chunk in chunks:
            answer = client.chat(model=model, messages=[
                {'role': 'system', 'content': instructions},
                {'role': 'user', 'content': chunk},
            ], options={'temperature': 0}, keep_alive='0s')
            results.append(answer.message.content.strip())
    finally:
        # Ollama descargará el modelo después de la última solicitud.
        try:
            client.generate(model=model, prompt='', keep_alive=0)
        except Exception:
            pass
    return '\n'.join(results)


def media_flow(args):
    require_file(args.input, MEDIA)
    args.output.mkdir(parents=True, exist_ok=True)
    audio = args.output / '_audio_temporal.wav'
    try:
        normalize(args.input, audio)
        # Los motores se ejecutan fuera del proceso principal y liberan su VRAM al salir.
        if args.engine == 'local':
            run([sys.executable, '-m', 'app.worker_asr', 'local', str(audio),
                 str(args.output / '_segments.json'), args.whisper_model],
                env=asr_worker_env())
        elif args.engine in ('groq', 'openai'):
            if not args.allow_cloud:
                raise ValueError('Enviar audio a una API requiere --allow-cloud')
            run([sys.executable, '-m', 'app.worker_asr', args.engine, str(audio),
                 str(args.output / '_segments.json'), args.whisper_model])
        else:
            raise ValueError('Motor ASR no soportado')
        payload = json.loads((args.output / '_segments.json').read_text(encoding='utf-8'))
        if not payload['segments']:
            raise RuntimeError('El motor no produjo segmentos: no se exporta transcripción vacía')
        export_transcript(payload['segments'], args.output,
                          {'engine': args.engine, 'model': payload['model'], 'source': args.input.name})
        print(f'Transcripción exportada a: {args.output}')
    finally:
        audio.unlink(missing_ok=True)
        (args.output / '_segments.json').unlink(missing_ok=True)


def text_flow(args):
    text = read_text(args.input)
    if not text:
        raise ValueError('El archivo no contiene texto')
    args.output.mkdir(parents=True, exist_ok=True)
    # Nunca modifica el archivo de entrada.
    (args.output / 'original.txt').write_text(text + '\n', encoding='utf-8')
    selected = text
    if args.correct:
        selected = ollama_correct(text, args.ollama_model)
        (args.output / 'corregido.txt').write_text(selected + '\n', encoding='utf-8')
    if args.voice:
        if not args.authorized:
            raise ValueError('Se requiere --authorized para confirmar la autorización de clonación')
        if args.reference is None or not args.reference.is_file():
            raise ValueError('Se requiere --reference con un archivo de voz existente')
        source = args.output / '_texto_tts.txt'
        try:
            source.write_text(selected, encoding='utf-8')
            run([sys.executable, '-m', 'app.worker_tts', str(source), str(args.reference),
                 str(args.output / 'sintetizado.wav')])
        finally:
            source.unlink(missing_ok=True)
    print(f'Procesamiento finalizado: {args.output}')


def main():
    parser = argparse.ArgumentParser(description='Transcripción y síntesis local de voz')
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('transcribe', help='Flujo A: multimedia a TXT/JSON/SRT')
    a.add_argument('input', type=Path)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--engine', choices=['local', 'groq', 'openai'], default='local')
    a.add_argument('--whisper-model', default='medium')
    a.add_argument('--allow-cloud', action='store_true')
    b = sub.add_parser('process', help='Flujo B: texto a corrección y/o WAV')
    b.add_argument('input', type=Path)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--correct', action='store_true')
    b.add_argument('--ollama-model', default='qwen3:4b-instruct')
    b.add_argument('--voice', action='store_true')
    b.add_argument('--reference', type=Path)
    b.add_argument('--authorized', action='store_true')
    args = parser.parse_args()
    try:
        (media_flow if args.command == 'transcribe' else text_flow)(args)
    except (ValueError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
