"""Proceso ASR aislado: nunca invoca LLM textual."""
import json
import os
import sys
import wave
from pathlib import Path


def load_normalized_audio(audio):
    """Carga el WAV PCM que produce app.main sin depender del decodificador PyAV."""
    import numpy as np

    with wave.open(str(audio), 'rb') as stream:
        audio_format = (
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getframerate(),
            stream.getcomptype(),
        )
        if audio_format != (1, 2, 16000, 'NONE'):
            raise ValueError(
                'El audio temporal debe ser WAV PCM mono de 16 bits a 16000 Hz'
            )
        frames = stream.readframes(stream.getnframes())
    if not frames:
        raise ValueError('El audio temporal está vacío')
    return np.frombuffer(frames, dtype='<i2').astype(np.float32) / 32768.0


def local(audio, model):
    from faster_whisper import WhisperModel
    device = os.getenv('ASR_DEVICE', 'cuda')
    precision = 'int8_float16' if device == 'cuda' else 'int8'
    whisper = WhisperModel(model, device=device, compute_type=precision)
    # Pasar un ndarray evita la ruta decode_audio de faster-whisper 1.2.1,
    # incompatible con av>=19 por el argumento eliminado metadata_errors.
    samples = load_normalized_audio(audio)
    segments, _ = whisper.transcribe(samples, language='es', beam_size=5, vad_filter=True)
    return [{'start': s.start, 'end': s.end, 'text': s.text.strip()} for s in segments], model


def cloud(provider, audio):
    from openai import OpenAI
    if provider == 'groq':
        key = os.environ['GROQ_API_KEY']
        client = OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1')
        model = os.getenv('GROQ_ASR_MODEL', 'whisper-large-v3-turbo')
    else:
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        model = os.getenv('OPENAI_ASR_MODEL', 'whisper-1')
        if model != 'whisper-1':
            raise ValueError('Para SRT se requiere un modelo con verbose_json y segments; use whisper-1')
    # API: límites de carga/tamaño dependen del proveedor y del plan.
    with open(audio, 'rb') as stream:
        response = client.audio.transcriptions.create(
            model=model, file=('audio.wav', stream, 'audio/wav'),
            response_format='verbose_json', language='es')
    segments = getattr(response, 'segments', None)
    if not segments:
        raise RuntimeError('El proveedor no devolvió segmentos temporizados necesarios para SRT')
    output = []
    for s in segments:
        if isinstance(s, dict):
            output.append({'start': s['start'], 'end': s['end'], 'text': s['text']})
        else:
            output.append({'start': s.start, 'end': s.end, 'text': s.text})
    return output, model


if __name__ == '__main__':
    engine, audio, output, model = sys.argv[1:5]
    if engine == 'local':
        segments, used = local(audio, model)
    else:
        segments, used = cloud(engine, audio)
    Path(output).write_text(json.dumps({'segments': segments, 'model': used}, ensure_ascii=False), encoding='utf-8')
