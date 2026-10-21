"""Proceso separado para descargar XTTS/torch al salir."""
import sys
from pathlib import Path


def main():
    from TTS.api import TTS
    import torch
    source, reference, target = sys.argv[1:4]
    text = Path(source).read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError('No se puede sintetizar texto vacío')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
    tts.tts_to_file(text=text, speaker_wav=reference,
                    language='es', file_path=target, split_sentences=True)


if __name__ == '__main__':
    main()
