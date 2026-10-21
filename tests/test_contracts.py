import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from app.main import asr_worker_env, export_transcript, read_text, srt_time
from app.worker_asr import load_normalized_audio


class Contracts(unittest.TestCase):
    def test_srt_time(self):
        self.assertEqual(srt_time(3661.234), '01:01:01,234')

    def test_export_and_import(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            export_transcript([{'start': 0, 'end': 1.2, 'text': 'Hola.'}], p)
            self.assertEqual(read_text(p / 'transcripcion.txt'), 'Hola.')
            self.assertEqual(read_text(p / 'transcripcion.srt'), 'Hola.')
            self.assertEqual(read_text(p / 'transcripcion.json'), 'Hola.')
            self.assertEqual(json.loads((p / 'transcripcion.json').read_text())['schema_version'], 1)

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'bad.json'
            p.write_text('{"other": "test"}', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_text(p)

    def test_invalid_timing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                export_transcript([{'start': 2, 'end': 1, 'text': 'x'}], Path(td))

    def test_load_normalized_audio(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'audio.wav'
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(struct.pack('<hhh', -32768, 0, 32767))
            samples = load_normalized_audio(path)
            self.assertEqual(samples.dtype.name, 'float32')
            self.assertEqual(samples.tolist(), [-1.0, 0.0, 32767 / 32768])

    def test_reject_non_normalized_audio(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'audio.wav'
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(44100)
                stream.writeframes(struct.pack('<hhhh', 0, 0, 0, 0))
            with self.assertRaisesRegex(ValueError, 'WAV PCM mono'):
                load_normalized_audio(path)

    def test_asr_worker_env_includes_cuda_libraries(self):
        env = asr_worker_env()
        paths = env['LD_LIBRARY_PATH'].split(':')
        self.assertTrue(any(path.endswith('/nvidia/cublas/lib') for path in paths))
        self.assertTrue(any(path.endswith('/nvidia/cudnn/lib') for path in paths))


if __name__ == '__main__':
    unittest.main()
