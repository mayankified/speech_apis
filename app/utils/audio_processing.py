from pydub import AudioSegment
import io

def convert_to_wav(audio_bytes):
    try:
        # pydub auto-detects format if you don't specify
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {e}")
