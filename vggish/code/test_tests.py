import soundfile as sf

info = sf.info('../tests/sample_wav_resampled.wav')
print(info.duration)
