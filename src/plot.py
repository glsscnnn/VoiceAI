import matplotlib.pyplot as plt
import tensorflow as tf
from utility import get_spectrogram
import numpy as np

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

audio_path = 'my_voice.wav'
# audio_path = 'data/mini_speech_commands/no/ac652c60_nohash_0.wav'
audio_bina = tf.io.read_file(audio_path)

waveform = decode_audio(audio_bina)
spectrogram = get_spectrogram(waveform)

print(spectrogram)

# plt.plot(waveform)
# plt.savefig('waveform.png', dpi=300)

plt.plot(spectrogram)
plt.savefig('spec.png', dpi=300)
