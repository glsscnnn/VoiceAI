import os
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models

# useful global alias
AUTOTUNE = tf.data.AUTOTUNE

# downloading data
def download_data():
    data_dir = pathlib.Path('data/mini_speech_commands')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir='.', cache_subdir='data')
    return data_dir

# possible options / labels
def get_commands():
    data_dir = download_data()
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md'] # ignore the README file
    return commands

# decoding audio files
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

# the label for each wav file is it's parent directory
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

# return each waveform with it's label
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

# getting our train test and validation files
def get_files():
    data_dir = download_data()
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)

    train_files = filenames[:6400]

    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    val_files = filenames[6400:6400+800]
    test_files = filenames[-800:]

    return waveform_ds, val_files, test_files

def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
  waveform_ds, _, _ = get_files()

  # Concatenate audio with padding so that all audio clips will be of the
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

def get_spectrogram_and_label_id(audio, label):
    commands = get_commands()
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds
