import tensorflow as tf
from utility import preprocess_dataset, get_commands, get_spectrogram_and_label_id, AUTOTUNE
import matplotlib.pyplot as plt

# some_file = 'data/mini_speech_commands/no/ac652c60_nohash_0.wav'
some_file = 'my_voice.wav'
sample_ds = preprocess_dataset([str(some_file)])
commands = get_commands()

print(sample_ds.batch(1))

model = tf.keras.models.load_model('models/compiled_model')

print(model.summary())

for spectrogram, label in sample_ds:
    # prediction = model(spectrogram)
    print(spectrogram)
    # plt.bar(commands, tf.nn.softmax(prediction[0]))
    # plt.title(f'Predictions for "{commands[label[0]]}"')
    # plt.savefig('predicted.png', dpi=300)
