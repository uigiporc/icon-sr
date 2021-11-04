import numpy as np
import threading
import tensorflow as tf
import librosa
import queue
import wave
import pyaudio
from tensorflow import keras

FORMAT = pyaudio.paInt16
CHANNELS = 2
p = pyaudio.PyAudio()
buffer = queue.Queue(10)
SAMPLE_RATE = 16000

# The set of characters accepted in the transcription.
characters = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


model = tf.keras.models.load_model('./inference/pretrained_model/', compile=False)


def predict_audio(audio, sr, model):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=int(sr / 40), hop_length=int(sr / 100),
                                                 n_mels=128)
    log_mel_spectrogram = np.log(spectrogram + 1e-14)
    spectrogram = log_mel_spectrogram

    spectrogram = tf.convert_to_tensor(value=spectrogram)
    # Padding and transposing to match the model input shape
    spectrogram = tf.transpose(spectrogram, perm=[1, 0])
    spectrogram = tf.pad(spectrogram, [[0, 3328 - spectrogram.shape[0], ], [0, 0]])
    spectrogram = tf.expand_dims(spectrogram, 1)
    spectrogram = tf.transpose(spectrogram, perm=[1, 0, 2])

    classes = model.predict(spectrogram)
    return decode_batch_predictions(classes)


def producer():
    chunk = 1024
    record_seconds = 3

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=chunk)

    while True:
        frames = []
        for i in range(0, int(SAMPLE_RATE / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
        buffer.put(frames)

    stream.stop_stream()
    stream.close()

    p.terminate()


def consumer():
    while True:
        if not buffer.empty():
            wf = wave.open('temp.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(buffer.get()))
            wf.close()

            audio, sr = librosa.load('temp.wav', sr=SAMPLE_RATE)

            text = predict_audio(audio, sr, model)
            print(text)


prod_th = threading.Thread(target=producer, args=())
consumer_th = threading.Thread(target=consumer, args=())

prod_th.start()
consumer_th.start()
