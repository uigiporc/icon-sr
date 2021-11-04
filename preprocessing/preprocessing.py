import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import time
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Tested on https://www.openslr.org/resources/12/train-clean-360.tar.gz
# Download and extract before using

DATASET_PATH = '../datasets/test-clean/'
PROCESSED_PATH = 'D:/test_clean_preprocessed/'
SAMPLE_RATE = 16000
EPSILON = 1e-14


def generate_dataset(dataset_path, destination_path):

    file_count = 0
    for root2, dirs2, files2 in os.walk(dataset_path):
        for name2 in files2:
            if name2.endswith('.flac'):
                file_count = file_count + 1

    print(f"Number of examples: {file_count}")

    # to improve data pipelining and I/O parallelization, see TFRecord docs
    number_of_hosts = 1
    number_of_tfrecords = 10 * number_of_hosts
    files_per_tfrecords = int(file_count // number_of_tfrecords)+1

    print(f"Examples per file: {files_per_tfrecords}")
    i = 0
    tfrecord_writer = None

    max_duration = 0
    max_timesteps = 0
    max_duration_filename = "x"
    max_timesteps_filename = "x"
    max_label = 0
    dropped = 0

    # The set of characters accepted in the transcription.
    characters = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith('.txt'):

                f = open(os.path.join(root, name))

                while True:

                    text = f.readline()
                    if text == "":
                        break

                    if i % files_per_tfrecords == 0:
                        if tfrecord_writer is not None:
                            tfrecord_writer.close()
                        print(f"i: {i}, i/tfrecord: {str(int(i // files_per_tfrecords))}")
                        tfrecord_writer = tf.io.TFRecordWriter(os.path.join(destination_path, f"librispeech-test-clean.tfrecord-{int(i // files_per_tfrecords):05d}-of-{int(number_of_tfrecords):05d}"))

                    i = i + 1
                    split_text = text.split(" ", 1)
                    path = os.path.join(root, split_text[0] + '.flac')

                    # librosa resamples at 22050, we want to use native sampling of 16kHz
                    signal, sr = librosa.load(path=path, sr=None)


                    duration = librosa.get_duration(filename=path)
                    if duration > 33.0:
                        dropped += 1
                        continue

                    if duration > max_duration:
                        max_duration = duration
                        max_duration_filename = path

                    if len(split_text) < 2:
                        # if the audio file does not have a transcript, we assume it's a long silence
                        split_text[1] = " "
                    if split_text[1].endswith('\n'):
                        split_text[1] = split_text[1][:-1]  # remove "\n"


                    # 25 ms window, 10 ms stride, 128 dimension filterbank as in ContextNet paper (useful for CNNs)
                    spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=int(sr / 40), hop_length=int(sr / 100), n_mels=128)

                    log_mel_spectrogram = np.log(spectrogram + EPSILON)

                    spectrogram = log_mel_spectrogram


                    text = tf.strings.unicode_split(split_text[1], input_encoding="UTF-8")
                    text = char_to_num(text)

                    if len(text) > max_label:
                        max_label = len(text)

                    # Possible approaches to store a spectrogram:
                    # (Expensive) Flatten to 1D array: https://github.com/DarshanDeshpande/tfrecord-generator/blob/main/audio/spectrogram_to_tfrecords.py
                    # (Untested) Convert to multiple (80) 1D features: https://titanwolf.org/Network/Articles/Article?AID=12bee64b-d71b-4aab-af82-0c0ef51a4385#gsc.tab=0
                    # (SOTA in tf2) Convert tensor to bytestring and restore it:
                    #   (see last comment) https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
                    #   (documentation, see Note) https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample
                    timesteps = len(spectrogram[0])
                    if timesteps > max_timesteps:
                        max_timesteps = timesteps
                        max_timesteps_filename = path

                    serialized_spectrogram = tf.io.serialize_tensor(spectrogram.T)
                    serialized_transcript = tf.io.serialize_tensor(text)

                    feats = {
                        # see tf.io.serialize_tensor documentation
                        'spectrogram': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[serialized_spectrogram.numpy()])),

                        'text': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[serialized_transcript.numpy()])
                        )
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString()
                    tfrecord_writer.write(example)

    print(f"Max duration: {max_duration}, name: {max_duration_filename}")
    print(f"Max timesteps: {max_timesteps}, name: {max_timesteps_filename}")
    print(f"Max label: {max_label}")
    print(f"Dropped: {dropped}")

def show_spectrogram(spectrogram, sr=16000):
    fig, ax = plt.subplots()
    print(len(spectrogram.T))
    img = librosa.display.specshow(spectrogram.T, x_axis='mel', y_axis='time', sr=sr, hop_length=(sr/100), ax=ax)
    plt.show()

def parse_tfr_audio_element(element):
    data = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        'text': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    spectrogram = content['spectrogram']
    transcript = content['text']

    spectrogram = tf.io.parse_tensor(spectrogram, tf.float32) # note that we change the data type to float32
    text = tf.io.parse_tensor(transcript, tf.int32)

    return spectrogram, text


if __name__ == '__main__':
    start_time = time.time()
    generate_dataset(DATASET_PATH, PROCESSED_PATH)
    end_time = time.time()
    print(f"Started at {start_time} and ended at {end_time}")
