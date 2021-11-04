# icon-sr
Project repository of Knowledge Engineering. Project name: Automatic Speech Recognition (ASR) building, trainging and inferring on a Recurrent Neural Network for offline automatic subtitles generation. 

Based on:
- Python 3.9
- Tensorflow 2.6.1

# Prerequisites

- Clone the repository:
```
git clone https://github.com/uigiporc/icon-sr.git
```
- Download and extract the dataset from [LibriSpeech](https://www.openslr.org/12)
```
curl.exe -o dataset.zip https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzvf dataset.zip
```
- Install the dependencies in requirements.txt
```
pip3 install -r icon-sr/preprocessing/requirements.txt
```
# Preprocessing
The preproccesing is necessary **only** if you want to train from scratch the model. To do so, edit _DATASET_PATH_ and _PROCCESSED_PATH_ in _preprocessing/preprocessing.py_. Then run:

```
python3 icon-sr/preprocessing/preprocessing.py
```

# Training
To train the model, upload _PROCESSED_PATH_ to a Google Cloud Bucket (for TPU hardware acceleration) or Google Drive (for GPU or no hardware acceleration).
Then, set the paths in the notebook speech_to_text.ipynb, and run on Colab.

With Librispeech-train-clean-360 expect around 60 minutes per Epoch on TPU.

# Inference
## Windows
Install:
```
pip3 install pipwin
pipwin install pyaudio
```
Enable and set Stereo Mix as your default input device. For instructions, see [this](https://www.howtogeek.com/howto/39532/how-to-enable-stereo-mix-in-windows-7-to-record-audio/)

Then run:
```
python3 inference.py
```

## Linux
Untested. Should work without too many changes. 
