# Music Genre Classification
## Simple music genre classification with PyTorch (version 1.0.1)
## Accuracy ~88%

![Alt text](helper_images/pytorch-logo-dark.png?raw=true "PyTorch")

## The dataset used is GTZAN Genre Collection.
It can be found at http://marsyas.info/downloads/datasets.html

In order to work with audio this project transforms input audio file into MEL-spectrogram. Then this image is fed into CNN. Pretrained ResNet34 was used as classifier's architecture. After that the model was fine-tuned.

## How to use this project:
http://opihi.cs.uvic.ca/sound/genres.tar.gz
    
    1) Download the dataset and place it in main directory:
       The speed is not great =(.
       GTZAN Genre Collection consists of 10 genres, which are:
            - blues
            - classical
            - country
            - disco
            - hiphop
            - jazz
            - metal
            - pop
            - reggae
            - rock
       Each class consists of 100 30 seconds length audio files. The tracks are all 22050Hz 
       Mono 16-bit audio files in .au format.
    2) Unzip the archive into genres folder. So, the audio files will be in 
       music_genre_classification/genres/*genre*/ directory.
    3) Use conversion.py to create 'data' folder which will have same structure as genres folder.
       It should have the same audio files but in wav format.
![Alt text](helper_images/wave.png?raw=true "Waveplot")

    4) Use prepare_dataset utility function declared in audio_preprocess.py to create train 
       and test folders with MEL-spectrograms of wav audio tracks.
       Before creating MEL-spectrogram higher frequencies of audio files were boosted.
       Each audio instance was separated into 3 10 seconds parts. For all of those parts 
       MEL-spectrogram was created. So, now dataset consists of totally 3 * 100 * 10 = 3000 images.
   As mentioned [here](https://arxiv.org/pdf/1804.01149.pdf)
![Alt text](helper_images/preprocess.png?raw=true "Preprocess filter")
![Alt text](helper_images/MEL.png?raw=true "MEL-spectrogram")

    5) Create folder and name it 'mel'. Put train and test folders into it. Also, create 'models' folder.
    6) Open genre_classification.ipynb with Jupyter Notebook. Change Adam's learning rate to 0.0003 
       and lr_scheduler's step_size to 5.
    7) Run all cells, then change the optimizer's parameters back and train once again.
       (You can make batch_size higher if you system allows you)
    8) Put your second model in 'models' folder and name it 'final.pt'
    9) Now with predict.py you can make predictions for new audio files. They should be at least
       30 seconds long and be in .au format.
