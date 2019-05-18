import os
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pylab


def random_file(path):
    """
    Returns path to a random file in 'path' variable
    """
    files_amount = len(os.listdir(path)) - 1
    idx = np.random.randint(files_amount)
    return path + '/' + os.listdir(path)[idx]


def random_10_seconds(path):
    """
    Returns random 10 seconds jupyter audio object
    from from a random instance in dataset
    """
    for _ in range(2):
        path = random_file(path)
    print(path)
    audio, sr = librosa.load(path, dtype=np.float16)
    audio_length = audio.shape[0] / sr
    print(audio_length)
    starting_point = round(np.random.uniform(audio_length - 10))
    x = audio[starting_point*sr:(starting_point+10)*sr]
    return ipd.Audio(data=x, rate=sr)


def read_wav_file(path, preprocess=True): 
    """Reads wav file and cuts it to 3 10-seconds long parts
    
    path: path to file
    preprocess: if True, apply: y(t) = x(t) − α ∗ x(t−1);
    where α is set to 0.97 in order to boost higher frequencies
    
    Returns: list of signals as np.arrays and sample rate as int
    """
    y, sr = librosa.load(path)
    if preprocess:
        x = y.copy()
        z = y[1:] - 0.97 * x[:-1]
        y = z
        
    signals = []
    for signal_id in range(3):
        signals.append(z[signal_id*sr*10:(signal_id+1)*sr*10])
    return signals, sr


def create_spectrogram(z, sr, save_path):
    """Saves signal as MEL-spectrogram
    
    z: signal (np.array)
    sr: sample rate (usually 22050)
    save_path: MEL-spectrogram save path
    """
    #matplotlib.use('Agg')    #uncomment this line when in *.py file extension
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    S = librosa.feature.melspectrogram(y=z, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()
    
    
def resize_image(path):    
    """
    Loads image, resizes and saves it to the same path
    """
    SIZE = 224, 224
    im = Image.open(path)
    resized_im = im.resize(SIZE)
    resized_im.save(path)
    
    
def full_preprocess(path, save_path):
    """
    Full preprocess of audio file
    """
    y, sr = read_wav_file(path)
    idx = 0
    for signal_id in y:
        save_to = save_path + '_' + str(idx) + '.jpg'
        create_spectrogram(signal_id, sr, save_to)
        resize_image(save_to)
        idx += 1
        
        
def prepare_dataset(path='./', test_size=0.2):
    """
    Creates spectograms for all audio files in dataset and separates them into
    'train' and 'test' folders
    """
    folders = ('train', 'test')
    for folder in folders:
        if folder not in os.listdir():
            os.mkdir('./' + folder)
    
    genres = os.listdir(path + '/data')
    for genre in genres:
        for folder in folders:
            if genre not in ('./' + folder):
                os.mkdir('./' + folder + '/' + genre)
        audios = os.listdir(path + '/data/' + genre)
        audios_amount = len(audios)
        id_audio = 0
        while id_audio < audios_amount:
            if id_audio < int(test_size * audios_amount):
                folder = './test/'
            else:
                folder = './train/'
            full_preprocess(path + '/data/' + genre + '/' + audios[id_audio], \
                            folder + genre + '/' + audios[id_audio][:-3] + '.jpg')
            id_audio += 1
            