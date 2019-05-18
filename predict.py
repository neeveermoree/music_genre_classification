import sys
import os
from PIL import Image
import numpy as np
import audio_preprocessing as aupr

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

LABELS = sorted(os.listdir('./data/'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file = sys.argv[1]
os.system('sox '+ str(file) + ' ' + str(file[:-3]) + '.wav')

aupr.full_preprocess(str(file[:-3]) + '.wav', str(file[:-3]))

im = Image.open(str(file[:-3]) + '_0.jpg')

os.system('rm ' + str(file[:-3]) + '_0.jpg ' + str(file[:-3]) + '_1.jpg ' + str(file[:-3]) + '_2.jpg ' + \
str(file[:-3]) + '.wav')

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

im = transform_pipeline(im)
im = im.unsqueeze(0)
im = Variable(im)
im = im.to(device)
model = torch.load('./models/final.pt', map_location=device)
model.eval()
pred = model(im)
pred = pred.data.cpu().numpy().argmax()

print('The prediction is: \t', LABELS[pred])
