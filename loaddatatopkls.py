import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from PIL import Image
import os

from typing import Iterable
def product(l:Iterable)->int:
    x=1
    for i in l:
        x*=i
    return x

datapath = "../fingerprint-to-blood-group/input/dataset_blood_group/"
listdirs = os.listdir(datapath)
fftresults = []
fftshifteds = []
imgs = []
origs =[]
footprints=[]
outs = []
__map={'A+':0,'A-':1,'AB+':2,'AB-':3,'B+':4,'B-':5,'O+':6,'O-':7}
for dir in listdirs:
    filenames = os.listdir(datapath+'/'+dir)
    for index,filename in enumerate(filenames):
        image = Image.open(datapath+'/'+dir+'/'+filename)
        imgarr = np.array(image)
        if len(imgarr.shape)==3:
            imgarr = np.mean(imgarr,axis=2)

        fftresult = np.fft.fft2(imgarr)
        fftshifted = np.fft.fftshift(fftresult)

        if fftresult.shape!=(103,96):continue
        fftresults.append(fftresult)
        fftshifteds.append(fftshifted)
        imgs.append(image)
        outs.append(__map[dir])
        footprint = np.abs(np.sin(np.abs(fftresult)))
        footprints.append(footprint)
        orig  = np.abs(np.fft.ifft2(fftresult))
        origs.append(orig)

print('--saving--')
import pickle
with open('pkls/imgs.pkl','wb') as f:pickle.dump(imgs,f)
del imgs
with open('pkls/fftresults.pkl','wb') as f:pickle.dump(fftresults,f)
del fftresults
with open('pkls/fftshifteds.pkl','wb') as f:pickle.dump(fftshifteds,f)
del fftshifteds
with open('pkls/origs.pkl','wb') as f:pickle.dump(origs,f)
del origs
with open('pkls/outs.pkl','wb') as f:pickle.dump(outs,f)
del outs
print('----------Executed-------------')