# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:01:42 2021

@author: Ghozy El Fatih
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as inter
from PIL import Image

#load gambar semblance
path_gambar = "IMAGE_PATH\\"
path_pick = "PICK_PATH\\normpick"
gambar = "*"

img = []
pick1 = []
for i in gambar:
    gg = Image.open(path_gambar+i+".png")
    img.append(gg)
    pp = np.loadtxt("interp_normpick"+i+".txt")
    pp = pp[:,0]
    pick1.append(pp)

pick1 = np.array(pick1)
pick1 = np.reshape(pick1,(10,80)).T

pick2 = np.zeros([80,200])
for i in range(10):
    aa = list(pick1[:,i])*20
    aa = np.array(aa)
    aa = np.reshape(aa,(20,80)).T
    pick2[:,i*20:(i+1)*20] = aa

# sort the coordinate
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

# normalize coordinate from pixel to x = velocity, y = time
  
x_pick,y_pick = selection_sort(pick[:,0]),selection_sort(pick[:,1])
norm_x_pick = ((x_pick-0))/(500-0)*(4000-1000)+1000
norm_y_pick = ((y_pick-0))/(500-0)*(3-0)+0

norm = np.array([norm_x_pick,norm_y_pick]).T

savenorm = np.savetxt('normpick'+gambar+'.txt',norm)

vel = pick2[:,0]
t = pick2[:,1]
t[len(t)-1] = 3
t[0] = 0

y = t
x = vel

# Interpolate the data using a cubic spline to "new_length" samples
new_length = 80
new_y = np.linspace(np.min(y), np.max(y), new_length)
new_x = inter.interp1d(y, x, kind='linear')(new_y)

new = np.array([new_x,new_y]).T
neww = np.savetxt("interp_normpick"+no+".txt",new)

# load rms buat pembanding
path_rms = "C:\\Users\\Acer\\OneDrive - UNIVERSITAS INDONESIA\\Documents\\Kuliah\\Crispy\\velocity rms\\RMS Model 3\\"
rms1 = pd.read_csv(path_rms+"RMS_3_80x200_1.csv",header=None)
rms1 = np.array(rms1)
rms = rms1[:,20]
Y = np.linspace(0,3,80)

error = (abs(rms1-pick2)/rms1)*100
mean_error = np.mean(error)/2

plt.figure()
plt.suptitle('Perbandingan RMS Velocity Map, Error : '+str(round(mean_error,3))+'%',fontsize=20)
plt.subplot(211)
plt.title('RMS Velocity Hasil Pick CNN')
plt.imshow(rms1,extent=[0,10000,3,0],aspect='auto',cmap='jet',interpolation='bicubic')
plt.ylabel('Kedalaman (s)')
plt.subplot(212)
plt.title('RMS Velocity Hasil Perhitungan')
plt.imshow(rms1,extent=[0,10000,3,0],aspect='auto',cmap='jet',interpolation='bicubic')
plt.xlabel('Offset (m)')
