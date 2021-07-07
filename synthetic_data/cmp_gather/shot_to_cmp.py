# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 19:42:16 2020

@author: Ghozy El Fatih
"""
import time
t0 = time.process_time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model = '1'
layer = '8'
percobaan = '1'

### Shot Gather to CMP Gather

data = pd.read_csv('shot_{}_{}_{}.csv'.format(model,layer,percobaan),header=None)
data = np.array(data) # Shot Gather with 100 shots

a = np.arange(0,19801,200)
c = []
for i in range(len(a)):
    b = a[i]+i-1
    c.append(b)
c = np.array(c)

d = []
for j in a:
    data[:,j+2:j+198] = np.fliplr(data[:,j+2:j+198])
    e = data[:,j+2:j+198]
    d.append(e)
d = np.array(d)

cdp = np.arange(0,100,1)
for i in cdp:
    dd = d[:,i]

zero = np.zeros([150100,296])

f = []
for i in range(100):
    zero[i*1501:(i+1)*1501,i:i+196] = d[i]
    f.append(zero)

f = np.array(f)
ff = np.reshape(f[0],(100,1501,296))

zero2 = np.zeros([1501,10000])
for i in range(100):
    zero2[:,i*100:(i*100)+100] = ff[:,:,i+99].T
    zero2[:,i*100:(i*100)+100] = np.fliplr(zero2[:,i*100:(i*100)+100])

cmp = pd.DataFrame(zero2)

plt.imshow(data,cmap='Greys',aspect='auto')
cmp.to_csv('cmp_gather_{}_{}_{}.csv'.format(model,layer,percobaan),index=False,header=False)

t1 = time.process_time()
t = t1-t0

tt = t/60

print('Waktu proses:',int(tt),'menit',(tt-int(tt))*60,'detik')