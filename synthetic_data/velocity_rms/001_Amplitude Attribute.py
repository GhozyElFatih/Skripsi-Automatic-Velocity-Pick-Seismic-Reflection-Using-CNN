import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

t1 = time.process_time()

model = 8
percobaan = 10

traces = pd.read_csv('vint_{}_80x200_{}.csv'.format(model,percobaan),header=None)
traces = np.array(traces)*1000
nt = traces.shape[0]
nx = traces.shape[1]

Time = 50/traces

window = 25
Np = int(np.floor(window/2))
matrix_zero = np.zeros((Np,nx))
sp = np.vstack((matrix_zero,((traces**2)*Time),matrix_zero))
sp2 = np.vstack((matrix_zero,Time,matrix_zero))

sm = np.zeros((nt,nx))
# Algoritma 1
#for j in range(nx):
#    for i in range(nt):
#        sm[i,j] = np.mean(sp[np.arange(0,window,1)+i,j])

# Algoritma 2 (lebih cepat)
for i in range(nt):
    sm[i,:] = np.sqrt(np.sum(sp[np.arange(0,window,1)+i,:],axis=0)/np.sum(sp2[np.arange(0,window,1)+i,:],axis=0))

#plt.figure(1)
#plt.subplot(121)
#plt.imshow(traces,aspect='auto',interpolation='bilinear',cmap='seismic')
#plt.colorbar()
#plt.subplot(122)
#plt.imshow(sm,aspect='auto',interpolation='bilinear',cmap='seismic')
#plt.colorbar()
#
#plt.figure(2)
#plt.plot(sm[:,0],np.linspace(0,4000,80),label='0')
#plt.plot(sm[:,40],np.linspace(0,4000,80),label='40')
#plt.plot(sm[:,80],np.linspace(0,4000,80),label='80')
#plt.plot(sm[:,120],np.linspace(0,4000,80),label='120')
#plt.plot(sm[:,190],np.linspace(0,4000,80),label='190')
#plt.gca().invert_yaxis()
#plt.legend()

import csv 
  
with open('RMS_{}_{}x{}_{}.csv'.format(model,len(sm),len(np.transpose(sm)),percobaan), 'w', newline='') as f: 
    write = csv.writer(f)
    write.writerows(sm)

fig = plt.figure(figsize=(15,7))
plt.imshow(sm,aspect='auto',extent=[0,10000,4000,0],interpolation='bilinear',cmap='jet')
plt.suptitle('RMS | Model {} | Ukuran = {}x{} | Percobaan = {}'.format(model,len(sm),len(sm.T),percobaan),size=20)
plt.xlabel('Offset (m)')
plt.ylabel('Kedalaman (m)')
bar = plt.colorbar()
bar.set_label('Kecepatan (m/s)')
fig.savefig('RMS_{}_{}x{}_{}.png'.format(model,len(sm),len(sm.T),percobaan),dpi=300)

t2 = time.process_time()
t = t2-t1

tt = t/60

print('Waktu proses:',int(tt),'menit',(tt-int(tt))*60,'detik')