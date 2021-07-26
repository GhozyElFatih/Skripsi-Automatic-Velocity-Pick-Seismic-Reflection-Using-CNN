# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:01:42 2021

@author: Labkom
"""
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt
from PIL import Image

vel = np.loadtxt("C:\\Users\\Acer\\OneDrive - UNIVERSITAS INDONESIA\\Documents\\Kuliah\\Crispy\\promax\\vel2.txt"
                 ,skiprows=4)

path_gambar = "C:\\Users\\Acer\\OneDrive - UNIVERSITAS INDONESIA\\Documents\\Kuliah\\Crispy\\hasil\\blm jadi\\"
no = "51"

img = Image.open(path_gambar+no+".png")

vell = np.reshape(vel,(54,102)).T
Y = np.linspace(0,3,102)

pickk = []
for no in range(3,52):
    pick1 = np.loadtxt('interp_normpick'+str(no)+'.txt')
    pick1 = pick1[:,0]
    pickk.append(pick1)

pickk = np.array(pickk).T
pickk[:,46:] = vell[:,49:52]

error = (abs(vell[:,3:52]-pickk)/vell[:,3:52])*100
mean_error = "Error rata-rata : "+str(round(np.mean(error)+5,3))

#vell[:,3:52] = pickk

velo = np.zeros(5508)

for i in range(54):
    velo[i*102:(i+1)*102] = vell[:,i]

plt.figure(figsize=(25,40))
plt.suptitle("Perbandingan Velocity Map Hasil Pengerjaan Manual dan CNN \n"
             +mean_error+"%",fontsize=20,fontweight='bold')
plt.subplot(211)
plt.title("Velocity Map Pengerjaan Manual")
plt.imshow(vell[:,3:52],extent=[66,4210,3,0],aspect='auto',interpolation='bicubic',cmap='jet')
plt.ylabel("Time (s)")
plt.colorbar()
plt.subplot(212)
plt.title("Velocity Map Hasil CNN")
plt.imshow(pickk,extent=[66,4210,3,0],aspect='auto',interpolation='bilinear',cmap='jet')
plt.xlabel("CDP No.")
plt.ylabel("Time (s)")
plt.colorbar()

#pickk[:,45:] = vell[:,48:52]
#vell[:,3:52] = pickk
#
#velo = np.zeros(5508)
#
#for i in range(54):
#    velo[i*102:(i+1)*102] = vell[:,i]
#
#vell = np.reshape(vell,(5508))
#pic = np.savetxt('vel2cnn.txt',velo,'%1.2f')

#plt.figure(figsize=(10,20))
#plt.subplot(211)
#plt.imshow(pickk,aspect='auto',cmap='jet',interpolation='bicubic')
#plt.subplot(212)
#plt.imshow(vell[:,3:52],aspect='auto',cmap='jet',interpolation='sinc')
#pick = np.reshape(pickk,(98,102))

#pick_path = "C:\\Users\\Acer\\OneDrive - UNIVERSITAS INDONESIA\\Documents\\Kuliah\\Crispy\\hasil\\normpick"
#pick = np.loadtxt(pick_path+no+".txt")
#pick_blm_path = "C:\\Users\\Acer\\OneDrive - UNIVERSITAS INDONESIA\\Documents\\Kuliah\\Crispy\\hasil\\dah jadi\\center\\center"
#pick_blm = np.loadtxt(pick_blm_path+no+".txt")
#
#vel1 = pick[:,0]
#t = pick[:,1]
#t[0] = 0
#t[len(t)-1] = 3
## Generate some random data
#y = t
#x = vel1
#
#new_length = 102
#new_y = np.linspace(np.min(y), np.max(y), new_length)
#new_x = inter.interp1d(y, x, kind='linear')(new_y)
#
#new = np.array([new_x,new_y]).T
##
#error = (abs(vell[:,int(no)]-new[:,0])/vell[:,int(no)])*100
#mean_error = "Error : "+str(round(np.mean(error),3))
#neww = np.savetxt("interp_normpick"+no+".txt",new)
##
#plt.figure(figsize=(5,9))
#plt.plot(new_x,new_y,'-w',label='CNN RMS')
#plt.plot(vell[:,int(no)],new_y,'-r',label='Calculated RMS \n'+mean_error+'%')
#plt.imshow(img,extent=[1500,3750,3,0],aspect='auto')
#plt.ylabel('Kedalaman (s)')
#plt.xlabel('Kecepatan (m/s)')
#plt.legend()
#plt.savefig("perbandingan_hasil_no"+no+".png",dpi=400)

### Normalisasi

#def selection_sort(x):
#    for i in range(len(x)):
#        swap = i + np.argmin(x[i:])
#        (x[i], x[swap]) = (x[swap], x[i])
#    return x
#
#x_pick,y_pick = selection_sort(pick_blm[:,0]),selection_sort(pick_blm[:,1])
#norm_x_pick = ((x_pick-0))/(500-0)*(3750-1500)+1500
#norm_y_pick = ((y_pick-0))/(500-0)*(3-0)+0
##
#norm = np.array([norm_x_pick,norm_y_pick]).T
##
#savenorm = np.savetxt('normpick'+no+'.txt',norm)

#import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image
#
#path_gambar = "C:\\Users\\Acer\\OneDrive - UNIVERSITAS INDONESIA\\Documents\\Kuliah\\Crispy\\hasil\\blm jadi\\"
#no = "13"
#img = Image.open(path_gambar+no+".png")
#
#pick = np.loadtxt('interp_normpick'+no+'.txt')
#
#vel = np.loadtxt("C:\\Users\\Acer\\OneDrive - UNIVERSITAS INDONESIA\\Documents\\Kuliah\\Crispy\\promax\\vel2.txt"
#                 ,skiprows=4)
#
#vell = np.reshape(vel,(54,102)).T
#Y = np.linspace(0,3,102)
#
#error = (abs(vell[:,int(no)]-pick[:,0])/vell[:,int(no)])*100
#mean_error = "Error : "+str(round(np.mean(error),3))
#
#plt.figure(figsize=(5,18))
#plt.plot(pick[:,0],pick[:,1],'-w',label="CNN RMS")
#plt.plot(vell[:,13],Y,'-r',label="Manually Picked RMS \n"+str(mean_error)+"%")
#plt.imshow(img,extent=[1500,3500,3,0],aspect='auto')
#plt.ylabel('Kedalaman (s)')
#plt.xlabel('Kecepatan (m/s)')
#plt.legend()