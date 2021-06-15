# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:05:30 2020

@author: Acer
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as intr

model = '8'
percobaan = '10'

data = pd.read_csv('8_5_10.csv')

model1_v1 = data['v1'].tolist()
model1_v2 = data['v2'].tolist()
model1_v3 = data['v3'].tolist()
model1_v4 = data['v4'].tolist()
model1_v5 = data['v5'].tolist()
#model1_v6 = data['v6'].tolist()
#model1_v7 = data['v7'].tolist()
#model1_v8 = data['v8'].tolist()

model1,model2,model3,model4,model5 = [],[],[],[],[]
for i in range(len(model1_v1)):
    aa = model1_v1[i],model1_v1[i],model1_v1[i],model1_v1[i]
    bb = model1_v2[i],model1_v2[i],model1_v2[i],model1_v2[i]
    cc = model1_v3[i],model1_v3[i],model1_v3[i],model1_v3[i]
    dd = model1_v4[i],model1_v4[i],model1_v4[i],model1_v4[i]
    ee = model1_v5[i],model1_v5[i],model1_v5[i],model1_v5[i]
    #ff = model1_v6[i],model1_v6[i]
    #gg = model1_v7[i],model1_v7[i]
    #hh = model1_v8[i],model1_v8[i]
    model1.append(aa)
    model2.append(bb)
    model3.append(cc)
    model4.append(dd)
    model5.append(ee)
    #model6.append(ff)
    #model7.append(gg)
    #model8.append(hh)

model1 = np.array(model1).flatten()
model2 = np.array(model2).flatten()
model3 = np.array(model3).flatten()
model4 = np.array(model4).flatten()
model5 = np.array(model5).flatten()
#model6 = np.array(model6).flatten()
#model7 = np.array(model7).flatten()
#model8 = np.array(model8).flatten()

V1 = np.array([model1,model1,model1,model1,model1,model1,model1,model1,
               model2,model2,model2,model2,model2,model2,model2,model2,
               model3,model3,model3,model3,model3,model3,model3,model3,
               model4,model4,model4,model4,model4,model4,model4,model4,
               model5,model5,model5,model5,model5,model5,model5,model5])

xx = np.linspace(0,10000,len(np.transpose(V1)))
yy = np.linspace(0,4000,len(V1))

xj,yj = [],[]

for i in yy:
    for j in xx:
        xj.append(i)
        yj.append(j)
Xx = np.array(xj)
Yy = np.array(yj)

X1,Y1 = np.meshgrid(xx,yy)

v_int1,v_int2,v_int3,v_int4,v_int5 = V1[:,0],V1[:,5],V1[:,10],V1[:,15],V1[:,20]
v_int6,v_int7,v_int8,v_int9,v_int10 = V1[:,25],V1[:,30],V1[:,35],V1[:,40],V1[:,45]
v_int11,v_int12,v_int13,v_int14,v_int15 = V1[:,50],V1[:,55],V1[:,60],V1[:,65],V1[:,70]
v_int16,v_int17,v_int18,v_int19,v_int20 = V1[:,75],V1[:,80],V1[:,85],V1[:,90],V1[:,95]

t = []

for i in range(len(v_int1)):
    i = 100
    t.append(i)

vint1,vint2,vint3,vint4,vint5,vint6,vint7,vint8,vint9,vint10 = [],[],[],[],[],[],[],[],[],[]
vint11,vint12,vint13,vint14,vint15,vint16,vint17,vint18,vint19,vint20 = [],[],[],[],[],[],[],[],[],[]

tt = []

### Bikin Segitiga

for i in range(0,len(v_int1)):
    for j in range(0,i+1):
        b1 = v_int1[j]**2
        b2 = v_int2[j]**2
        b3 = v_int3[j]**2
        b4 = v_int4[j]**2
        b5 = v_int5[j]**2
        b6 = v_int6[j]**2
        b7 = v_int7[j]**2
        b8 = v_int8[j]**2
        b9 = v_int9[j]**2
        b10 = v_int10[j]**2
        b11 = v_int11[j]**2
        b12 = v_int12[j]**2
        b13 = v_int13[j]**2
        b14 = v_int14[j]**2
        b15 = v_int15[j]**2
        b16 = v_int16[j]**2
        b17 = v_int17[j]**2
        b18 = v_int18[j]**2
        b19 = v_int19[j]**2
        b20 = v_int20[j]**2
        time = t[j]
        tt.append(time)
        vint1.append(b1)
        vint2.append(b2)
        vint3.append(b3)
        vint4.append(b4)
        vint5.append(b5)
        vint6.append(b6)
        vint7.append(b7)
        vint8.append(b8)
        vint9.append(b9)
        vint10.append(b10)
        vint11.append(b11)
        vint12.append(b12)
        vint13.append(b13)
        vint14.append(b14)
        vint15.append(b15)
        vint16.append(b16)
        vint17.append(b17)
        vint18.append(b18)
        vint19.append(b19)
        vint20.append(b20)

def rev_slice(mylist):
    a = mylist[::-1]
    return a
vint1,vint2,vint3,vint4,vint5 = rev_slice(vint1),rev_slice(vint2),rev_slice(vint3),rev_slice(vint4),rev_slice(vint5)
vint6,vint7,vint8,vint9,vint10 = rev_slice(vint6),rev_slice(vint7),rev_slice(vint8),rev_slice(vint9),rev_slice(vint10)
vint11,vint12,vint13,vint14,vint15 = rev_slice(vint11),rev_slice(vint12),rev_slice(vint13),rev_slice(vint14),rev_slice(vint15)
vint16,vint17,vint18,vint19,vint20 = rev_slice(vint16),rev_slice(vint17),rev_slice(vint18),rev_slice(vint19),rev_slice(vint20)

### Masukin ke matriks diagonal atas

def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(len(v_int1), 0)] = values
    return(upper)

d1 = create_upper_matrix(vint1, len(v_int1))
d2 = create_upper_matrix(vint2, len(v_int2))
d3 = create_upper_matrix(vint3, len(v_int3))
d4 = create_upper_matrix(vint4, len(v_int4))
d5 = create_upper_matrix(vint5, len(v_int5))
d6 = create_upper_matrix(vint6, len(v_int6))
d7 = create_upper_matrix(vint7, len(v_int7))
d8 = create_upper_matrix(vint8, len(v_int8))
d9 = create_upper_matrix(vint9, len(v_int9))
d10 = create_upper_matrix(vint10, len(v_int10))
d11 = create_upper_matrix(vint11, len(v_int11))
d12 = create_upper_matrix(vint12, len(v_int12))
d13 = create_upper_matrix(vint13, len(v_int13))
d14 = create_upper_matrix(vint14, len(v_int14))
d15 = create_upper_matrix(vint15, len(v_int15))
d16 = create_upper_matrix(vint16, len(v_int16))
d17 = create_upper_matrix(vint17, len(v_int17))
d18 = create_upper_matrix(vint18, len(v_int18))
d19 = create_upper_matrix(vint19, len(v_int19))
d20 = create_upper_matrix(vint20, len(v_int20))
T = create_upper_matrix(tt, len(v_int1))

### Ngitung RMS nya

vrms1,vrms2,vrms3,vrms4,vrms5,vrms6,vrms7,vrms8,vrms9,vrms10 = [],[],[],[],[],[],[],[],[],[]
vrms11,vrms12,vrms13,vrms14,vrms15,vrms16,vrms17,vrms18,vrms19,vrms20 = [],[],[],[],[],[],[],[],[],[]
for i in range(len(v_int1)):
    vsum1,vsum2,vsum3,vsum4,vsum5 = np.sum(d1[i]),np.sum(d2[i]),np.sum(d3[i]),np.sum(d4[i]),np.sum(d5[i])
    vsum6,vsum7,vsum8,vsum9,vsum10 = np.sum(d6[i]),np.sum(d7[i]),np.sum(d8[i]),np.sum(d9[i]),np.sum(d10[i])
    vsum11,vsum12,vsum13,vsum14,vsum15 = np.sum(d11[i]),np.sum(d12[i]),np.sum(d13[i]),np.sum(d14[i]),np.sum(d15[i])
    vsum16,vsum17,vsum18,vsum19,vsum20 = np.sum(d16[i]),np.sum(d17[i]),np.sum(d18[i]),np.sum(d19[i]),np.sum(d20[i])
    Time = np.sum(T[i])
    rms1 = np.sqrt((vsum1*Time)/Time)
    rms2 = np.sqrt((vsum2*Time)/Time)
    rms3 = np.sqrt((vsum3*Time)/Time)
    rms4 = np.sqrt((vsum4*Time)/Time)
    rms5 = np.sqrt((vsum5*Time)/Time)
    rms6 = np.sqrt((vsum6*Time)/Time)
    rms7 = np.sqrt((vsum7*Time)/Time)
    rms8 = np.sqrt((vsum8*Time)/Time)
    rms9 = np.sqrt((vsum9*Time)/Time)
    rms10 = np.sqrt((vsum10*Time)/Time)
    rms11 = np.sqrt((vsum11*Time)/Time)
    rms12 = np.sqrt((vsum12*Time)/Time)
    rms13 = np.sqrt((vsum13*Time)/Time)
    rms14 = np.sqrt((vsum14*Time)/Time)
    rms15 = np.sqrt((vsum15*Time)/Time)
    rms16 = np.sqrt((vsum16*Time)/Time)
    rms17 = np.sqrt((vsum17*Time)/Time)
    rms18 = np.sqrt((vsum18*Time)/Time)
    rms19 = np.sqrt((vsum19*Time)/Time)
    rms20 = np.sqrt((vsum20*Time)/Time)
    vrms1.append(rms1)
    vrms2.append(rms2)
    vrms3.append(rms3)
    vrms4.append(rms4)
    vrms5.append(rms5)
    vrms6.append(rms6)
    vrms7.append(rms7)
    vrms8.append(rms8)
    vrms9.append(rms9)
    vrms10.append(rms10)
    vrms11.append(rms11)
    vrms12.append(rms12)
    vrms13.append(rms13)
    vrms14.append(rms14)
    vrms15.append(rms15)
    vrms16.append(rms16)
    vrms17.append(rms17)
    vrms18.append(rms18)
    vrms19.append(rms19)
    vrms20.append(rms20)

vrms1,vrms2,vrms3,vrms4,vrms5 = rev_slice(vrms1),rev_slice(vrms2),rev_slice(vrms3),rev_slice(vrms4),rev_slice(vrms5)
vrms6,vrms7,vrms8,vrms9,vrms10 = rev_slice(vrms6),rev_slice(vrms7),rev_slice(vrms8),rev_slice(vrms9),rev_slice(vrms10)
vrms11,vrms12,vrms13,vrms14,vrms15 = rev_slice(vrms11),rev_slice(vrms12),rev_slice(vrms13),rev_slice(vrms14),rev_slice(vrms15)
vrms16,vrms17,vrms18,vrms19,vrms20 = rev_slice(vrms16),rev_slice(vrms17),rev_slice(vrms18),rev_slice(vrms19),rev_slice(vrms20)

### Digabung

vrms = np.array([vrms1,vrms1,vrms1,vrms1,vrms1,vrms2,vrms2,vrms2,vrms2,vrms2,
                 vrms3,vrms3,vrms3,vrms3,vrms3,vrms4,vrms4,vrms4,vrms4,vrms4,
                 vrms5,vrms5,vrms5,vrms5,vrms5,vrms6,vrms6,vrms6,vrms6,vrms6,
                 vrms7,vrms7,vrms7,vrms7,vrms7,vrms8,vrms8,vrms8,vrms8,vrms8,
                 vrms9,vrms9,vrms9,vrms9,vrms9,vrms10,vrms10,vrms10,vrms10,vrms10,
                 vrms11,vrms11,vrms11,vrms11,vrms11,vrms12,vrms12,vrms12,vrms12,vrms12,
                 vrms13,vrms13,vrms13,vrms13,vrms13,vrms14,vrms14,vrms14,vrms14,vrms14,
                 vrms15,vrms15,vrms15,vrms15,vrms15,vrms16,vrms16,vrms16,vrms16,vrms16,
                 vrms17,vrms17,vrms17,vrms17,vrms17,vrms18,vrms18,vrms18,vrms18,vrms18,
                 vrms19,vrms19,vrms19,vrms19,vrms19,vrms20,vrms20,vrms20,vrms20,vrms20])
vrms = np.transpose(vrms)

### Interpolasi

rbf1 = intr.Rbf(Xx,Yy,vrms,smooth=0.5)
vel1 = rbf1(Y1,X1)

### Save CSV
'''
import csv 
  
with open('RMS_{}_{}x{}_{}.csv'.format(model,len(V1),len(np.transpose(V1)),percobaan), 'w', newline='') as f: 
    write = csv.writer(f)
    write.writerows(vrms)

### Plotting

fig = plt.figure(figsize=(15,7))
plt.contourf(X1,Y1,vel1,len(V1)-1)
plt.suptitle('RMS | Model {} | Ukuran = {}x{} | Percobaan = {}'.format(model,len(V1),len(np.transpose(V1)),percobaan),size=20)
plt.gca().invert_yaxis()
plt.xlabel('Offset (m)')
plt.ylabel('Kedalaman (m)')
bar = plt.colorbar()
bar.set_label('Kecepatan (km/s)')
fig.savefig('RMS_{}_{}x{}_{}.png'.format(model,len(V1),len(np.transpose(V1)),percobaan),dpi=300)
'''