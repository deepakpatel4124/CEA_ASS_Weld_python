#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:15:25 2021

@author: anishkumar
"""

#M2M B-scan data SAFT processing
#Import functions
#Import functions
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal 

#Enter parameter values
p=1 # pitch in mm
dT=0.01 # dT in us
thick=29.8#thickness for SS weld#0=19.27 # thickness in 23.38 mm for 617 buttering layer; 15.05 for

filename1='/Volumes/Transcend-N/data/SP/tt ss/50.capture_acq/data.bin'

filename10='/Volumes/Transcend-N/data/SP/TT ss 316 weld sample 0/2.25 1p 100 mhz 60 db.capture_acq/data.bin'


data=np.fromfile(filename1, dtype='<i2',sep='')

dL= data[63]+10 #data_points 
print(dL)
N = int(data[55]) #transducer_num
data=data[64:]
data = data.clip(-512, 512) # Clipping the data to 10 bit i.e. +-512
len=1800 #800 for 15 mm #1500 for buttering layer&1800 for FS
data4=np.zeros((N,N,len),dtype='<i2')

for x in range (N):
    for y in range (N):
        data4[x,y,:]=data[x*(N*dL+10)+y*dL:x*(N*dL+10)+y*dL+len]

k=2
BSCAN=data4[k,:,:]
bscan1=(np.transpose(BSCAN))
#plt.subplot(2,2,1)

#plt.imshow(bscan1, aspect='auto', cmap = 'viridis', )

bscan1T=bscan1.transpose()
sig1=(bscan1T[k]/(np.max(bscan1T[k])))
sig2=sig1[0:int(len/3)]
sig3=sig1[int(2*len/3):]
sig4=np.zeros(int(len/3),dtype='<i2')
sig_0=np.concatenate((sig2,sig4,sig3), axis=0)
sig1=sig_0
#plt.subplot(2,2,2)
#plt.plot(sig1)
corr=signal.correlate((sig1[0:int(len/2)]),(sig1[int(len/2):]))
max_p=np.argmax(corr)
tof_0=len-max_p


#plt.subplot(2,2,3)
#plt.plot(sig1[tof_1:])
print(tof_0)
Theta=[]
Vel=[]
TOF_1=[]
#N=22#for small sample

for j in range(int(N)):
    k=j
    BSCAN=data4[k,:,:]
    bscan1=(np.transpose(BSCAN))
    bscan1T=bscan1.transpose()
    sig1=(bscan1T[k]/(np.max(bscan1T[k])))
    sig2=sig1[0:int(len/3)]
    sig3=sig1[int(2*len/3):len]
    sig4=np.zeros(int(len/3),dtype='<i2')
    sig1=np.concatenate((sig2,sig4,sig3), axis=0)
#    plt.subplot(2,2,2)
#    plt.plot(sig1)
    corr=signal.correlate((sig_0[0:int(len/2)]), (sig1[0:int(len/2)]))
    max_p=np.argmax(corr)
    tof_1=len-max_p+tof_0
    TOF_1.append(tof_1)
    
    TOF=[]
    VEL=[]
    THETA=[]
    for i in range (int(N)):
        len1=400+abs((k-i))*11 #900 for 617 
#        len1=600
        corr=signal.correlate(sig_0[0:len1],bscan1T[i][0:len1])
        max_p=np.argmax(corr)
        tof=(len1-max_p+tof_0/2)*dT #512 for 15 mm side  
        TOF.append(tof)
        dist=(thick**2+((k-i)*p)**2)**0.5
        vel=dist*1000/tof
        VEL.append(vel)
        theta=(np.arctan(((k-i)*p)/thick))*180/np.pi
        THETA.append(theta)
        Theta.append(theta.round(0))
        Vel.append(vel.round(3))
#plt.subplot(2,2,4)
n=0
J=[0,16,32,48,63]
for i in range (5):    
    plt.subplot(1,5,i+1)
    n=J[i]
    plt.plot(Theta[n*64:(n+1)*64],Vel[n*64:(n+1)*64], linewidth=1.6)
    plt.ylim(5500,5950)
    plt.xlim(-60,60)
    plt.xlabel('Angle, deg)')
    plt.ylabel('Ultrasonic velocity, m/s')



Theta1=np.around(Theta,decimals=0)
Vel1=np.around(Vel,decimals=3)
#plt.subplot(2,2,3)
#plt.plot(TOF_1)
#plt.ylabel('Normal beam TOF, microsecond')
#plt.scatter(Theta1,Vel1, s=1)

"""
Theta_min=int(np.min(Theta1))
Theta_max=int(np.max(Theta1))
zipped=zip(Theta1,Vel1)
res = sorted(zipped, key = lambda x: x[0])
n=int(np.size(res)/2)

V=[]
T=[]

for i in range (Theta_min, Theta_max, 1):
    k=0
    r=0
    for j in range (n):
        if res[j][0]==i:
            k+=res[j][1]
            r+=1
    if r>=1:
        V.append(np.round(k/r, 3))
        T.append(i)

P=np.polyfit(T,V, 6)
V_P=[]
for i in (T):
    V_P.append(P[0]*i**6+P[1]*i**5+P[2]*i**4+P[3]*i**3+P[4]*i**2+P[5]*i+P[6])

print(P)
plt.plot(T, V_P, color='black', linewidth=1.5)

sinf=[]
I=[]
for i in range (-70,70,1):
    sinf.append(5.96+0.3*(np.sin(3.75*(i-28)*np.pi/180)))
    I.append(i)
plt.plot(I,sinf, linestyle='dashed', linewidth=2, color='black')
plt.ylim(5.6, 6.4)

"""
"""
for x in range (N):
    for y in range (N):
        r=y
        if r>31:
            r=r-32
        else:
            r=r+32
        data5[x,y,:]=data[64+x*(N*dL+10)+r*dL:64+x*(N*dL+10)+r*dL+dL]
BSCAN2=data5[k,:,:]
bscan2=np.transpose(BSCAN2)
plt.subplot(1,2,2)
plt.imshow(bscan2, aspect='auto', cmap = 'viridis', )
"""
############# open M2M data file and plot Raw B-scan