#M2M B-scan data SAFT processing
#Import functions
#Import functions
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal 

#Enter parameter values
p=1# pitch in mm
dT=0.01 # dT in us
thick=29.808#=19.27 # thickness in 23.38 mm for 617 buttering layer; 15.05 for
filename1=r'D:\Py_Codes\Python FMC-TFM\Weld_exp_data\1 50db rev.capture_acq\data.bin'
filename='/Users/anishkumar/Documents/data/SP/Frritic steel/TT 100 mhz 25db ferritic.capture_acq/data.bin'
filename2='/Volumes/Transcend-N/Viswanath/Al_2.25MHz_30mm thick-TT-40dB-1.capture_acq/data.bin'
filename3='/Volumes/Transcend-N/Viswanath/2.25mhz-64e-100mhz-tt-cw0-rd-38dB.capture_acq/data.bin'
filename4='/Users/anishkumar/Documents/data/SP/Butter/Al617 100MHz AP25 1mm elevation/40 db 15mm side.capture_acq/data.bin'
filename5='/Users/anishkumar/Documents/data/SP/Butter/Al617 100MHz AP25 1mm elevation/40 db 10mm side.capture_acq/data.bin'
filename6='/Volumes/Transcend-N/data/Gekko/PA_CW/CW0-15p068-RD-100MHz-36dB.capture_acq/data.bin'
filename7='/Volumes/Transcend-N/data/SP/tt ss/40.capture_acq/data.bin'
data=np.fromfile(filename1, dtype='<i2',sep='')

dL= data[63]+10 #data_points 
N = int(data[55]) #transducer_num
data=data[64:]
data = data.clip(-512, 512) # Clipping the data to 10 bit i.e. +-512
leng=1250 #800 for 15 mm #1500 for buttering layer&1800 for FS
data4=np.zeros((N,N,leng),dtype='<i2')

for x in range (N):
    for y in range (N):
        data4[x,y,:]=data[x*(N*dL+10)+y*dL:x*(N*dL+10)+y*dL+leng]

k=0
BSCAN=data4[k,:,:]
bscan1=(np.transpose(BSCAN))
plt.subplot(2,2,1)
plt.xlabel('Distance, mm')
plt.xlim(N-1,0)
plt.ylabel('Time, microsecond')

plt.imshow(bscan1, aspect='auto', cmap = 'viridis', extent=(0, (N-1)*p, leng*dT, 0) )

bscan1T=bscan1.transpose()
sig1=(bscan1T[k]/(np.max(bscan1T[k])))
sig2=sig1[0:int(leng/3)]
sig3=sig1[int(2*leng/3):leng]
sig4=np.zeros(int(leng/3),dtype='<i2')
#sig_0=np.concatenate((sig2,sig4,sig3), axis=0)
sig_0=sig1
plt.subplot(2,2,2)
plt.plot(sig1)
plt.xlabel('Time, data point')
plt.ylabel('Amplitude, A.U.')
corr=signal.correlate((sig_0[0:int(leng/2)]),(sig_0[int(leng/2):]))
max_p=np.argmax(corr)
tof_0=leng-max_p  ## This is to cal sig start i think


#plt.subplot(2,2,3)
#plt.plot(sig1[tof_1:])
print(tof_0)
Theta=[]
Vel=[]
TOF_1=[]
#N=22#for small sample
Vel0=[]

for j in range(int(N)):
    k=j
    BSCAN=data4[k,:,:]
    sig1=(BSCAN[k]/(np.max(BSCAN[k])))
    corr=signal.correlate((sig_0[0:int(leng/2)]), (sig1[0:int(leng/2)]))
    max_p=np.argmax(corr)
    tof_1=(leng/2-max_p+tof_0/2)*dT
    vel=1000*thick/tof_1
    TOF_1.append(tof_1)
    Vel0.append(vel)
    
plt.subplot(2,2,3)
plt.plot(Vel0)
plt.ylabel('Normal beam Velocity, m/s')
plt.xlabel('Element NUmber')
plt.xlim(N-1,0)
plt.ylim(5400,6000)

k3=34
k2=1

for k1 in range (k2):
    k=k3+k1
    BSCAN=data4[k,:,:]/np.max(data4[4,:,:])
    TOF=[]
    VEL=[]
    THETA=[]
    for i in range (int(N)):
        len1=400+abs((k-i))*11 #900 for 617 
        corr=signal.correlate(sig1[0:len1],BSCAN[i][0:len1])
        max_p=np.argmax(corr)
        tof=(len1-max_p+tof_0/2)*dT #512 for 15 mm side  
        TOF.append(tof)
        dist=(thick**2+((k-i)*p)**2)**0.5
        vel=1000*dist/tof
        VEL.append(vel)
        theta=(np.arctan(((k-i)*p)/thick))*180/np.pi
        THETA.append(theta)
        Theta.append(theta.round(0))
        Vel.append(vel.round(3))
plt.subplot(2,2,4)
plt.scatter(Theta,Vel, linewidth=0.6)
plt.xlim(-70,70)
# plt.ylim(5400,6000)
plt.xlabel('Angle, deg')
plt.ylabel('Velocity, m/s')
plt.tight_layout()


Theta1=np.around(Theta,decimals=0)
Vel1=np.around(Vel,decimals=3)


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