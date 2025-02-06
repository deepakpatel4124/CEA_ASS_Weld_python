#M2M B-scan data SAFT processing
#Import functions
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.signal as sig
import itertools

tic=time.time()

color_list = [(0.0, 'white'),   # white
              (0.33, 'blue'),   # blue
              (0.67, 'yellow'), # yellow
              (1.0, 'red')]     # red
cmap = colors.LinearSegmentedColormap.from_list('mycmap', color_list)

#Enter parameter values
p=1.0 # pitch in mm
dT=0.02 # dT in us
#N=64 # No. of A-scans
dx=float(.25) # dx in mm for both x and y

vel=5.7 #velocity in mm/us
thick=28 # thickness in mm

mmdp=vel*dT/2

min_x = float(-40) #float(input('Enter the min_X(mm) : '))/1000
max_x = float(40) #float(input('Enter the max_X(mm) : '))/1000
min_y = float(0) #float(input('Enter the min_Y(mm) : '))/1000
max_y = float(50) #float(input('Enter the max_X(mm) : '))/1000

x_num=int((max_x-min_x)/dx)
y_num=int((max_y-min_y)/dx)

filename = r"H:\Deepak PWI WP#0\WP#0 PM WC root 0 to 88 deg step 1 2.25 50Mhz 30dB.capture_acq\data.bin"
data=np.fromfile(filename, dtype='<i2',sep='')

dL= int(data[63]+10) #data_points 
N = int(data[55]) #transducer_num
print(dL, N)
AngStart=0
AngEnd=88
AngStep=1
M=1+int((AngEnd-AngStart)/AngStep)

PW=(N-1)*p#PW is the probe width

AS=0#start angnle for processing
AE=10 #end angnle for processing
AJ=1 #Angnle step for processing
AN=int((AE-AS)/AJ)+1 # Angle numbers for processing
print(AN)

data=(data[64:].clip(-512, 512)[:M*(N*dL+10)].reshape(M,(N*dL+10))[:,:N*dL]).reshape(M,N,dL)[:,:,:dL-10]# Clipping the data to 10 bit i.e. +-512

data1=np.zeros((M,N,dL-10))
for i in range (M):
    theta=(i*AngStep+AngStart)*np.pi/180
    D1=int(PW*(((np.sin(AngEnd*np.pi/180)-np.sin(theta))/2)/(2*mmdp))) # data length to be deleted from every A-scan of kth angle with respect to the max angle data 
    data1[i,:,:dL-10-D1]=data[i,:,D1:]

#plt.subplot(1,2,1)
#plt.imshow(np.transpose(data[0,:,:]), aspect='auto')
#plt.subplot(1,2,2)
#plt.imshow(np.transpose(data1[0,:,:]), aspect='auto')

min_T=-(N/2-0.5)*p

image=np.zeros((x_num,y_num))               

x1=np.arange(min_x, max_x, dx)
y1=np.arange(min_y, max_y, dx)
x,y=np.meshgrid(x1,y1)

T=np.arange(AS,AE+AJ,AJ) #Different angle values
#t1=(T*p+min_T)

t1=T*np.pi/180 #Different angle values in radians

R=np.arange(0,N,1) #Receiver No.
r1=(R*p+min_T) #Receiver position

#D1=int(PW*(((np.sin(AngEnd*np.pi/180)-np.sin(theta))/2)/(2*mmdp))) # data length to be deleted from every A-scan of kth angle with respect to the max angle data 


####
#SAMPLE=[(((np.sqrt((np.square(x-r))+np.square(y)))+(np.sqrt((np.square(x-t))+np.square(y))))/(vel*dT)).astype(int) for t,r in itertools.product(t1,r1)]
#SAMPLE=[(10+(((PW*(np.sin(AngEnd*np.pi/180)-np.sin(t))/2)+((np.sqrt((np.square(x-r))+np.square(y)))+((y*np.cos(t))+((PW/2-x)*np.sin(t)))))/(vel*dT))).astype(int) for t,r in itertools.product(t1,r1)]
SAMPLE=[(((np.sqrt((np.square(x-r))+np.square(y)))+((y*np.cos(t))+((PW/2-x)*np.sin(t))))/(vel*dT)).astype(int) for t,r in itertools.product(t1,r1)]
print(np.shape(SAMPLE))
image=np.zeros((y_num,x_num))
for i in range (AN):
    b=np.zeros((y_num,x_num))
    for j in range (N):
        a=data1[int(i*AJ/AngStep),N-j-1,SAMPLE[i*N+j]]
        b+=a
    image+=b

#image=image.transpose()
#plt.subplot(2,1,1)
plt.imshow(image, aspect='auto', cmap = cmap, extent=(min_x,max_x,max_y,min_y))
#plt.colorbar()



X_NUM=np.arange(0,x_num, 1)    
image1=[(((np.imag(sig.hilbert((((image.transpose()))[i]))))**2+(np.real(sig.hilbert((((image.transpose()))[i]))))**2)**0.5) for i in X_NUM]

image2=20*np.log(image1/np.max(image1))


plt.subplot(2,1,2)
plt.imshow(np.transpose(image2), aspect='auto', cmap = cmap, vmin=-30, vmax=0, extent=(min_x,max_x,max_y,min_y) )
plt.colorbar()

toc=time.time()
print(toc-tic)
