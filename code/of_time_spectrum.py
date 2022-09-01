#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:57:00 2022

@author: xuyang
"""

global Rsun
from matplotlib import pyplot as plt
import numpy as np
import os,glob
import of_tools as tools
  
dirname='/mb_data_sample/'
dir_out='/mb_data_sample/of_outflow/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
#define your multiband wavelengths
wave=['r080','r040','r000','b040','b080']
wavenum=len(wave)#total number of wavelengths
#define the optical flow co-alignment sequence 
align=[[1,0],[2,1],[3,2],[4,3],
       [0,4],[1,3],
       [0,2],[2,4]]
plt.close('all')


target=1#reference wavelength, other wavelengths will be aligned to this one
centlist=sorted(glob.glob(dirname+wave[target]+'/*.fts'))
tot=len(centlist)#total number of scans, can make it shorter for debug
im0org,head0=tools.fitsread(centlist[0])#read the first fits file in the reference wavelength
imm,head0=tools.fitsread(centlist[round(tot/2)]) #select a reference time for sequence alignment. Here align the dataset to its half sequence file.
Tr=tools.num_time(head0['TIME-OBS'])#Reference time, for derotation. If your dataset is de-rotated or no need for de-rotation, skip this step.
dis=None#flag for display

im1=im0org.copy()
wd=500#Half FOV size for the alignment
im1=im1/im1[wd:-wd,wd:-wd].mean()*10000#Adjust the intensity 
dxy=[]#Record for relative displacements, from the previous image
xy_shift=[]#Record for accumulative displacements, from the first image
xxc=[]#Record for displacements from the reference image
yyc=[]
for i in range(3):
    im2org,hd=tools.fitsread(centlist[i])
    difT=tools.num_time(hd['TIME-OBS'])-Tr
    rot = 360./24/3600*difT            
    im2org = tools.imrotate(im2org,rot)
    im2=im2org/im2org[wd:-wd,wd:-wd].mean()*10000

    d,model,flag,flow =tools.align_opflow(im1,im2,winsize=31,step=5,r_t=5,arrow=0)#Optical Flow, calculate displacements
    dxy.append(d)
    im2dxy=np.sum(np.array(dxy),axis=0)
    xy_shift.append(im2dxy) 
    im2new=tools.immove2(im2org,-im2dxy[0],-im2dxy[1])#Align and move current image with to the first image 
    print(i,-im2dxy)
    
    im3=im2new[wd:-wd,wd:-wd]
    #Here gives an example that align current image to the reference image (half sequence) with our Fourier Correlation Tracker. 
    #This CT algorithm uses similar stratgy as our OF algorithm.
    #You can also use tools.align_opflow to do it.
    Dx,Dy,cor=tools.xcorrcenter(imm[wd:-wd,wd:-wd],im3)
    print(i,Dy,Dx,cor)

    xxc.append(Dx)
    yyc.append(Dy)
    im1=im2.copy()
    
xxc=np.array(xxc)
yyc=np.array(yyc) 
xy_shift=np.array(xy_shift)   
x=np.array(range(len(xxc)))
Dx,Dy=tools.fit_dxy(x,xxc,yyc)#Fitting for the displacements, not necessary. You can use the original calculated displacements from the reference image
plt.figure()
plt.plot(xxc) 
plt.plot(yyc)
plt.plot(Dx)
plt.plot(Dy)

plt.figure('color')
filelist=[]#Get a filelist for all data set
for i in range(wavenum):
    tmpfile=sorted(glob.glob(dirname+wave[i]+'/*.fts'))
    filelist.append(tmpfile)
    
for i in range(0,tot):
    im=[]#For each scan, readfits
    for j in range(wavenum):
        tmp=tools.fitsread(filelist[j][i])[0]
        im.append(tmp)
        print(filelist[j][i])
    im=np.array(im)    
    
    #Initiate reference for each scan
    for j in range(wavenum):
        im[j]=tools.immove2(im[j],-xy_shift[i,0]+Dy[i],-xy_shift[i,1]+Dx[i])

    IM=im[:,wd:-wd,wd:-wd]
    #Align multiband data set for each scan 
    im=tools.all_align(im,align,wavenum)
    im=im.astype(np.float32) 
    subfile=os.path.basename(filelist[2][i])
    print(i, subfile)
    tools.fitswrite(dir_out+subfile,im,header=None)
