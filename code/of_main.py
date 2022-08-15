#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:20:27 2022

@author: xuyang
"""

from matplotlib import pyplot as plt
import numpy as np
import glob,os
import of_tools as tools
plt.close('all')

dir_in=r'/media/xuyang/E/test0/*.fts'
dir_out='/media/xuyang/E/test0/of/'

if not os.path.exists(dir_out):
    os.makedirs(dir_out)
filelist=sorted(glob.glob(dir_in))
tot=len(filelist)

im0,h0=tools.fitsread(filelist[0])
OT = h0['TIME-OBS']
T1 = tools.num_time(OT)

h,w=im0.shape
icube=np.zeros((tot,h,w))
for ii in range(tot):
    im,hd2= tools.fitsread(filelist[ii])
    OT = hd2['TIME-OBS']
    T2 = tools.num_time(OT)
    difT=T2-T1#+210*15
#    print(difT)
    rot = 360./24./3600.*difT  
    im = tools.imrotate(im,rot)
    print(filelist[ii])
    icube[ii,:,:]=im#[250:1749,250:1749]
    
ncube=tools.cubealign(icube.copy(),wd=250,winsize=31)#To align the images in a time series, use this command.\
#Please check the of_tools for the comments and explanations
ncube=ncube.astype(np.single)
tcube=[]
fn=[]
print('write fits')
for jj in range(tot):
    tcube.append(os.path.basename(filelist[jj]))
    nfn=dir_out+os.path.basename(filelist[jj])
    im,head= tools.fitsread(filelist[ii])
    tools.fitswrite(nfn,ncube[jj,:,:],head)   
    print(nfn)
    
tools.array2movie(ncube,movie_name=dir_out+'of_moive',title_cube=tcube) 
