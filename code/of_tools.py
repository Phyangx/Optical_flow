#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:21:36 2022

@author: xuyang
"""

from skimage.transform import warp, SimilarityTransform,rotate
import numpy as np
import cv2
import glob

def fitswrite(fileout, im, header=None):
    from astropy.io import fits
    import os
    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix', overwrite=True, checksum=False)
    else:        
        fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):
    from astropy.io import fits
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head 
    
def imrotate(im,para):
    return rotate(im,para,mode='reflect')
    
def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2

def num_time(obstime):
    #GST Obs_time to number
    Th = obstime[0:2]
    Tm = obstime[3:5]
    if obstime[7] == '.':
        Ts=obstime[6]
    else:
        Ts=obstime[6:8]
    NT = 3600*int(Th)+60*int(Tm)+int(Ts)
    return NT

def zscore2(im):
    im = (im - np.mean(im)) / im.std()
    return im  

def toMP4(mp4name,jpgdir='JPG'):
    filelist=sorted(glob.glob(jpgdir+'*.jpg'))
    frame = cv2.imread(filelist[0])

    fps = 10.0 #frame rate
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    size = (frame.shape[1]-10,frame.shape[0]-10) 
    out = cv2.VideoWriter(mp4name+'.mp4', fourcc, fps, size) 

    for image_file in filelist:
        frame = cv2.imread(image_file)
        out.write(frame[:size[1],:size[0],:])
        print(image_file)
     
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break    
     
    out.release()
    cv2.destroyAllWindows()
    
def array2movie(imgarr,movie_name,tmp='/home/xuyang/tmp/',title_cube=0,color='gray'):
    import os
    from matplotlib import pyplot as plt
    import shutil
    if not os.path.exists(tmp):
        os.makedirs(tmp) 
    ni,h,w=imgarr.shape
    K=0
    for i in range(ni):
        im=imgarr[i,:,:]
#    IM=tools.tocolor2(im)
        if title_cube == 0:
            subfile="{:0>3d}".format(i)
        else:
            subfile=title_cube[i]
        
        mtmp=zscore2(im)
        if K==0:            
            dis=plt.imshow(mtmp,vmax=4,vmin=-4,cmap=color,origin='lower')      
            plt.pause(0.1)
            plt.draw()
        else:
            dis.set_data(mtmp)
            plt.pause(0.1)
            plt.draw()
        K=1
        plt.title(subfile)
        print(i,subfile)
        plt.savefig(tmp+subfile+'.jpg',dpi=300)
       
    mp4n=movie_name
    toMP4(mp4n,jpgdir=tmp) 
    shutil.rmtree(tmp)
    
def immove2(im,dx=0,dy=0):
    im2=im.copy()
    tform = SimilarityTransform(translation=(dx,dy))
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='constant',cval=0)
    return im2

def cubealign(im,wd=100,winsize=21,step=2):
    #wd is half fov size to do the alignment. Dont have to be your whole FOV. winsize is subfield size to calculate the OF. step is the gap between subfield centers
    avg=im[:,wd:-wd,wd:-wd].mean()
    im=im/im[:,wd:-wd,wd:-wd].mean(axis=(-2,-1))[:,np.newaxis,np.newaxis]*avg
    IM=np.mean(im,axis=0)
    for j in range(step):
        im2=[]
        for i in range(im.shape[0]):
            d,model,flag,flow =align_opflow(IM[wd:-wd,wd:-wd],im[i,wd:-wd,wd:-wd],winsize=winsize,step=5,r_t=5)
            print('shift:',j,i,d,round(flag,3))#,model_robust.scale)
           
            tmp=immove2(im[i],-d[0],-d[1])
            im2.append(tmp)
        im2=np.array(im2)
        IM=np.mean(im2,axis=0)     
    return(im2)  
    
def align_opflow(im1org,im2org,winsize=11,step=5,r_t=5):

    dx,dy=0,0
    w=10

    im1=im1org.copy()
    im2=im2org.copy()
    
    #Intensity standardize
    im1=im1/np.mean(im1)*10000
    im2=im2/np.mean(im2)*10000


    #Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(im1, im2, flow=None, pyr_scale=0.5, levels=5, winsize=winsize, iterations=10, poly_n=5, poly_sigma=1.2, flags=0)
    
    #Select effective points based on step setup
    h,w=im1.shape
    x1, y1 = np.meshgrid(np.arange(w), np.arange(h))
    x2=x1.astype('float32')+flow[:,:,0]
    y2=y1.astype('float32')+flow[:,:,1]
    x1=x1[::step,::step]
    y1=y1[::step,::step]
    x2=x2[::step,::step]
    y2=y2[::step,::step]

    
    src=(np.vstack((x1.flatten(),y1.flatten())).T) #source coordinates
    dst=(np.vstack((x2.flatten(),y2.flatten())).T) #destination coordinates
    s=dst-src #displacement
    Dlt0=((np.abs(s[:,0])>0) + (np.abs(s[:,1])>0))>0 #Any displacement here?

    if Dlt0.sum()>0: #for displaced images
        dst=dst[Dlt0]
        src=src[Dlt0]
        s=s[Dlt0]
        #####Select effective reference points################
        d, D= mode_model((src,dst),  residual_threshold=r_t) #calculate systematic displacements
        # model, D= ransac((src, dst), func, min_samples=200,residual_threshold=r_t, max_trials=2000) #for rotation cases only. Caution! Rotation brings computations and accumulative errors.
        ###########Draw OF map if needed##################
     #####Outputs，
        #1，Ratio of effective area in the FOV；
        #2，FLAG，population of effective points; <=1, better >0.1; used as weight for cross alignment between wavelengths 
        #3，Number of effctive points, expect to be more than 1000, less than 200 may be bad; uniformity is also important
        #4，calculated displacement
        #5，accuracy of calculated displacement
        flag=D.sum()/Dlt0.sum() #population of the effective points
#        print(round(flag,3),D.sum(),d,np.std(s[D],axis=0)/np.sqrt(D.sum()-1))#,model_robust.scale)
        
        model=1 # mark for displaced image
        d[0]+=-dy
        d[1]+=-dx
    else:
        d=[-dy,-dx]
        model=1 #mark for identical image
        flag=0

    
    return d,model,flag,flow 

#select effective points with hitogram, and calculate displacement
def mode_model(data, residual_threshold=5):
    (src, dst)=data
    s=src-dst
    r=np.zeros(2)
    D=np.ones((src.shape[1],src.shape[0])).astype('bool') #position for effective points
    for j in range(2): #loop
        dat=s[:,j]
        tmp=((np.abs(dat))>0.01) #neglect small displacements
        D[j]*=tmp
        me=0 #original guess for displacement
        gain=0.1 #histogram scale
        rang=[-300,300] #histogram range
        bins=int((rang[1]-rang[0])/gain+1) #histogram bin
        wd=int(residual_threshold/gain) #FWHM
        if D[j].sum()>0:
           z=np.histogram(dat,bins=bins,range=rang) #make histogram
           k=z[0].argmax() #most populated position
           x=z[1][k-wd:k+wd+1]
           y=z[0][k-wd:k+wd+1]
           #ymin=y.min()
           # y=np.maximum(y-ymin,0)
           me=np.sum(x*y)/np.sum(y) #weight of the histogram, within thr range
        r[j]=me
        D[j]=np.abs(dat-me)<residual_threshold #get the position for effective points
        
    D=D[0]&D[1] #We process for x and  y directions
    r=np.median(s[D,:],axis=0) #calculate the mean displacement for the effective points 
    return -r,D   

