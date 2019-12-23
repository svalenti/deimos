#!/usr/bin/env python

from matplotlib import pylab as plt
import numpy as np
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from astropy.io import fits
import os
import sys
pyversion = sys.version_info[0]
plt.ion()

def gendeimosarc():
    data= np.genfromtxt('lamp_NIST_blue.dat')
    x,y,note,z = zip(*data)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

#    x,y,z = [],[],[]
#    hdu = fits.open('lamps.fits')
#    for l in hdu[1].data[0:1]:
#        for m,g in enumerate(l[1]):
#            if l[2][m]>1000 and l[1][m]>3500:
#                z.append(l[0])
#                x.append(l[1][m])
#                y.append(l[2][m])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    ss = np.argsort(y)[::-1]
    x = x[ss]
    z = z[ss]
    y = y[ss]
    
    wave = np.arange(3000,11000,1)
    zero = wave - wave

    tot = 30
    count=0
    fig=plt.figure(1)
    for l,m in enumerate(x):
        params = [y[l],x[l],3]
        model1 = get_profile_model(params, wave)
        zero = zero + model1
        if count < tot:
            plt.text(x[l],y[l]*1.1,str(x[l]),rotation=90, rotation_mode='anchor')
        count += 1
    plt.plot(wave,zero)

    
        
def get_profile_model(params, ys):
#    a1, cy1, sigma1, a2, cy2, sigma2 = params
    a1, cy1, sigma1 = params
    
    p1 = np.exp(-(ys - cy1)**2 / 2 / sigma1**2) 
    p1 /= p1.max()
    
#    p2 = np.exp(-(ys - cy2)**2 / 2 / sigma2**2) 
#    p2 /= p2.max()
    return a1 * p1 #+ a2 * p2

def get_profile_chisq(params, ys, profile):
    model = get_profile_model(params, ys)
    return np.sum( (profile - model)**2 / np.sqrt(np.abs(profile)) ) / (profile.size - len(params))

def fitline(xx,yy,center,amplitude=1,sigma=3,verbose=True):
    guess = [amplitude,float(center),sigma]
    params = fmin(get_profile_chisq, guess, args=(xx, yy))
    model = get_profile_model(params, xx)
    if verbose:
        plt.plot(xx,model)
        print(params)
    return params

#######################################
#hdu  = fits.open('keck_deimos_600.fits')
#dd =hdu[1].data
#xx,yy = zip(*dd)
#yy = yy - np.percentile(yy,.1)
#yy = yy/np.max(yy)

data = np.genfromtxt('arc_7.txt')
#data = np.genfromtxt('skybgopt_7.txt')

x,y = zip(*data)
x = np.array(x)
y = np.array(y)
y = y - np.percentile(y,.1)
y = y/np.max(y)

answ = 'n'


arclist = 'arc_list_7.txt'
if os.path.isfile(arclist):
    data = np.genfromtxt(arclist)
    ll,ww = zip(*data)
    ww = np.array(ww)
    sort = np.argsort(np.array(ww))
    pixels = np.array(ll)[sort]
    waves = ww[sort]

    plt.clf()
    plt.plot(pixels,waves,'or')
    params = np.polyfit(pixels,waves, deg= 4)
    p = np.poly1d(params)
    print(p)
    plt.plot(pixels,p(pixels),'-b')
    if pyversion>=3:
        input('stop')
    else:
        raw_input('stop')
else:
    pixels = []
    waves = []


dd= 30
fig=plt.figure(1)
plt.clf()
plt.plot(x,y)

#plt.plot(xx,yy,'-r')
f = interp1d(x, y)

if len(waves):
    intpeak = f(pixels)
    for m,l in enumerate(waves):
        #plt.text(waves[m], intpeak[m]*1.01, str(waves[m]),rotation=90, rotation_mode='anchor')
        plt.text(pixels[m], intpeak[m]*1.01, str(waves[m]),rotation=90, rotation_mode='anchor')

plt.xlim(4100,-100)
if pyversion>=3:
    input('stop1')
else:
    raw_input('stop1')
    
while answ in ['YES','yes','y','Y']:
    if pyversion>=3:
        center = input('pixel? ')
    else:
        center = raw_input('pixel? ')
        
    ww = [( x < float(center) + dd) & ( x > float(center) - dd)]
    params = fitline(x[ww],y[ww],center, 1,  3, verbose=True)
    if pyversion>=3:
        wave = input('which wave? ')
    else:
        wave = raw_input('which wave? ')
    if pyversion>=3:
        stop = input('one more ')
    else:
        stop = raw_input('one more ')
        
    pixels = np.append(pixels,params[1])
    waves = np.append(waves,float(wave))
    if stop in ['No','no','n']:
        answ = 'n'

np.savetxt('arc_list_7.txt',np.c_[pixels,waves])


#params = np.polyfit(waves,pixels, deg= 4)
#p = np.poly1d(params)

#np.savetxt('arc_good_7.txt',np.c_[p(x),y])


print(p)
plt.clf()
#plt.plot(pixels,waves,'ob')
plt.plot(p(x),y,'-r')

print(answ)
