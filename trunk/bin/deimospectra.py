#!/usr/bin/env python
#
#  python 3 script for deimos reduction
#
###############################################################
description = "> reduce deimos data, run the script in a directory with deimos data "
usage = "%prog   [--iraf (reduce with iraf )\n --directory (reduce data in a different directory)\n  --interactive (reduce in interactive mode)"
import deimos
from deimos import deimosutil
from optparse import OptionParser, OptionGroup
import pyds9
import os
import pickle
import time
import re
import sys
import numpy as np
from scipy.interpolate import LSQBivariateSpline, LSQUnivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import fmin
import numpy.polynomial.legendre as leg
from matplotlib import pylab as plt
from astropy.io import fits
import glob
from deimos import __path__ as _path
from deimos import irafext
from deimos import deimoswave
import string
#os.environ["XPA_METHOD"] = 'unix'

from astropy.coordinates import SkyCoord
from astropy import units as u
std, rastd, decstd, magstd = deimosutil.readstandard('standard_deimos_mab.txt')
scal = np.pi / 180.
                
pyversion = sys.version_info[0]

# check that ds9 is open 
plt.ion()
#ds9 = pyds9.DS9(str(time.time()))
#ds9 = pyds9.DS9('deimos')
#ds9.set('frame 1')
#ds9.set('scale zscale');

# open two plot figures that will be used in the script 
fig3 = plt.figure(3)
fig2 = plt.figure(2)
verbose= True

if __name__ == "__main__":
    parser = OptionParser(usage=usage, description=description, version="%prog 1.0")
    parser.add_option("-d", "--directory", dest="directory", default=None, type="str",
                      help='reduce data in this directory \t [%default]')
    parser.add_option("-F", "--force", dest="force", action="store_true",default=False)
    parser.add_option("--iraf", dest="iraf", action="store_true")
    parser.add_option("--nosky", dest="nosky", action="store_true")
    parser.add_option("--stage", dest="stage", default=None, type="str",
                      help='reduce data a single stage \t [%default, trim,sky,trace,extract,wave,flux]')
    
    parser.add_option("-i", "--interactive", action="store_true",
                      dest='interactive', default=False, help='Interactive \t\t\ [%default]')
    option, args = parser.parse_args()
    _interactive = option.interactive
    _iraf = option.iraf
    _nsky = option.nosky
    _directory = option.directory
    _force = option.force
    stage = option.stage
    
    if _interactive:
        verbose= True
    else:
        verbose= False
    #
    #  initialize the dictionary with all the infos
    #
    dictionary, setup_object, setup_arc, setup_flat = deimosutil.checkalldata(directory=_directory,verbose=verbose)
    #
    #  check that all data are in the directory and
    # 
    run = True
    for key in setup_object:
        if key not in setup_arc:
            print('ERROR: not arc found with the same setup')
            run = False
        if key not in setup_flat:
            print('ERROR: not flat found with the same setup')
            run = False
    
    if run:
        for setup in setup_object:
            for img in setup_object[setup]:
                for key in [3,7]:
                    ##################     check if it is a standard and add it to the dictionary  ######################3
                    print(dictionary[img]['OBJECT'],dictionary[img]['RA'],dictionary[img]['DEC'])
                    _dec = dictionary[img]['DEC']
                    _ra = dictionary[img]['RA']
                    c = SkyCoord(_ra, _dec, frame='icrs',unit=(u.hourangle, u.deg))
                    _ra  = c.ra.value
                    _dec = c.dec.value
                    dd = np.arccos(np.sin(_dec * scal) * np.sin(decstd * scal) +
                                   np.cos(_dec * scal) * np.cos(decstd * scal) *
                                   np.cos((_ra - rastd) * scal)) * ((180 / np.pi) * 3600)
                    if np.min(dd) < 5200:
                        dictionary[img]['std']= std[np.argmin(dd)]
                    ######################################################
                print(setup)
                deimos.deimosutil.summary(dictionary)
                
            if setup[1] == 'LVMslitC':
                _dir = '_'.join(setup)
                if stage in [None,'trim']:
                    # trim and rotate dimension 3 and 7
                    dictionary = deimosutil.trim_rotate_split(setup_object, setup_flat, setup_arc, dictionary, setup, _force)
                else:
                    print('skip trim')

                masterflat={}    
                if stage in [None,'flat']:
                    # make masterflat3 and masterflat 7
                    print(setup_flat)
                    print(setup)
                    masterflat[3] = deimosutil.makeflat(setup_flat,dictionary,setup,3,verbose=True)
                    masterflat[7] = deimosutil.makeflat(setup_flat,dictionary,setup,7,verbose=True)
                    for img in setup_object[setup]:
                        for key in [3,7]:
                            dictionary[img]['masterflat'+str(key)]= masterflat[key]
                            dictionary[img]['trimflat'+str(key)]= dictionary[img]['trimmed'+str(key)][0].data / masterflat[key]
                else:
                    print('skip flatfield stage')
                    print('I need to add to read flat from the directory')
                        

                ########################################################
                #   
                ##    This stage is done once and it is not related to stages
                #
                print('\n####  Find curvature solution of different setup ')
                lambdas={}
                lamb={}
                for key in [3,7]:#lambdas:
                        if os.path.isfile('lambdas_' + str(pyversion) + '_' + str(key) + '.dat'):
                            print('\n### I Found the curvature solution, skip this step')
                            lambdas[key] = pickle.load(open(os.path.join('./','lambdas_' + str(pyversion) + '_'  + str(key) + '.dat'), 'rb'))
                        else:
                            lambdas[key] = None
            
                        if lambdas[key] is None:
                            print('\n### Select which arc do you want to use ')
                            for img in setup_arc[setup]:
                                if 'trimflat' + str(key)  in dictionary[img]:
                                    image = dictionary[img]['trimflat' + str(key)]
                                elif 'trimmed' + str(key)  in dictionary[img]:
                                    image = dictionary[img]['trimmed' + str(key)][0].data
                                    deimosutil.image_plot(image,3,dictionary[img]['OBJECT'])
                                    if pyversion>=3:
                                        input(' see figure 3, ' + img+' ')
                                    else:
                                        raw_input(' see figure 3, '  + img+' ')
                                        
                            if pyversion>=3:                            
                                img0 = input('which image do you want to use to do find the frame curvature [' + str(setup_arc[setup][0]) + ']? ')
                            else:
                                img0 = raw_input('which image do you want to use to do find the frame curvature [' + str(setup_arc[setup][0]) + ']? ')
                            if not img0:
                                img0 = setup_arc[setup][0]

                            # if you do not observe all arc together you can combine the frames 
                            if ',' in img0:
                                import string
                                imgarclist = string.split(img0,',')
                                dictionary['arcmerge_' + str(key)] = dictionary[imgarclist[0]]
                                dictionary['arcmerge_' + str(key)]['trimmed'+str(key)][0].data = dictionary[imgarclist[1]]['trimmed'+str(key)][0].data +\
                                                                                                 dictionary[imgarclist[0]]['trimmed'+str(key)][0].data
                                img0 = 'arcmerge_' + str(key)
                                raw_input('stop here')
                                
                            lambdas[key] = deimosutil.retify_frame(img0, dictionary, key,True)
                            
                        img0 = setup_object[setup][0]
                        if 'trimflat' + str(key)  in dictionary[img]:
                            image = dictionary[img0]['trimflat' + str(key)]
                        else:
                            image = dictionary[img0]['trimmed' + str(key)][0].data

                        order = 2
                        # get the a pixel coordinate near the image center
                        ny, nx = image.shape
                        cy, cx = ny//2, nx//2
                        # create 1d arays of the possible x and y values
                        xs = np.arange(nx)
                        ys = np.arange(ny)
                        # pixel coordinates for each pixel
                        yvals, xvals = np.indices(image.shape)
                            
                        hwidth = 300 # width = 2 * hwidth + 1
                        cols = np.arange(hwidth, nx, 2 * hwidth)
                        cols = cols[1:]
                
                        lambdafit = np.zeros(image.shape)
                        for y in range(ny):
                            c = np.polyfit(cols, lambdas[key][y, cols] - xs[cols], order)
                            lambdafit[y, :] = np.polyval(c, xs) + xs

                        lamb[key] = lambdafit
                        if verbose:
                            if pyversion>=3:
                                input('lambda solution pixel by pixel found')
                            else:
                                raw_input('lambda solution pixel by pixel found')
                                
##############################################################################
                #####################################
                ##########   sky subtraction    ##################
                if stage in [None,'sky']:
                    print('\n#### Sky Subtraction ')                    
                    for key in [3,7]:#lambdas:
                        if not os.path.isdir(_dir):
                            os.mkdir(_dir)
                        if not os.path.isdir(_dir + '/' + str(key)):
                            os.mkdir(_dir + '/' + str(key))
                        for img in setup_object[setup]:
                            print(img,dictionary[img]['OBJECT'],key)
                            dosky = True
                            if 'nosky'+str(key) in dictionary[img] and _force==False:
                                image = dictionary[img]['trimmed' + str(key)][0].data
                                nosky = dictionary[img]['nosky' + str(key)]
                   
                                deimos.deimosutil.image_plot([image,nosky],frame=3,_title='image - nosky')
                   
                                if pyversion>=3:
                                    answ = input('do you want to do the sky subtraction again ? [y/n] [n]')
                                else:
                                    answ = raw_input('do you want to do the sky subtraction again ? [y/n] [n]')
                                if not answ: answ = 'n'
                                if answ in ['n','N','NO','no']:
                                    dosky = False
                            if dosky:
                                imgnosky = re.sub('.fits','',img) + '_' + str(key) + '_nosky.fits'
                                header = dictionary[img]['trimmed' + str(key)][0].header
                                image = dictionary[img]['trimmed' + str(key)][0].data
#                                if 'trimflat' + str(key) in dictionary[img]:
#                                    image = dictionary[img]['trimflat' + str(key)]
#                                elif 'trimmed' + str(key) in dictionary[img]:
#                                    image = dictionary[img]['trimmed' + str(key)][0].data
#                                else:
#                                    print('warning: no trim and no flat filded image found')
                   
                                if 1==1:    
                                    objrows, skymask =  deimosutil.find_sky_object(image, 0.3, key, True)                                
                                    yvals, xvals = np.indices(image.shape)
                                    lambdafit = lamb[key] 
                                    # use the (unmasked) sky background pixels and fit the 2D spline
                                    skyfit = deimosutil.fit_sky(lambdafit[skymask], yvals[skymask], image[skymask])
                                    
                                    # evaluate the 2D sky at every pixel
                                    sky = skyfit.ev(lambdafit, yvals)
                   
                                    if verbose:
                                        deimos.deimosutil.image_plot([image,image-sky],frame=3,_title='image - nosky')
                                        if pyversion>=3:
                                            input('sky subtraction: original sky and residual')
                                        else:
                                            raw_input('sky subtraction: original sky and residual')
                                            
                                    dictionary[img]['sky' + str(key)] = sky
                                    # subtract the sky
                                    nosky = image - sky
                                    dictionary[img]['nosky' + str(key)] = nosky
                   
                                    # nosky
                                    if not os.path.isdir(_dir):
                                        os.mkdir(_dir)
                                    if not os.path.isdir(_dir + '/' + str(key)):
                                        os.mkdir(_dir + '/' + str(key))
                                    imgnosky = re.sub('.fits','',img) + '_' + str(key) + '_nosky.fits'
                                    nosky = dictionary[img]['nosky' + str(key)]
                                    _out = fits.ImageHDU(data=nosky, header= header)
                                    fits.writeto(_dir + '/' + str(key)  + '/' + imgnosky, _out.data,header=_out.header,overwrite='yes')

                else:
                    print('skip sky step')
                    
#####################################################################
########################################  trace  #############################
#   This stage is mainly to get the aperture and background region correct
#   the exact position of the trace will be then improved with possbile a shift
#
                if stage in [None,'trace']:
                    if _iraf:
                        #
                        # use iraf and add to the dictionary all the result
                        #
                        #
                        for img in setup_object[setup]:
                            for key in [3,7]:
                                _ext_trace = False
                                _dispersionline = False
               
                                ## write nosky file from dictionary
                                imgout = re.sub('.fits','',img) + '_' + str(key) + '_nosky.fits'
                                nosky = dictionary[img]['nosky' + str(key)]
                                header = dictionary[img]['trimmed' + str(key)][0].header
                                _out = fits.ImageHDU(data=nosky,header=header)
                                fits.writeto(imgout, _out.data,overwrite='yes')
               
                                ######  trace using iraf and write trace in iraf database
                                dictionary = deimos.irafext.extractspectrum(dictionary,img, key, _ext_trace, _dispersionline, verbose, 'obj')          
                    else:
                        step = 50
                        polyorder = 3
                        sigma = 4
                        niteration = 10
                        for img in setup_object[setup]:
                            for key in [3,7]:
                                print('\n#### Trace ',img)                    
                                print(img,dictionary[img]['OBJECT'],key)
                                dotrace = True
                                if 'peakpos_'+str(key) in dictionary[img] and _force==False:
                                    imm = dictionary[img]['trimmed' + str(key)][0].data
                                    plt.figure(3)
                                    plt.clf()
                                    deimos.deimosutil.image_plot(imm,frame=3,_title=dictionary[img]['OBJECT'])
                                    peakpos1 = dictionary[img]['peakpos_'+str(key)]
                                    xs = np.arange(len(peakpos1))
                                    plt.plot(xs,peakpos1,'-c')
                                    plt.plot(xs,peakpos1 + float(dictionary[img]['aplow_'+str(key)]),'-w')
                                    plt.plot(xs,peakpos1 + float(dictionary[img]['aphigh_'+str(key)]),'-w')
                                    
                                    if pyversion>=3:
                                        answ = input('do you want to trace again (see figure 3) ? [y/n] [n]')
                                    else:
                                        answ = raw_input('do you want to trace again (see figure 3) ? [y/n] [n]')
                                    if not answ: answ = 'n'
                                    if answ in ['n','N','NO','no']:
                                        dotrace = False
                                    plt.clf()
                                
                                if dotrace:
                                    # the trace is better on skysutracted images
                                    _rawdataname = 'nosky'
                                    peakpos1,centerv,highv,fwhmv,dictionary = deimos.deimosutil.tracenew(img, dictionary, key, step, True, polyorder, sigma, niteration,rawdataname=_rawdataname)
                else:
                    print('skip trace')
#####################################################################
#                                    if _nsky:
#                                        print('\n##### Warning: Trace on the trimmed image instead of the sky')
#                                        _rawdataname = 'trimmed'
#                                    else:
#                                data = dictionary[img]['nosky'+str(key)]
#                                center, lower, upper, l1,l2,u1,u2 = deimos.deimosutil.profile(data,dispersion=None)
#                                print(center, lower, upper, l1,l2,u1,u2)                           
#                                if key==3:
#                                    dictionary = deimosutil.trace(img,dictionary, 10, 85, 3, key, True)
#                                else:
#                                    dictionary = deimosutil.trace(img,dictionary, 10, 60, 7, key, True)


#####################################################  extraction  #############################
                if stage in [None,'extract']:
                    for img in setup_object[setup]:
                        for key in [3,7]:
                            print('\n#### Extraction ',img)                    
                            print(img,dictionary[img]['OBJECT'],key)
                            doextraction = True
                            if 'spec_basic'+str(key) in dictionary[img] and _force==False:
                                spec_basic = dictionary[img]['spec_basic' + str(key)]
                                spec_opt = dictionary[img]['spec_opt' + str(key)]
                                xx = np.arange(0,len(spec_basic))
                                
                                plt.figure(1)
                                plt.clf()
                                plt.plot(xx, spec_opt, label='optimal extraction')
                                plt.plot(xx, spec_basic, label='basic extraction')
                                plt.xlabel('pixels')
                                plt.ylabel('counts')
                                plt.legend()
        
                                imm = dictionary[img]['trimmed' + str(key)][0].data
                                plt.figure(3)
                                plt.clf()
                                deimos.deimosutil.image_plot(imm,frame=3,_title=dictionary[img]['OBJECT'])
                                peakpos1 = dictionary[img]['peakpos_'+str(key)]
                                xs = np.arange(len(peakpos1))
                                plt.plot(xx,peakpos1,'-c')
                                plt.plot(xx,peakpos1 + float(dictionary[img]['aplow_'+str(key)]),'-w')
                                plt.plot(xx,peakpos1 + float(dictionary[img]['aphigh_'+str(key)]),'-w')                            
                                
                                if pyversion>=3:
                                    answ = input('do you want to extract again ? [y/n] [n]')
                                else:
                                    answ = raw_input('do you want to extract again ? [y/n] [n]')
                                if not answ: answ = 'n'
                                if answ in ['n','N','NO','no']:
                                    doextraction = False
                                
                            if doextraction:
                                if 'trimflat' + str(key) in dictionary[img]:                           
                                    image = dictionary[img]['trimflat' + str(key) ]
                                elif 'trimmed' + str(key) in dictionary[img]:
                                    image = dictionary[img]['trimmed' + str(key) ][0].data
                                    
                                sky = dictionary[img]['sky' + str(key)]
                                nosky = image - sky
                                ny, nx = nosky.shape
                                xs = np.arange(nx)
                                peak = dictionary[img]['peakpos_' + str(key)]
                                plt.figure(3)
                                plt.clf()
                                deimos.deimosutil.image_plot(nosky,3,dictionary[img]['OBJECT'])
                                plt.plot(xs,peak,'-r')
                                plt.plot(xx,peak + float(dictionary[img]['aplow_'+str(key)]),'-w')
                                plt.plot(xx,peak + float(dictionary[img]['aphigh_'+str(key)]),'-w')
                                bkg = np.array(string.split(dictionary[img]['bckgrintervals_' + str(key)],','),float)
                                for nn in bkg:
                                    plt.plot(xx,peak + nn,'-y')
                                othertrace = None
                                _shift=0
                                if pyversion>=3:
                                    answ = input('trace ok [[y]/n] (see figure 3)? ')
                                else:
                                    answ = raw_input('trace ok [[y]/n]  (see figure 3)? ')
                                    
                                if not answ:
                                    answ='y'
                                if answ in ['Yes','yes','Y','y']:
                                    othertrace = None
                                else:
                                    for image in setup_object[setup]:
                                        print(image,setup)
                                    for image in setup_object[setup]:
                                        _shift=0
                                        peak = dictionary[image]['peakpos_' + str(key)]
                                        plt.figure(3)
                                        plt.clf()
                                        deimos.deimosutil.image_plot(nosky,3)
                                        plt.plot(xs,peak,'-b')
                                        plt.plot(xx,peak + float(dictionary[image]['aplow_'+str(key)]),'-w')
                                        plt.plot(xx,peak + float(dictionary[image]['aphigh_'+str(key)]),'-w')
                                        bkg = np.array(string.split(dictionary[image]['bckgrintervals_' + str(key)],','),float)
                                        for nn in bkg:
                                            plt.plot(xx,peak + nn,'-y')
                                            
                                        if pyversion>=3:
                                            answ0 = input('is trace ok (different object) [y] , [n], [s] (need shift)  [n]? ')
                                        else:
                                            answ0 = raw_input('is trace ok (different object) [y] , [n], [s] (need shift)  [n]? ')
                                            
                                        if not answ0:
                                            answ0='n'
                                        if answ0 in ['y','Y','yes']:
                                            othertrace = image
                                            break
                                        elif answ0 in ['S','s','shift']:
                                            othertrace = image
                                            answ0='s'
                                            while answ0=='s':
                                                _shift=0
                                                if pyversion>=3:
                                                    _shift = input('how much shift ? ')
                                                else:
                                                    _shift = raw_input('how much shift ? [0] ')
                                                    if not _shift:
                                                        _shift = 0
                                                _shift = float(_shift)
                                                plt.plot(xs,peak + _shift ,'.')
                                                if pyversion>=3:
                                                    answ0 = input('is shift ok  y/[n] ?')
                                                else:
                                                    answ0 = raw_input('is shift ok  y/[n] ?')
                                                if answ0 in ['NO','N','n','']:
                                                    answ0='s'
                                            break
                                print(_shift)
                                print(othertrace)
                                if othertrace:
                                    meta ={
                                        'aplow':               dictionary[img]['aplow_' + str(key)],
                                        'bckgrfunc':           dictionary[img]['bckgrfunc_' + str(key)],
                                        'bckgr_low_reject':    dictionary[img]['bckgr_low_reject_' + str(key)],
                                        'displine':            dictionary[img]['displine_' + str(key)],
                                        'aphigh':              dictionary[img]['aphigh_' + str(key)],
                                        'bckgrfunc_iraforder': dictionary[img]['bckgrfunc_iraforder_' + str(key)],
                                        'coeffs':              dictionary[img]['coeffs_' + str(key)],
                                        'bckgrintervals':      dictionary[img]['bckgrintervals_' + str(key)],
                                        'bckgr_niterate':      dictionary[img]['bckgr_niterate_' + str(key)],
                                        'bckgr_high_reject':   dictionary[img]['bckgr_high_reject_' + str(key)],
                                        }
                                    apcent = dictionary[othertrace]['peakpos_'+str(key)]
                                    trace = apcent + _shift
                                    dictionary[img]['peakpos_'+str(key)] = trace
                                    imgtrace = re.sub('.fits','',img) + '_' + str(key) + '_trace.ascii'
                                    output = _dir + '/' + str(key)  + '/' + imgtrace                                
                                    # write the new trace for this object
                                    deimos.deimosutil.writetrace(trace,meta, 'trace', output)
                                    
                  ############################################################################################################                        
                                ## write nosky file from dictionary
                                readnoise = 16
                                gain = 1
                                apmedfiltlength = 61 # no idea what is it
                                colfitorder, scattercut = 15, 25  # no idea what is it
                                if _nsky:
                                   print('\n##### Warning: extract on the trimmed image instead of the sky')
                                   _rawdataname = 'trimmed'
                                else:
                                    _rawdataname = 'nosky'
                                   
                                spec_opt, spec_basic, skybg_opt, spec_var = deimos.irafext.opextract_new(img, 0, 0, False, 1,\
                                                                                                         readnoise, gain, apmedfiltlength,
                                                                                                         colfitorder, scattercut,
                                                                                                         colfit_endmask=10,
                                                                                                         diagnostic= False,
                                                                                                         production= True,\
                                                                                                         other = othertrace, shift=_shift,
                                                                                                         dictionary=dictionary, key=key,
                                                                                                         rawdataname = _rawdataname,
                                                                                                         bckhigh = False, bcklow = False)
                                
                                # add exstraction to the dictionary
                                #  iraf dimensions (to be checked)
                                #  1 optimal extraction
                                #  2 basic extraction
                                #  3 sky
                                #  4 errors
                                dictionary[img]['spec_basic' + str(key)]= spec_basic
                                dictionary[img]['spec_opt' + str(key)]= spec_opt
                                dictionary[img]['skybg_opt' + str(key)]= skybg_opt
                                dictionary[img]['spec_var' + str(key)]= spec_var
                                
                                ########    my simple sum 
                                peakpos = dictionary[img]['peakpos_' + str(key)]
                                xx = np.arange(0,len(peakpos))
                                aplow = dictionary[img]['aplow_' + str(key)]
                                aphigh = dictionary[img]['aphigh_' + str(key)]
                                image  = dictionary[img]['nosky' + str(key)]
                                yvals, xvals = np.indices(image.shape)
                                if 'trimflat' + str(key) in dictionary[img]:                           
                                    skyimage = dictionary[img]['trimflat' + str(key)]
                                elif 'trimmed' + str(key) in dictionary[img]:
                                    skyimage = dictionary[img]['trimmed' + str(key)][0].data
                                    
                                aa = [(yvals> peakpos + aplow) & (yvals < peakpos + aphigh )][0]*1
                                bb = [(yvals< peakpos + aplow) | (yvals > peakpos + aphigh )][0]*1
                                apsum = image * aa
                                skysum = skyimage * bb
                                spec_my = apsum.sum(axis=0)
                                skymy = skysum.sum(axis=0)
                                skymy = (skymy - np.median(skymy))/np.max(skymy)
                                dictionary[img]['mysky' + str(key)]= skymy
                                dictionary[img]['mybasic' + str(key)]= spec_my
                                                            
                                imgout = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_ex.ascii'
                                np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[xx,spec_basic,spec_opt,skybg_opt,spec_var,spec_my,skymy],\
                                           header=' pixel  spec_basic   spec_opt   skybg_opt   spec_var   mybasic  mysky')
                                
                                if verbose:
                                    plt.figure(2)
                                    plt.clf()
                                    deimos.deimosutil.image_plot(skysum,2)
                                    plt.figure(1)
                                    plt.clf()
                                    plt.plot(xx, spec_opt, label='optimal extraction')
                                    plt.plot(xx, spec_basic, label='basic extraction')
                                    plt.xlabel('pixels')
                                    plt.ylabel('counts')
                                    plt.legend()
            
                                    if pyversion>=3:
                                        input('extraction completed')
                                    else:
                                        raw_input('extraction completed')
                else:
                    print('skip extraction')
#    
#####  initial wavelengh calibration #############################
#
# I'm just using a default wavelengh calibration (solution on the top of the script)
# We are missing at least a check with sky line (at the moment sky is plotted vs sky reference, missing cross correlation
#
##################################################################
                if stage in [None,'wave']:
                    print('\n### Select which arc do you want to use ')
                    for img in setup_arc[setup]:
                        if 'trimflat' + str(key)  in dictionary[img]:
                            image = dictionary[img]['trimflat' + str(key)]
                        elif 'trimmed' + str(key)  in dictionary[img]:
                            image = dictionary[img]['trimmed' + str(key)][0].data
                            deimosutil.image_plot(image,3,dictionary[img]['OBJECT'])
                        if pyversion>=3:
                            input(' see figure 3, ' + img+' ')
                        else:
                            raw_input(' see figure 3, '  + img+' ')
                                            
                    if pyversion>=3:                            
                        arc = input('which image do you want to use to do wavelegnth calibration [' + str(setup_arc[setup][0]) + ']? ')
                    else:
                        arc = raw_input('which image do you want to use to do wavelength caibration [' + str(setup_arc[setup][0]) + ']? ')
                    if not arc:
                        arc = setup_arc[setup][0]
        
                    # if you do not observe all arc together you can combine the frames 
                    if ',' in arc:
                        import string
                        imgarclist = string.split(arc,',')
                        dictionary['arcmerge_' + str(key)] = dictionary[imgarclist[0]]
                        dictionary['arcmerge_' + str(key)]['trimmed'+str(key)][0].data = dictionary[imgarclist[1]]['trimmed'+str(key)][0].data +\
                                                                                         dictionary[imgarclist[0]]['trimmed'+str(key)][0].data
                        arc = 'arcmerge_' + str(key)
                        raw_input('stop here')
        
        
                    # load the sky reference
                    skyref = np.genfromtxt(os.path.join(deimos.__path__[0]+'/resources//sky/','UVES_nightsky_lowres.dat'), names='wav, flux')
                    # normalize sky spectrum
                    skyref['flux'] = skyref['flux']-np.min(skyref['flux'])
                    skyref['flux'] = skyref['flux']/np.max(skyref['flux'])
                    # interpolate the sky template
                    skyref_interp = interp1d(skyref['wav'], skyref['flux'], bounds_error=False)
                        
                    sameforall = {}
                    for img in setup_object[setup]:
                        for key in [3,7]:
                            print('\n#### wavelength solution: ',img)                    
                            print(img,dictionary[img]['OBJECT'],key)
        
                            # use my sky for wavelength check for scince frames
                            sky = np.array(dictionary[img]['mysky' + str(key)])
                            # use the flux for wavelength check for standards  frames
                            flux = dictionary[img]['spec_opt' + str(key)]
                            
                            # reject negative values
                            sky[sky<0]=0
                            # normalize my sky spectrum
                            sky = sky - np.min(sky)
                            sky = sky / np.max(sky)
                            
                            answ = False
                            if key in sameforall:
                                if pyversion>=3:                            
                                    answ = input('Do you want to use the same wavelength calibration computed before [y,Y,yes] [y]? ')
                                else:
                                    answ = raw_input('Do you want to use the same wavelength calibration computed before [y,Y,yes] [y]? ')
                                if not answ:
                                    answ = True
                                elif answ in ['Y','y','yes']:
                                    answ = True
                                else:
                                    answ= False
                            else:
                                answ= False
                                
                            if answ is True:
                                # use solution computed before
                                wave = sameforall[key]
                                dowave = False
                            else:
                                # do a new calibration
                                dowave = True
                                
                                if 'wave'+str(key) in dictionary[img] and _force is False:
                                    # there is a calibration already in the dictionary
                                    wave = dictionary[img]['wave' + str(key)]
                                    flux = dictionary[img]['spec_opt' + str(key)]                                                            
                                    plt.figure(1)
                                    plt.clf()
                                    plt.plot(wave,sky,'-r')
                                    plt.plot(skyref['wav'],skyref['flux'],'-b')
                                    
                                    if pyversion>=3:
                                        answ = input('do you want to do wavelength solution again ? [y/n] [n]')
                                    else:
                                        answ = raw_input('do you want to do wavelength solution again ? [y/n] [n]')
                                    if not answ: answ = 'n'
                                    if answ in ['n','N','NO','no']:
                                        dowave = False
                                else:
                                    # wave not defined I really need ot do the wave
                                    dowave = True
                                                                    
                                if dowave:
                                    print('\n#### wavelength solution ',img)
                                    print(arc)
                                    dictionary[img]['arcfile' + str(key)]= arc
                                    print(dictionary[img]['OBJECT'])
                                    
                                    #####    load the arc trimmed image and extract the arc in the same way the object was extracted
                                    peakpos = np.array(dictionary[img]['peakpos_' + str(key)])
                                    aplow = dictionary[img]['aplow_' + str(key)]
                                    aphigh = dictionary[img]['aphigh_' + str(key)]
                                    
                                    if 'trimflat' + str(key) in dictionary[arc]:                           
                                        imagearc = np.array(dictionary[arc]['trimflat' + str(key)])
                                    elif 'trimmed' + str(key) in dictionary[arc]:
                                        imagearc = np.array(dictionary[arc]['trimmed' + str(key)])[0].data
                                        
                                    ny, nx = imagearc.shape
                                    # create 1d arays of the possible x and y values
                                    xs = np.arange(nx)
                                    ys = np.arange(ny)
                                    # pixel coordinates for each pixel                        
                                    yvals, xvals = np.indices(imagearc.shape)
                                    aa = [(yvals> peakpos + aplow) & (yvals < peakpos + aphigh )][0]*1
                                    arcsum = imagearc * aa
                                    arcspec = arcsum.sum(axis=0)
                                    dictionary[img]['arcspec' + str(key)] = arcspec
                                    imgout = 'arc_'+ _dir + '_' + str(key) + 'ascii'
                                    np.savetxt(imgout, np.c_[xs, arcspec ], header='pixel  flux ')
                                    ################################################################
                                    reference = os.path.join(deimos.__path__[0]+'/resources/wave/', ('/').join(setup),str(key),'arc.fits')
                                    initial_solution = deimosutil.poly_arc[key]
                                    radius = 10
                                    edges = 7
                                    deg = 5
                                    
                                    from deimos import deimoswave
                                    finalsolution = deimos.deimoswave.wavesolution(reference, imgout, key, radius, edges, initial_solution, deg)
                                    print(finalsolution)
                                    p = np.poly1d(finalsolution)
                                    wave = p(xs)
                                    wave = sameforall[key]
        
                                    
#################################################   check wave calibration    #################################################
                                # Here I check the wave solution for standard and object separetly
                                shift = 0
                                if 'std' not in dictionary[img].keys():                                
                                    # check if it is monotonically increasing
                                    dxa = np.diff(wave)
                                    if (dxa > 0).all() is False:
                                        print('invert vector')
                                        sky0 = sky[::-1]
                                        wave0 = wave[::-1]
                                    else:
                                        sky0 = sky
                                        wave0 = wave
                                        
                                    guess = (.000001,  1.00001, 0.00001)
                                    bestparams = fmin(deimos.deimoswave.get_lamp_difference, guess, args=(wave0, sky0, skyref_interp), maxiter=10000)
                                    if (dxa > 0).all() is False:
                                        shift = bestparams[0]
                                    else:
                                        shift = (-1) * bestparams[0]
        
                                    print('shift the spectrum of ',shift)        
                                    #  wavelength calibration in the database
                                    wave = p(xs) + shift
                                        
                                    if verbose:
                                        plt.figure(2)
                                        fig2 = plt.figure(2)
                                        fig2.clf()
                                        # compare the reference spectrum and the extracted sky spectrum
                                        ax2 = fig2.add_subplot(2, 1, 1)
                                        ax22 = fig2.add_subplot(2, 1, 2)
                                        ax2.plot(skyref['wav'], skyref['flux']/np.max(skyref['flux']))
                                        ax2.axes.set_ylabel('Flux Density ($10^{16} f_{\lambda}$)')
                                        ax2.axes.set_xlabel('Wavelength ($\AA$)')
                                        ax2.plot(wave, sky0)
                                        
                                        # plot the extracted sky spectrum 
                                        ax22.plot(wave, flux)
                                        ax22.axes.set_ylabel('Counts')
                                        ax22.axes.set_xlabel('wavelenght');
                                        if pyversion>=3:
                                            input('stop here')
                                        else:
                                            raw_input('stop here')
        
                                else:
                                    #
                                    ref_filename = os.path.join(deimos.__path__[0]+'/resources/sky/','std_telluric.fits')
                                    imgout = 'std_'+ _dir + '_' + str(key) + '.ascii'
                                    np.savetxt(imgout, np.c_[wave, flux ], header='wave  flux ')
                                    
                                    shift, scalefactor = deimos.deimoswave.checkwithtelluric(wave, flux , key, ref_filename, guess=(0.001,1.0001), verbose=True)
                                    print ('myshift: '+str(shift))
                                    
                                    print('shift the spectrum of ',shift)        
                                    #  wavelength calibration in the database
                                    wave = p(xs) + shift
        
                                    
                                # adding the wave solution to the dictionary   
                                dictionary[img]['wave' + str(key)]= wave
        
                                # storing the wave solution in the data folder
                                spec_opt = dictionary[img]['spec_opt' + str(key)]
                                spec_basic = dictionary[img]['spec_basic' + str(key)]
                                skybg_opt = dictionary[img]['skybg_opt' + str(key)]
                                spec_var = dictionary[img]['spec_var' + str(key)]
                                skymy = dictionary[img]['mysky' + str(key)]
                                spec_my = dictionary[img]['mybasic' + str(key)]
                                
                                imgout = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_wave.ascii'
                                np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[wave, xs,spec_basic,spec_opt,skybg_opt,spec_var,spec_my,skymy, arcspec],\
                                           header='wave  pixel  spec_basic   spec_opt   skybg_opt   spec_var   mybasic  mysky arcspec')
                                           ## shift ' + str(shift))
                else:
                    print('skip wave')
    #################################################################3####################################    
    #################################################################3####################################
    ############# make response
                if stage in [None,'response']:
                    std = []
                    for img in setup_object[setup]:
                        if 'std' in dictionary[img].keys():
                            std.append(img)
                    ##############################
                    print(std)
                    for img in std:
                        for key in [3,7]:
                            doresponse = True
                            if 'response'+str(key) in dictionary[img] and _force==False:
                                print(img,dictionary[img]['OBJECT'],key)
                                if pyversion>=3:
                                    answ = input('do you want to comute the response again ? [y/n] [n]')
                                else:
                                    answ = raw_input('do you want to comute the response again ? [y/n] [n]')
                                if not answ: answ = 'n'
                                if answ in ['n','N','NO','no']:
                                    doresponse = False
                            if doresponse:
                                std0 = dictionary[img]['std']
                                liststd = glob.glob(_path[0]+'/resources/onedstds/*/'+std0)
                                if not len(liststd):
                                    print('Error: standard name not found')
                                if len(liststd):
                                    dostandard = True
                                    if len(liststd)>1:
                                        plt.figure(1)
                                        plt.clf()
                                        for standard in liststd:
                                            print(standard)
                                            data = np.genfromtxt(standard)
                                            x,y,z = zip(*data)
                                            std_flux = deimos.deimosutil._mag2flux(np.array(x),np.array(y))              
                                            plt.plot(x,std_flux,'-',label=re.sub(_path[0],'',standard))
                                        plt.legend()
                                        if pyversion>=3:
                                            standard = input('few standard file in the database, which one do you want to use?')
                                        else:
                                            standard = raw_input('few standard file in the database, which one do you want to use?')
                                        if not standard:
                                            standard = liststd[0]
                                    else:
                                        standard = liststd[0]  
                                else:
                                    dostandard = False
                            else:
                                dostandard = False
        
                            print(dictionary[img]['OBJECT'])
                            print(dostandard)
                            if dostandard:
                                std_spec = dictionary[img]['spec_opt'+str(key)]
                                _exptime = dictionary[img]['EXPTIME']
                                _airmass = dictionary[img]['AIRMASS']
                                wave = dictionary[img]['wave'+str(key)]
                                    
                                response = deimosutil.DefFluxCal(wave, std_spec, stdstar=re.sub(_path[0]+'/resources/onedstds/','',standard),\
                                                                 mode='spline', polydeg=4, exptime= _exptime, airmass= _airmass,\
                                                                 display=verbose, interactive= verbose)
        
                                data = np.genfromtxt(standard)
                                x,y,z = zip(*data)
                                std_flux = deimos.deimosutil._mag2flux(np.array(x),np.array(y))                        
                                rep = response
                                                                
                                # save response function in the dictionary
                                dictionary[img]['response'+str(key)] = rep
                                dictionary[img]['wave'+str(key)] = wave
        
                                # write response function in the directory
                                imgout = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_response.ascii'
                                np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[wave, rep],\
                                           header='wave  response ')
                                if pyversion>=3:
                                    input('response function applyed to the standard')
                                else:
                                    raw_input('response function applyed to the standard')
                else:
                    print('skip sensitivitity function step')
                #########################################################         
                ######################################################### 
                if stage in [None,'flux']:
                    #  chose sensitivity
                    use_standard = {}
                    for key in [3,7]:
                        std=[]
                        use_standard[key] = None
                        for img in setup_object[setup]:
                            if 'response' + str(key) in dictionary[img] and 'std' in dictionary[img]:
                                std.append(img)
                        if len(std)>1:
                            print(std)
                            if pyversion>=3:
                                img0 = input('which standard do you want to use?')
                            else:
                                img0 = raw_input('which standard do you want to use?')                            
                            if not img0:
                                use_standard[key] = std[0]
                            else:
                                use_standard[key] = img0
                        else:
                            use_standard[key] = std[0]
                            
                    ######################################
                    # apply sensitivity
                    _dir = '_'.join(setup)
                    if not os.path.isdir(_dir):
                        os.mkdir(_dir)
                    for key in [3,7]:
                        if not os.path.isdir(_dir + '/' + str(key)):
                            os.mkdir(_dir + '/' + str(key))
        
                        if use_standard[key]: 
                            for img in setup_object[setup]:
                                doflux = True
                                if 'spec_flux'+str(key) in dictionary[img] and _force==False:
                                    print(img,dictionary[img]['OBJECT'],key)
                                    if pyversion>=3:
                                        answ = input('do you want to flux calibrate again ? [y/n] [n]')
                                    else:
                                        answ = raw_input('do you want to flux calibrate again ? [y/n] [n]')
                                    if not answ: answ = 'n'
                                    if answ in ['n','N','NO','no']:
                                        doflux = False
        
                                if doflux:
                                    spec_opt = np.array(dictionary[img]['spec_opt' + str(key)])
                                    wave = np.array(dictionary[img]['wave' + str(key)])
                                    respfn = dictionary[use_standard[key]]['response'+str(key)]
                                    exptime = dictionary[img]['EXPTIME']
                                    airmass = dictionary[img]['AIRMASS']
                                    
                                    spec_opt = deimos.deimosutil.atmoexp_correction(wave, spec_opt, exptime, airmass, site='mauna', verbose = True)
                                    
                                    dictionary[img]['spec_flux'+str(key)] = spec_opt * respfn
                                    dictionary[img]['response'+str(key)] = respfn
                                    
                                    if verbose:
                                        plt.figure(2)
                                        plt.clf()
                                        fig2 = plt.figure(2)                            
                                        plt.clf()
                                        plt.plot(wave, spec_opt * respfn, label='Calibrated Spectrum')
                                        if pyversion>=3:
                                            input('look final spectrum')
                                        else:
                                            raw_input('look final spectrum')
                
                            ################      write file
                                    spec_basic = dictionary[img]['spec_basic' + str(key)]
                                    spec_opt = dictionary[img]['spec_opt' + str(key)]
                                    skybg_opt = dictionary[img]['skybg_opt' + str(key)]
                                    spec_var = dictionary[img]['spec_var' + str(key)]
                                    skymy = dictionary[img]['mysky' + str(key)]
                                    spec_my = dictionary[img]['mybasic' + str(key)]
                                    wave = dictionary[img]['wave' + str(key)]
                                    response = dictionary[img]['response' + str(key)]
                                    spec_flux = np.array(dictionary[img]['spec_flux' + str(key)])
                                    arcspec = np.array(dictionary[img]['arcspec' + str(key)])
                                    
                                    if key==7:
                                        spec_basic =  spec_basic#[::-1]
                                        spec_opt   =  spec_opt#[::-1]
                                        skybg_opt  =  skybg_opt#[::-1]
                                        spec_var   =  spec_var#[::-1]
                                        skymy      =  skymy#[::-1]
                                        spec_my    =  spec_my#[::-1]
                                        wave       =  wave#[::-1]
                                        response   =  response#[::-1]                          
                                        spec_flux  =  spec_flux#[::-1]                     
                                        arcspec    =  arcspec#[::-1]
                                            
                                    imgout = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_flux.ascii'
                                    np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[wave, xs,spec_basic,spec_opt,skybg_opt,spec_var,\
                                                                                           spec_my,skymy,arcspec, response,spec_flux],\
                                               header='wave  pixel  spec_basic   spec_opt   skybg_opt   spec_var   mybasic  mysky  arcspec  response spec_flux')
                                           #\n# shift ' + str(shift))
                        else:
                             print('no standard for ' + str(key))
                else:
                    print('skip apply sensitivitity function')
###########################################################################################
    
    
    
    
    
    
    
    
    
    
