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
import re
import sys
import numpy as np
from scipy.interpolate import LSQBivariateSpline, LSQUnivariateSpline
from matplotlib import pylab as plt
from astropy.io import fits
import glob
from deimos import __path__ as _path
pyversion = sys.version_info[0]

# check that ds9 is open 
plt.ion()
ds9 = pyds9.DS9()
ds9.set('frame 1')
ds9.set('scale zscale');

# open two plot figures that will be used in the script 
fig1 = plt.figure(1)
fig2 = plt.figure(2)
verbose= True

if __name__ == "__main__":
    parser = OptionParser(usage=usage, description=description, version="%prog 1.0")
    parser.add_option("-d", "--directory", dest="directory", default=None, type="str",
                      help='reduce data in this directory \t [%default]')
    #parser.add_option("-F", "--force", dest="force", action="store_true")
    parser.add_option("--iraf", dest="iraf", action="store_true")
    parser.add_option("-i", "--interactive", action="store_true",
                      dest='interactive', default=False, help='Interactive \t\t\t [%default]')
    option, args = parser.parse_args()
    _interactive = option.interactive
    _iraf = option.iraf
    _directory = option.directory
    
    if _interactive:
        verbose= True
    #
    #  initialize the dictionary with all the infos
    #
    dictionary, setup_object, setup_arc, setup_flat = deimosutil.checkalldata(directory=_directory)
    
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
        if verbose:
            print('READY TO GO!!')
            print(setup_object)
            print(setup_arc)
            print(setup_flat)
        for setup in setup_object:
            if verbose:
                print(setup)
                                
            if setup[1] == 'LVMslitC':
                # trim and rotate dimension 3 and 7
                dictionary = deimosutil.trim_rotate_split(setup_object,setup_flat,setup_arc,dictionary,setup)
    
                # make masterflat3 and masterflat 7
                masterflat3 , masterflat7 =  deimosutil.makeflat(setup_flat,dictionary,setup)
    
                ########################################################
                #   
                ##
                if os.path.isfile('lambdas_' + str(pyversion) + '_3.dat'):
                    print('registered found')
                    lambdas3 = pickle.load(open(os.path.join('./','lambdas_' + str(pyversion) + '_3.dat'), 'rb'))
                else:
                    lambdas3 = None
    
                if os.path.isfile('lambdas_' + str(pyversion) + '_7.dat'):
                    print('registered found')
                    lambdas7 = pickle.load(open(os.path.join('./','lambdas_' + str(pyversion) + '_7.dat'), 'rb'))
                else:
                    lambdas7 = None
    
                if lambdas3 is None:
                    for img in setup_arc[setup]:
                        if 'trimmed3' in dictionary[img]:
                            image = dictionary[img]['trimmed3'].data
                            print(img,dictionary[img]['OBJECT'])
                            deimosutil.image_plot(image)
    
                            if pyversion>=3:
                                input('stop')
                            else:
                                raw_input('stop')
                                
                    if pyversion>=3:                            
                        img0 = input('which image to do the sky correction ? ')
                    else:
                        img0 = raw_input('which image to do the sky correction ? ')
                        
                    lambdas3 = deimosutil.retify_frame(img0, dictionary, 3,True)
                #####################
                if lambdas7 is None:
                    for img in setup_arc[setup]:
                        if 'trimmed7' in dictionary[img]:
                            image = dictionary[img]['trimmed7'].data
                            print(img,dictionary[img]['OBJECT'])
                            deimosutil.image_plot(image)
                            if pyversion>=3:
                                input('stop')
                            else:
                                raw_input('stop')
                                
                    if pyversion>=3:
                        img0 = input('which image to do the sky correction ? ')
                    else:
                        img0 = raw_input('which image to do the sky correction ? ')
                        
                    lambdas7 = deimosutil.retify_frame(img0,dictionary, 7,True)
    
                lambdas = {3: lambdas3,\
                           7: lambdas7}
                
                #####################################
                ##########   sky subtraction    ##################
                for key in [3,7]:#lambdas:
                    img0 = setup_object[setup][0]
                    image = dictionary[img0]['trimmed' + str(key)].data
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
                    if verbose:
                        if pyversion>=3:
                            input('lambda solution pixel by pixel found')
                        else:
                            raw_input('lambda solution pixel by pixel found')
                            
                    for img in setup_object[setup]:
                        if 'trimmed' + str(key) in dictionary[img]:
                            image = dictionary[img]['trimmed' + str(key)].data
                            objrows, skymask =  deimosutil.find_sky_object(image, 0.3, key, True)
                            
                            yvals, xvals = np.indices(image.shape)
                            # use the (unmasked) sky background pixels and fit the 2D spline
                            skyfit = deimosutil.fit_sky(lambdafit[skymask], yvals[skymask], image[skymask])
                            
                            # evaluate the 2D sky at every pixel
                            sky = skyfit.ev(lambdafit, yvals)
    
                            if verbose:
                                #plot the image, sky model, and differece (and/or display in DS9)
                                ds9.set('frame 1')
                                ds9.set_np2arr(image)
                                ds9.set('frame 2')
                                ds9.set_np2arr(sky)
                                ds9.set('frame 3')
                                ds9.set_np2arr(image - sky)
                                if pyversion>=3:
                                    input('stop sky subtraction: original sky and residual in ds9')
                                else:
                                    raw_input('stop sky subtraction: original sky and residual in ds9')
                                    
                            dictionary[img]['sky' + str(key)] = sky
                            # subtract the sky
                            nosky = image - sky
                            dictionary[img]['nosky' + str(key)] = nosky

                ###########  trace  #############################
                if _iraf:
                    print('USE IRAF, TO BE IMPLEMENTED')
                    #
                    # use iraf and add to the dictionary all the result
                    #
                    #
                    sys.exit()
                else:                    
                    for img in setup_object[setup]:
                        for key in [3,7]:
                            if key==3:
                                dictionary = deimosutil.trace(img,dictionary, 10, 85, 3, key, True)
                            else:
                                dictionary = deimosutil.trace(img,dictionary, 10, 60, 7, key, True)
    
                #####  extraction  #############################
                if _iraf:
                    print('USE IRAF, TO BE IMPLEMENTED')
                    #
                    # use iraf and add to the dictionary all the result
                    #
                    #
                    sys.exit()
                else:
                    for img in setup_object[setup]:
                        print('\n#### Extraction ',img)
                        print(img,dictionary[img]['OBJECT'])
                        for key in [3,7]:
                            sky = dictionary[img]['sky' + str(key)]
                            image = dictionary[img]['trimmed' + str(key) ]
                            nosky = image - sky
                            ny, nx = nosky.shape
                            xs = np.arange(nx)
                            peak = dictionary[img]['peakpos' + str(key)]
                            plt.figure(1)
                            plt.clf()
                            deimos.deimosutil.image_plot(nosky)
                            plt.plot(xs,peak,'.r')
                            othertrace = None
                            _shift=0
                            if pyversion>=3:
                                answ = input('trace ok [[y]/n]? ')
                            else:
                                answ = raw_input('trace ok [[y]/n]? ')
                                
                            if not answ:
                                answ='y'
                            if answ in ['Yes','yes','Y','y']:
                                othertrace = None
                            else:
                                for image in setup_object[setup]:
                                    print(image)
                                    _shift=0
                                    peak = dictionary[image]['peakpos' + str(key)]
                                    plt.clf()
                                    deimos.deimosutil.image_plot(nosky)
                                    plt.plot(xs,peak,'.')
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
                                            
                                        othertrace = image
                                        break
        
                            print(_shift)
                            print(othertrace)
                            #
                            # Shift is given interactively, but I still need to implement in the extract definition 
                            # 
                            dictionary = deimosutil.extract(img,dictionary, key, 30, 30, othertrace, shift=_shift, verbose=True)
    
                #####  initial wavelengh calibration #############################
                #
                # I'm just using a default wavelengh calibration (solution on the top of the script)
                # We are missing at least a check with sky line (at the moment sky is plotted vs sky reference, missing cross correlation
                #
                ##################################################################
                arc = setup_arc[setup][1]
                for img in setup_object[setup]:
                    for key in [3,7]:
                        print('\n#### wavelength solution ',img)
                        image = np.array(dictionary[arc]['trimmed' + str(key)])
                        slitpos = np.array(dictionary[img]['trace' + str(key)])
                        sky = np.array(dictionary[img]['skybg_opt' + str(key)])
                        spec_opt = np.array(dictionary[img]['spec_opt' + str(key)])
                        
                        ######### extract arc with the object
                        # get the a pixel coordinate near the image center
                        ny, nx = image.shape
                        cy, cx = ny//2, nx//2
                        # create 1d arays of the possible x and y values
                        xs = np.arange(nx)
                        ys = np.arange(ny)
                        # pixel coordinates for each pixel
                        yvals, xvals = np.indices(image.shape)
                        # select which rows to sum
                        w = (slitpos > -10) & (slitpos < 10)
                        ymin, ymax = yvals[w].min(), yvals[w].max()
                        # calculate the sum
                        spec_basic = image[ymin:ymax, :].sum(axis=0)
                        dictionary[img]['arcspec' + str(key)]= spec_basic
                        dictionary[img]['arcfile' + str(key)]= arc
                        np.savetxt('arc_' + str(key) + '.txt',np.c_[xs,spec_basic])
                        np.savetxt('skybgopt_' + str(key) + '.txt',np.c_[xs,sky])
    
                        #  initial wavelength calibration
                        p = np.poly1d(deimosutil.poly_arc[key])
                        dictionary[img]['wave' + str(key)]= p(xs)
                        ######### wavelenght
                        skyref = np.genfromtxt(os.path.join('./','UVES_nightsky_lowres.dat'), names='wav, flux')
    #
    #                    if verbose:
                        if True:
                            # compare the reference spectrum and the extracted sky spectrum
                            plt.figure(2)
                            fig2 = plt.figure(2)
                            fig2.clf()
                            ax2 = fig2.add_subplot(2, 1, 1)
                            ax22 = fig2.add_subplot(2, 1, 2)
                            ax2.plot(skyref['wav'], skyref['flux'])
                            ax2.axes.set_ylabel('Flux Density ($10^{16} f_{\lambda}$)')
                            ax2.axes.set_xlabel('Wavelength ($\AA$)')
                            sky1 = (sky - np.percentile(sky,0.1))/np.max(sky)
                            ax2.plot(p(xs), sky1)
    
                            # plot the extracted sky spectrum 
                            ax22.plot(p(xs), spec_opt)
                            ax22.axes.set_ylabel('Counts')
                            ax22.axes.set_xlabel('wavelenght');
                            if pyversion>=3:
                                input('stop here')
                            else:
                                raw_input('stop here')
                ##################################################################
                # sensitivity function
                from astropy.coordinates import SkyCoord
                from astropy import units as u
                std, rastd, decstd, magstd = deimosutil.readstandard('standard_deimos_mab.txt')
                scal = np.pi / 180.
                for img in setup_object[setup]:
                    for key in [3,7]:
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
                            std0 = std[np.argmin(dd)]
                            liststd = glob.glob(_path[0]+'/resources/onedstds/*/'+std0)
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
                                        standard = input('which standard do you want to use?')
                                    else:
                                        standard = raw_input('which standard do you want to use?')
                                    if not standard:
                                        standard = liststd[0]
                                else:
                                    standard = liststd[0]  
    
                            else:
                                dostandard = False
                        else:
                            dostandard = False
    
                        if dostandard:
                            std_spec = dictionary[img]['spec_opt'+str(key)]
                            wavs = dictionary[img]['wave'+str(key)]
                            if key==7:
                                wavs = wavs[::-1]
                                std_spec = std_spec[::-1]
                            plt.clf()
                            plt.plot(wavs, std_spec)
                            plt.xlabel('Wavelength ($\AA$)')
                            plt.ylabel('Counts');
                            
                            if pyversion>=3:
                                input('standard spectrum')
                            else:
                                raw_input('standard spectrum')
                                
                            print(standard)
                            response = deimosutil.DefFluxCal(wavs, std_spec, stdstar=re.sub(_path[0]+'/resources/onedstds/','',standard),\
                                                             mode='spline', polydeg=9, display=verbose)
    
                            data = np.genfromtxt(standard)
                            x,y,z = zip(*data)
                            std_flux = deimos.deimosutil._mag2flux(np.array(x),np.array(y))                        
                            plt.clf()
                            plt.plot(x,std_flux,'-r')
                            plt.plot(wavs,std_spec*response,'-b')
                            if pyversion>=3:
                                input('standard spectrum: done')
                            else:
                                raw_input('standard spectrum: done')
                                
                            if key==7:
                                dictionary[img]['response'+str(key)] = response[::-1]
                            else:
                                dictionary[img]['response'+str(key)] = response
                            if pyversion>=3:
                                input('response function applyed to the standard')
                            else:
                                raw_input('response function applyed to the standard')
                        else:
                            print('object')
    
                #################### apply sensitivity            
                for key in [3,7]:
                    std=[]
                    science=[]
                    for img in setup_object[setup]:
                        if 'response'+str(key) in dictionary[img]:
                            std.append(img)
                        else:
                            science.append(img)
                    if len(std)>1:
                        print(std)
                        if pyversion>=3:
                            img0 = input('which standard do you want to use?')
                        else:
                            img0 = raw_input('which standard do you want to use?')
                            
                        if not img0:
                            img0 = std[0]
                    else:
                        img0 = std[0]
                    for img in science+std:
                        spec_opt = np.array(dictionary[img]['spec_opt' + str(key)])
                        wavs = np.array(dictionary[img]['wave' + str(key)])
                        respfn = dictionary[img0]['response'+str(key)]
                        plt.figure(2)
                        plt.clf()
                        fig2 = plt.figure(2)                            
                        plt.clf()
                        plt.plot(wavs, spec_opt * respfn, label='Calibrated Spectrum')
                        dictionary[img]['spec_flux'+str(key)] = spec_opt * respfn
                        if pyversion>=3:
                            input('look final spectrum')
                        else:
                            raw_input('look final spectrum')
    
                ################      write file
                _dir = '_'.join(setup)
                if not os.path.isdir(_dir):
                    os.mkdir(_dir)
                for img in setup_object[setup]:
                    for key in [3,7]:
                        if not os.path.isdir(_dir + '/' + str(key)):
                            os.mkdir(_dir + '/' + str(key))
    
                        # trimmed
                        imgout = re.sub('.fits','',img) + '_' + str(key) + '_trimmed.fits'
                        hdu = dictionary[img]['trimmed' + str(key)]
                        hdu2 = dictionary[img]['nosky' + str(key)]
                        _out = fits.ImageHDU(data=hdu.data, header=hdu.header)
                        fits.writeto(_dir + '/' + str(key)  + '/' + imgout, _out.data,header=_out.header,overwrite='yes')
    
                        # nosky
                        imgout = re.sub('.fits','',img) + '_' + str(key) + '_nosky.fits'
                        
                        _out = fits.ImageHDU(data=hdu2.data, header=hdu.header)
                        fits.writeto(_dir + '/' + str(key)  + '/' + imgout, _out.data,header=_out.header,overwrite='yes')
    
    
                        spec_opt = np.array(dictionary[img]['spec_opt' + str(key)])
                        spec_flux = np.array(dictionary[img]['spec_flux' + str(key)])
                        wavs = np.array(dictionary[img]['wave' + str(key)])
    
                        if key==7:
                            wavs = wavs[::-1]
                            spec_opt = spec_opt[::-1]
                            spec_flux = spec_flux[::-1]
                        
                        imgout = re.sub('.fits','',img) + '_' + str(key) + '_ex.ascii'
                        np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[wavs,spec_opt])
    
                        imgout = re.sub('.fits','',img) + '_' + str(key) + '_f.ascii'
                        np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[wavs,spec_flux])
    
###########################################################################################
    
    
    
    
    
    
    
    
    
    
