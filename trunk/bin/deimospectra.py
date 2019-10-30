#!/usr/bin/env python
#
#  python 3 script for deimos reduction
#
###############################################################
import deimos
from deimos import deimosutil
import pyds9
import os
import pickle
import re
import numpy as np
from scipy.interpolate import LSQBivariateSpline, LSQUnivariateSpline
from matplotlib import pylab as plt
from astropy.io import fits
import glob
from deimos import __path__ as _path


plt.ion()

ds9 = pyds9.DS9()
ds9.set('frame 1')
#ds9.set_np2arr(image)
ds9.set('scale zscale');

fig1 = plt.figure(1)
fig2 = plt.figure(2)


verbose= True
dictionary, setup_object, setup_arc, setup_flat = deimosutil.checkalldata(directory=False)

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
            if os.path.isfile('lambdas3.dat'):
                print('registered found')
                lambdas3 = pickle.load(open(os.path.join('./','lambdas3.dat'), 'rb'))
            else:
                lambdas3 = None

            if os.path.isfile('lambdas7.dat'):
                print('registered found')
                lambdas7 = pickle.load(open(os.path.join('./','lambdas7.dat'), 'rb'))
            else:
                lambdas7 = None

            if lambdas3 is None:
                for img in setup_arc[setup]:
                    if 'trimmed3' in dictionary[img]:
                        image = dictionary[img]['trimmed3'].data
                        print(img,dictionary[img]['OBJECT'])
                        deimosutil.image_plot(image)
                        input('stop')
                    
                img0 = input('which image to do the sky correction ? ')
                lambdas3 = deimosutil.retify_frame(img0, dictionary, 3,True)
            #####################
            if lambdas7 is None:
                for img in setup_arc[setup]:
                    if 'trimmed7' in dictionary[img]:
                        image = dictionary[img]['trimmed7'].data
                        print(img,dictionary[img]['OBJECT'])
                        deimosutil.image_plot(image)
                        input('stop')
                    
                img0 = input('which image to do the sky correction ? ')
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
                    input('lambda solution pixel by pixel found')
                
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
                            input('stop sky subtraction: original sky and residual in ds9')
                            
                        dictionary[img]['sky' + str(key)] = sky
                        # subtract the sky
                        nosky = image - sky
                        dictionary[img]['nosky' + str(key)] = nosky
                    
            #####  trace  #############################
            for img in setup_object[setup]:
                for key in [3,7]:
                    if key==3:
                        dictionary = deimosutil.trace(img,dictionary, 10, 85, 3, key,False)
                    else:
                        dictionary = deimosutil.trace(img,dictionary, 10, 60, 3, key, True)

            #####  extraction  ############################# 
            for img in setup_object[setup]:
                print('\n#### Extraction ',img)
                print(img,dictionary[img]['OBJECT'])
                for key in [3,7]:
                    sky = dictionary[img]['sky' + str(key)].data
                    image = dictionary[img]['trimmed' + str(key) ].data
                    nosky = image - sky
                    ny, nx = nosky.shape
                    xs = np.arange(nx)
                    peak = dictionary[img]['peakpos' + str(key)]
                    plt.figure(1)
                    plt.clf()
                    deimos.deimosutil.image_plot(nosky)
                    plt.plot(xs,peak,'.r')
                    othertrace = None
                    answ = input('trace ok [[y]/n]? ')
                    if not answ:
                        answ='y'
                    if answ in ['Yes','yes','Y','y']:
                        othertrace = None
                    else:
                        for image in setup_object[setup]:
                            print(image)
                            peak = dictionary[image]['peakpos' + str(key)]
                            plt.clf()
                            deimos.deimosutil.image_plot(nosky)
                            plt.plot(xs,peak,'.')
                            answ0 = input('trace ok [y/[n]]? ')
                            if not answ0:
                                answ0='n'
                            if answ0 in ['y','Y','yes']:
                                othertrace = image
                                break
                                
                    print(othertrace)
                    dictionary = deimosutil.extract(img,dictionary, key, 30, 30, othertrace, True)

            #####  initial wavelengh calibration #############################                     
            arc = setup_arc[setup][1]
            for img in setup_object[setup]:
                for key in [3,7]:
                    print('\n#### wavelength solution ',img)
                    image = np.array(dictionary[arc]['trimmed' + str(key)].data)
                    slitpos = np.array(dictionary[img]['trace' + str(key)].data)
                    sky = np.array(dictionary[img]['skybg_opt' + str(key)].data)
                    spec_opt = np.array(dictionary[img]['spec_opt' + str(key)].data)
                    
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
                        input('stop here')

                        
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
                                standard = input('which standard do you want to use?')
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
                        input('standard spectrum')
                        print(standard)
                        response = deimosutil.DefFluxCal(wavs, std_spec, stdstar=re.sub(_path[0]+'/resources/onedstds/','',standard),\
                                                         mode='spline', polydeg=9, display=verbose)

                        data = np.genfromtxt(standard)
                        x,y,z = zip(*data)
                        std_flux = deimos.deimosutil._mag2flux(np.array(x),np.array(y))                        
                        plt.clf()
                        plt.plot(x,std_flux,'-r')
                        plt.plot(wavs,std_spec*response,'-b')
                        input('standard spectrum: done')
                        
                        if key==7:
                            dictionary[img]['response'+str(key)] = response[::-1]
                        else:
                            dictionary[img]['response'+str(key)] = response                            
                        input('response function applyed to the standard')
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
                    img0 = input('which standard do you want to use?')
                    if not img0:
                        img0 = std[0]
                else:
                    img0 = std[0]
                for img in science+std:
                    spec_opt = np.array(dictionary[img]['spec_opt' + str(key)].data)
                    wavs = np.array(dictionary[img]['wave' + str(key)])
                    respfn = dictionary[img0]['response'+str(key)]
                    plt.figure(2)
                    plt.clf()
                    fig2 = plt.figure(2)                            
                    plt.clf()
                    plt.plot(wavs, spec_opt * respfn, label='Calibrated Spectrum')
                    dictionary[img]['spec_flux'+str(key)] = std_opt * respfn
                    input('look final spectrum')


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


                    spec_opt = np.array(dictionary[img]['spec_opt' + str(key)].data)
                    spec_flux = np.array(dictionary[img]['spec_flux' + str(key)].data)
                    wavs = np.array(dictionary[img]['wave' + str(key)])

                    if key==7:
                        wavs = wavs[::-1]
                        spec_opt = spec_opt[::-1]
                        spec_flux = spec_flux[::-1]
                    
                    imgout = re.sub('.fits','',img) + '_' + str(key) + '_ex.ascii'
                    np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[wavs,spec_opt])

                    imgout = re.sub('.fits','',img) + '_' + str(key) + '_f.ascii'
                    np.savetxt(_dir + '/' + str(key)  + '/' + imgout,np.c_[wavs,spec_flux])


########################################
                        
#                        print(dictionary[img].keys(),dictionary[img]['OBJECT'])                        
#                        stdfile = deimos.__path__[0]+'/standard/'+std[np.argmin(dd)]
#                        data = np.genfromtxt(stdfile)
#                        x,y,z = zip(*data)
                        ###########
                        ########
#                        dtype = []
#                        dtype.append( ('wav', float) )
#                        dtype.append( ('flux', float) ) # units are ergs/cm/cm/s/A * 10**16
#                        dtype.append( ('eflux', float) )
#                        dtype.append( ('dlam', float) )
#                        calspec = np.genfromtxt('ftp://ftp.eso.org/pub/stecf/standards/okestan/fbd28d4211.dat', dtype=dtype)
#                        plt.figure(2)
#                        plt.clf()
#                        fig2 = plt.figure(2)                            
#                        plt.plot(calspec['wav'], calspec['flux']);
#                        plt.xlabel('Wavelength ($\AA$)')
#                        plt.ylabel('$10^{16} * F_{\lambda}$');
#                        plt.yscale('log');
#                        input('from archive')
#                        # fit a spline to the tabulated spectrum
#                        t = np.arange(calspec['wav'][1], calspec['wav'][-2], np.int(np.median(calspec['dlam'])))
#                        stdflux = LSQUnivariateSpline(calspec['wav'], calspec['flux'], t, calspec['eflux'])
#                        # get the counts to flux density ratios in each wavelength bin
#                        # exclude significant line features (fluxes near these depend on spectral resolution)
#                        ratios = std_spec / stdflux(wavs)
#                        w = (wavs > calspec['wav'].min()) \
#                            & (wavs < calspec['wav'].max()) \
#                            & (np.abs(wavs - 7650) > 70) \
#                            & (np.abs(wavs - 6900) > 40) \
#                            & (np.abs(wavs - 6563) > 40) \
#                        # fit a spline to the ratios to determine the response function
#                        t = wavs[w][1:-2:50]
#                        print(wavs[w])
#                        print(len(t))
#                        respfn = LSQUnivariateSpline(wavs[w], ratios[w], t[1:-1])
#                        plt.figure(1)
#                        plt.clf()
#                        fig1 = plt.figure(1)                            
#                        plt.plot(wavs[w], ratios[w], 'ro')
#                        xwav = np.linspace(wavs[w][1], wavs[w][-1], 1000)
#                        plt.plot(xwav, respfn(xwav));
#                        plt.xlabel('Wavelength ($\AA$)')
#                        plt.ylabel('Response Function (counts/$F_{\lambda}$)');
#                        input('response function')                       
#                        # compare the tabulated and extracted flux densities (applying the response function)
#                        plt.figure(2)
#                        plt.clf()
#                        fig2 = plt.figure(2)                            
#                        plt.clf()
#                        plt.plot(calspec['wav'], calspec['flux'], label='Tabulated (published) Spectrum');
#                        plt.plot(wavs, std_spec / respfn(wavs), label='Extracted Spectrum')
#                        plt.xlim(5300, 9200) # standard star values are only tabulated out to ~9200 A
#                        plt.ylim(0, 1000)
#                        plt.xlabel('Wavelength ($\AA$)')
#                        plt.ylabel('$F_{\lambda}$')
#                        plt.ylim(50, 1000)
#                        plt.yscale('log')
#                        plt.legend();

                    
