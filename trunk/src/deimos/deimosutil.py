import astropy
from astropy.io import fits
import glob
import os
import re
from matplotlib import pylab as plt
import numpy as np
import math
from scipy.signal import find_peaks_cwt
import ccdproc
from astropy.nddata import CCDData
from astropy import units as u
from scipy.interpolate import LSQBivariateSpline, LSQUnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import SmoothBivariateSpline
from astropy.convolution import convolve, Box1DKernel
from astropy.stats import sigma_clip
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import pickle
from pylab import polyfit, polyval
from astropy.stats import sigma_clipped_stats
from deimos import __path__ as _path
import sys
import pyds9
import matplotlib.gridspec as gridspec

pyversion = sys.version_info[0]



poly_arc = {3: np.array([  1.03773471e-05,   5.78487274e-01,   4.45847046e+03]),
#            7: np.array([  2.30350858e-05,  -7.64099597e-01,   9.79141140e+03]),
            7: np.array([ -1.39838149e-13,   1.83212231e-09,  -8.83011172e-06, -6.28779911e-01,   9.64233695e+03])}


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


###################################################

def get_profile_spl(dls, stamp):
    # need to sort the data (and weights) so that the x values are increasing
    x, y = dls.ravel(), stamp.ravel()
    weights = np.sqrt(np.abs(y)) # not technically optimal for coadded data, but ok
    wsort = x.argsort()
    x, y, weights = x[wsort], y[wsort], weights[wsort]
    
    # set locations for spline knots
    t = np.linspace(x.min() + 1, x.max() - 1, np.int(x.max() - x.min()))
    spl = LSQUnivariateSpline(x, y, t, weights)
    return x, y, spl

############################################

def get_dl_model(params, dxs, dys):
    return dxs + params[0] * dys + params[1] * dys ** 2

###########################################

def check_dl_model(params, dxs, dys, stamp):
    dls = get_dl_model(params, dxs, dys)
    x, y, spl = get_profile_spl(dls, stamp)
    chisq = np.sum((stamp - spl(dls)) ** 2 / np.sqrt(np.abs(stamp)))
    return chisq / (stamp.size - len(params))

###############################################################################
# function to fit a 2D spline
def fit_sky(xvals, yvals, image, ky=1, dx=0.5):
    # select knot points in the x (wavelength) direction
    tx = np.arange(xvals.min() + 2, xvals.max() - 2, dx)
    
    # select knot points in the y (spatial) direction
    ty = [] # if there are no knots, the fit will be a poly nomial of degree ky
    
    # fit the 2D spline
    return LSQBivariateSpline(xvals.ravel(), yvals.ravel(), image.ravel(), tx, ty, ky=ky)

###########################

def checkalldata(directory=False,verbose=False):
    from astropy.io import ascii             
    if directory:
        imglist = glob.glob(directory + '*')
    else:
        imglist = glob.glob('*fits')
        
    listhd = ['GRATENAM','SLMSKNAM','MJD-OBS','DATE-OBS','EXPTIME','AIRMASS','OBJECT',\
              'SLSELNAM','TARGNAME','LAMPS','FLAMPS','OBSTYPE','IMTYPE','RA','DEC']
    dictionary={}
    for img in imglist:
        dictionary[img]={}
        hdu = fits.open(img)
        dictionary[img]['fits']= hdu
        for _header in listhd:
            if hdu[0].header.get(_header) is not None:
                dictionary[img][_header]= hdu[0].header.get(_header)
            else:
                dictionary[img][_header]= None
               
        dictionary[img]['type']= None
        if dictionary[img]['SLMSKNAM'] =='GOH_X':
            dictionary[img]['type']= 'focus'
        else:
            # select arcs
            if dictionary[img]['OBSTYPE'] =='Line':
                if dictionary[img]['LAMPS']!=['Off']:
                    dictionary[img]['type']= 'arc'
            #   select flats
            if dictionary[img]['OBSTYPE'] =='IntFlat':
                if hdu[0].header.get('FLAMPS')!=['Off']:
                    dictionary[img]['type']= 'flat'
            #   select objecs
            if dictionary[img]['OBSTYPE'] =='Object':
                dictionary[img]['type']= 'object'
                
    setup_object={}
    setup_flat={}
    setup_arc={}
    skip=[]
    for img in dictionary:
        if dictionary[img]['type'] is None:
            skip.append(img) 
            if verbose:
                print('skip file ' + str(img))
#            print(dictionary[img])
#            if pyversion>=3:
#                answ = input('what is it  [S]kip/[O]bject/[A]rc/[F]lat]  [S] ? ')
#                if not answ: answ='S'
#            else:
#                answ = raw_input('what is it  [S]kip/[O]bject/[A]rc/[F]lat]  [S] ? ')
#                if not answ: answ='S'
        else:
            _grism = dictionary[img]['GRATENAM']
            _slit = dictionary[img]['SLMSKNAM']        
            setup = (_grism,_slit)
            if dictionary[img]['type']=='object':
                if (_grism,_slit) not in setup_object:
                    setup_object[_grism,_slit]=[]
                setup_object[_grism,_slit].append(img)
                
            if dictionary[img]['type']=='arc':
                if (_grism,_slit) not in setup_arc:
                    setup_arc[_grism,_slit]=[]
                setup_arc[_grism,_slit].append(img)
        
            if dictionary[img]['type']=='flat':
                if (_grism,_slit) not in setup_flat:
                    setup_flat[_grism,_slit]=[]
                setup_flat[_grism,_slit].append(img)
        
            _dir = '_'.join(setup)
            print(img,setup,dictionary[img]['OBJECT'])
            
            for key in ['3','7']:
                imgtrimmed = re.sub('.fits','',img) + '_' + str(key) + '_trimmed.fits'
                imgnosky   = re.sub('.fits','',img) + '_' + str(key) + '_nosky.fits'
                imgtrace   = re.sub('.fits','',img) + '_' + str(key) + '_trace.ascii'
                imgex      = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_ex.ascii'
                imgwave    = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_wave.ascii'
                imgresponse = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_response.ascii'
                imgflux    = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_flux.ascii'

                if os.path.isfile(_dir + '/' + str(key) + '/' + imgtrimmed):
                    hdu = fits.open(_dir + '/' + str(key)  + '/' + imgtrimmed)
                    dictionary[img]['trimmed' + str(key)] = hdu
        
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgnosky):
                    hdu = fits.open(_dir + '/' + str(key)  + '/' + imgnosky)
                    nosky = hdu[0].data
                    dictionary[img]['nosky' + str(key)] = nosky
                    hdu1 = fits.open(_dir + '/' + str(key)  + '/' + imgtrimmed)
                    trimmed = hdu1[0].data
                    sky = trimmed - nosky
                    dictionary[img]['sky' + str(key)] = sky

                    
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgtrace):
                    aa, meta = readtrace(_dir + '/' + str(key) + '/' + imgtrace)
                    for key1 in meta:
                        if key1 in ['aplow','displine','aphigh']:
                            dictionary[img][key1 + '_' + str(key)] = float(meta[key1])
                        else:    
                            dictionary[img][key1 + '_' + str(key)] = re.sub('\[?]?','', meta[key1])
                        dictionary[img]['peakpos_'+str(key)] = aa
                        
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgflux):
                    aa = ascii.read(_dir + '/' + str(key) + '/' + imgflux)
                    if key==7:
#                        dictionary[img]['spec_basic' + str(key)] = aa['spec_basic']#[::-1]
#                        dictionary[img]['spec_opt' + str(key)] = aa['spec_opt']#[::-1]
#                        dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt']#[::-1]
#                        dictionary[img]['spec_var' + str(key)] = aa['spec_var']#[::-1]
#                        dictionary[img]['mysky' + str(key)] = aa['mysky']#[::-1]
#                        dictionary[img]['mybasic' + str(key)] = aa['mybasic']#[::-1]
#                        dictionary[img]['wave' + str(key)] = aa['wave']#[::-1]
#                        dictionary[img]['arcspec' + str(key)] = aa['arcspec']#[::-1]
                        dictionary[img]['response' + str(key)] = aa['response']#[::-1]
                        dictionary[img]['spec_flux' + str(key)] = aa['spec_flux']#[::-1]
                    else:
#                        dictionary[img]['spec_basic' + str(key)] = aa['spec_basic']
#                        dictionary[img]['spec_opt' + str(key)] = aa['spec_opt']
#                        dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt']
#                        dictionary[img]['spec_var' + str(key)] = aa['spec_var']
#                        dictionary[img]['mysky' + str(key)] = aa['mysky']
#                        dictionary[img]['mybasic' + str(key)] = aa['mybasic']
#                        dictionary[img]['wave' + str(key)] = aa['wave']
#                        dictionary[img]['arcspec' + str(key)] = aa['arcspec']
                        dictionary[img]['response' + str(key)] = aa['response']
                        dictionary[img]['spec_flux' + str(key)] = aa['spec_flux']
                        
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgwave):
                    aa = ascii.read(_dir + '/' + str(key) + '/' + imgwave)
#                    dictionary[img]['spec_basic' + str(key)] = aa['spec_basic']
#                    dictionary[img]['spec_opt' + str(key)] = aa['spec_opt']
#                    dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt']
#                    dictionary[img]['spec_var' + str(key)] = aa['spec_var']
#                    dictionary[img]['mysky' + str(key)] = aa['mysky']
#                    dictionary[img]['mybasic' + str(key)] = aa['mybasic']
                    dictionary[img]['wave' + str(key)] = aa['wave']
                    dictionary[img]['arcspec' + str(key)] = aa['arcspec']
                      
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgex):
                    aa = ascii.read(_dir + '/' + str(key) + '/' + imgex)
                    dictionary[img]['spec_basic' + str(key)] = aa['spec_basic']
                    dictionary[img]['spec_opt' + str(key)] = aa['spec_opt']
                    dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt']
                    dictionary[img]['spec_var' + str(key)] = aa['spec_var']
                    dictionary[img]['mysky' + str(key)] = aa['mysky']
                    dictionary[img]['mybasic' + str(key)] = aa['mybasic']
                    
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgresponse):
                    aa = ascii.read(_dir + '/' + str(key) + '/' + imgresponse)
#                    dictionary[img]['wave' + str(key)] = aa['wave']
                    dictionary[img]['response' + str(key)] = aa['response']

                    
    for img in skip:
        del dictionary[img]
    return dictionary, setup_object, setup_arc, setup_flat

########################################################################

def trim_rotate_split(setup_object,setup_flat,setup_arc,dictionary, setup, force=False, verbose=False):
    _dir = '_'.join(setup)
    slits = {}
    ############ split files
    for img in setup_object[setup] + setup_flat[setup]+setup_arc[setup]:
        for key in [3,7]:
            print(img,dictionary[img]['OBJECT'],key)
            dotrim = True
            if 'trimmed'+str(key) in dictionary[img] and force==False:
                image = dictionary[img]['trimmed' + str(key)][0].data
                image_plot(image,frame=3,_title=dictionary[img]['OBJECT'])
                
                if pyversion>=3:
                    answ = input('do you want to trim again (see figure 3) ? [y/n] [n] ')
                else:
                    answ = raw_input('do you want to trim again (see figure 3) ? [y/n] [n]')
                if not answ: answ = 'n'
                if answ in ['n','N','NO','no']:
                    dotrim = False
            if dotrim:
                if key not in slits:
                    if len(setup_flat[setup]):
                        imgslit = setup_flat[setup][0]
                        print('use flats to identify the slit position')
                    elif len(setup_arc[setup]):
                        imgslit = setup_arc[setup][0]
                        print('use arc to identify the slit position')
                    elif len(setup_object[setup]):
                        print('use object to identify the slit position')
                        imgslit = setup_object[setup][0]
                    
                    slits[key] = findslit(imgslit,key,verbose=True,cut=0.2)

                imgtrimmed = re.sub('.fits','',img) + '_' + str(key) + '_trimmed.fits'
                #
                # slit of one arcsec is the third one in the multislit
                #
                xmin,xmax = slits[key][2]
                print(xmin,xmax)
                    
                _header = dictionary[img]['fits'][key].header
                _data = np.transpose(dictionary[img]['fits'][key].data)
                
                for ll in ['DATASEC','DETSIZE','DETSEC']:
                    del _header[ll]
                science = CCDData(data=_data,header=_header,unit=u.adu)
          
                # add header from the 
                _header['exptime'] = dictionary[img]['EXPTIME']
                _header['MJD-OBS'] = dictionary[img]['MJD-OBS']
                _header['OBJECT']  = dictionary[img]['OBJECT']
                _header['OBSTYPE'] = dictionary[img]['OBSTYPE']
                _header['AIRMASS'] = dictionary[img]['AIRMASS']
                _header['RA']      = dictionary[img]['RA']
                _header['DEC']     = dictionary[img]['DEC']
             
                #  trim images 
                trimmed = ccdproc.trim_image(science, fits_section='[:,' + str(xmin) + ':' + str(xmax) + ']')
                dictionary[img]['trimmed'+str(key)] = trimmed.to_hdu()

                if not os.path.isdir(_dir):
                    os.mkdir(_dir)
                if not os.path.isdir(_dir + '/' + str(key)):
                    os.mkdir(_dir + '/' + str(key))

                hdu = dictionary[img]['trimmed' + str(key)]
                _out = fits.ImageHDU(data=hdu[0].data, header=hdu[0].header)
                fits.writeto(_dir + '/' + str(key)  + '/' + imgtrimmed, _out.data,header=_out.header,overwrite='yes')
#            else:
#                if verbose:
#                    print('read trimmed image from file')
#                hdu = fits.open(_dir + '/' + str(key)  + '/' + imgtrimmed)
#                dictionary[img]['trimmed' + str(key)] = hdu[0].data
    return dictionary

######################################################################

def makeflat(setup_flat,dictionary,setup,key,verbose=False):
    ######### make master flat 3 
    flatlist = []
    for img in setup_flat[setup]:
        if 'trimmed'+str(key) in dictionary[img]:
            flatlist.append(dictionary[img]['trimmed'+str(key)][0].data)
    if len(flatlist):
       stack, masterflat = flatcombine2(flatlist, verbose = verbose, response = True, Saxis=0)
       _dir = '_'.join(setup)
       masterflatname = 'masterflat_' +  _dir  + '_' + str(key) + '.fits'
       #_out = fits.ImageHDU(data=nosky)
       fits.writeto(_dir + '/' + str(key) + '/' + masterflatname, masterflat, overwrite='yes')       
    else:
        masterflat = None

        
    return masterflat

###############################################################

def image_plot(image0,frame=3,_title='',xlabel='Column Number',ylabel='Row Number'):
    fig = plt.figure(frame)
    fig.clf()
    
    if len(image0)==2:
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax.title.set_text(_title)
        
        # plot 1        
        image=image0[0]
        # determine the image pixel distribution (used for displaying below)
        sample = sigma_clip(image)
        vmin = sample.mean() - 1 * sample.std()
        vmax = sample.mean() + 3 * sample.std()
        yvals, xvals = np.indices(image.shape)
        extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
        ax.imshow(image, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)
        ax.axes.set_xlabel(xlabel)
        ax.axes.set_ylabel(ylabel)
        
        # plot 2
        image1=image0[1]
        # determine the image pixel distribution (used for displaying below)
        sample = sigma_clip(image1)
        vmin2 = sample.mean() - 1 * sample.std()
        vmax2 = sample.mean() + 3 * sample.std()
        yvals, xvals = np.indices(image1.shape)
        extent2 = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
        ax2.imshow(image1, origin='lower', cmap='gray', aspect='auto', vmin=vmin2, vmax=vmax2, extent=extent2)
#        ax2.axes.set_xlabel(xlabel)
#        ax2.axes.set_ylabel(ylabel)
        
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.title.set_text(_title)
        image = image0        
        # determine the image pixel distribution (used for displaying below)
        sample = sigma_clip(image)
        vmin = sample.mean() - 1 * sample.std()
        vmax = sample.mean() + 3 * sample.std()
        yvals, xvals = np.indices(image.shape)
        extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
        ax.imshow(image, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)





    
#######################################################

def find_sky_object(image,objectcut=0.2,key=3,interactive=False):
    # get the a pixel coordinate near the image center
    ny, nx = image.shape
    cy, cx = ny//2, nx//2
    # create 1d arays of the possible x and y values
    xs = np.arange(nx)
    ys = np.arange(ny)
    # pixel coordinates for each pixel
    yvals, xvals = np.indices(image.shape)

    # compute the row averages and normalize so that the background is near 0 and the peaks are near 1
    rowaverage = image.mean(axis=1)
#    rowaverage -= np.median(rowaverage)
    rowaverage -= np.percentile(rowaverage,0.1)
    rowaverage /= rowaverage.max()
    
    image_plot(image)

    plt.figure(2)
    plt.clf()
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(ys, rowaverage)
    ax2.set_xlabel('Row Number (y-coordinate)'), plt.ylabel('Normalized Row Average')
    ax2.grid();

    # find the rows with object light
    #objrows = ys[rowaverage > objectcut]

    rangeobj = {3:'70,100', 7:'55,90'}
    if interactive:
        if pyversion>=3:
            obj = input('give object position [eg ' + str(rangeobj[key]) + ']')
        else:
            obj = raw_input('give object position [eg ' + str(rangeobj[key]) + ']')
            
        if not obj:
            obj = rangeobj[key]
    else:
        obj = rangeobj[key]
        
    start,end= obj.split(',')
    objrows = np.arange(int(start),int(end))
            
    # mask to mark sky rows
    skymask = np.ones(image.shape, dtype=bool)
    objrows = objrows[objrows<ny]
    skymask[objrows, :] = False
    
    # also exclude bad rows
    badrows = ys[rowaverage < -0.05]
    skymask[badrows, :] = False
    
#    # rows with mostly sky background light
#    skyrows = ys[skymask.mean(axis=1) == 1]
#
#    # median (unmasked) sky spectrum and standard deviation
#    medspec = np.median(image[skyrows, :], axis=0)
#    stdspec = np.std(image[skyrows, :], axis=0, ddof=1)
#    
#    # exclude deviant pixels from the skymask
#    pull = (image - medspec) / stdspec
#    w = pull > 5
#    skymask[w] = False

    if interactive:
        plt.figure(3)
        plt.clf()
        plt.imshow(skymask, origin='lower', aspect='auto');
        
    return objrows, skymask

#############################################3

###################################################

def retify_frame(img0, dictionary, ext=3, verbose=False):            
    image = dictionary[img0]['trimmed'+str(ext)][0].data                
    # this is an arc, we do not need to mask
    skymask = np.ones(image.shape, dtype=bool)
    
#    # show the mask
#    if verbose:
#        plt.figure(3)
#        plt.clf()
#        plt.imshow(skymask, origin='lower', aspect='auto');
#        if pyversion>=3:
#            input('stop')
#        else:
#            raw_input('stop')
            
    # cut out a small image "stamp" near the center of the frame
    ny, nx = image.shape
    cy, cx = ny//2, nx//2
    xs = np.arange(nx)
    ys = np.arange(ny)
    # pixel coordinates for each pixel
    yvals, xvals = np.indices(image.shape)
    
    row = cy
    col = cx
    hwidth = 50
    
    stamp = image[:, col - hwidth : col + hwidth]
    yvals, xvals = np.indices(image.shape)
    ys_stamp = yvals[:, col - hwidth : col + hwidth]
    xs_stamp = xvals[:, col - hwidth : col + hwidth]

    if verbose:
        image_plot(stamp)
        # plot stamp values against column numbers
        plt.figure(3)
        plt.clf()
        plt.plot(xs_stamp.ravel(), stamp.ravel(), 'r.');
        plt.xlabel('Column Number'), plt.ylabel('Counts');
        if pyversion>=3:
            input('stop')
        else:
            raw_input('stop')
            
    # pixel offsets from the refernece pixel
    dxs = xs_stamp - col
    dys = ys_stamp - row
    
    # parameter guess
    guess = (1e-5, 1e-5)

    # get the wavelength offsets and plot vs. counts
    dls = get_dl_model(guess, dxs, dys)
    plt.figure(3)
    plt.clf()
    plt.plot(dls.ravel(), stamp.ravel(), 'r.')
    plt.xlabel('Wavelength Offset')
    plt.ylabel('Counts');
    if pyversion>=3:
        input('stop')
    else:
        raw_input('stop')
        
    # fit a spline to the data and plot
    x, y, spl = get_profile_spl(dls, stamp)

    plt.figure(3)
    plt.clf()
    fig3= plt.figure(3)
    ax1 = fig3.add_subplot(2, 1, 2)
#    fig, axarr = plt.subplots(2, sharex=True)
    ax1.plot(x, y, 'r.')
    ax1.plot(x, spl(x))
    ax1.plot(x, y - spl(x), 'r.')
    ax1.set_ylim(-200, 200)
    ax1.set_xlabel('Wavelength Offset');
    if pyversion>=3:
        input('stop')
    else:
        raw_input('stop')
        
    # see how good our guess is
    check_dl_model(guess, dxs, dys, stamp)

    # get the best model parameters for this stamp
    params = fmin(check_dl_model, guess, args=(dxs, dys, stamp))
    print("best model parameters are", params)

    hwidth = 300 # width = 2 * hwidth + 1
    cols = np.arange(hwidth, nx, 2 * hwidth)
    cols = cols[1:]

    # reference wavelength offsets to the center row
    row = cy

    # define a 2D array to hold the wavelength offsets for each pixel
    lambdas = np.zeros(image.shape) 

    # loop over each central column
    for col in cols:
        print('col = ', col)
    
        # slice the data
        inds = np.s_[:, col - hwidth : col + hwidth]
        stamp = image[inds]
        mask = skymask[inds]
        dys = yvals[inds] - row
        dxs = xvals[inds] - col
        
        # initial fit
        params = fmin(check_dl_model, guess, args=(dxs[mask], dys[mask], stamp[mask]))
    
        # check for outliers
        dls = get_dl_model(guess, dxs, dys)
        x, y, spl = get_profile_spl(dls, stamp)
        model = spl(dls)
        pull = (stamp - model) / np.sqrt(np.abs(model))
        w = (pull < 5) & mask
        params2 = fmin(check_dl_model, params, args=(dxs[w], dys[w], stamp[w]))
        
        # record
        lambdas[inds] = get_dl_model(params2, dxs, dys) + col
        pickle.dump(lambdas, open('lambdas_' + str(pyversion) + '_' + str(ext) + '.dat', 'wb'))

    plotcurvature(ny,cols,lambdas,xs,order=2)
    ###############
    # just plot offsets for a few of the rows across the image
#    plt.clf()
#    order = 2
#    for y in range(10, ny, 40):
#            p = plt.plot(cols, lambdas[y, cols] - xs[cols], 'o')
#            c = np.polyfit(cols, lambdas[y, cols] - xs[cols], order)
#            plt.plot(xs, np.polyval(c, xs), c=p[0].get_color(), label='row {}'.format(y))
#    plt.legend()
#    plt.xlabel('Column Number')
#    plt.ylabel('Wavelength Offset from Middle Row');

    if pyversion>=3:        
        input('\n### Solution of wavelength offset shown in Figure 3 ')
    else:
        raw_input('\n### Solution of wavelength offset shown in Figure 3 ')
    return lambdas

################################################3
def plotcurvature(ny,cols,lambdas,xs,order=2):
    global ax2, _ny, _cols, _lambdas,_xs, _order, lines
    _ny = ny
    _lambdas = lambdas
    _xs = xs
    _order = order
    _cols = cols
    
    fig = plt.figure(1)
    plt.clf()
    ax2 = fig.add_subplot(1, 1, 1)
    
    # just plot offsets for a few of the rows across the image
    for y in range(10, _ny, 40):
            p = plt.plot(_cols, _lambdas[y, _cols] - _xs[_cols], 'o')
            c = np.polyfit(_cols, _lambdas[y, _cols] - _xs[_cols], _order)
            lines = ax2.plot(_xs, np.polyval(c, _xs), c=p[0].get_color(), label='row {}'.format(y))

    plt.legend()
    plt.xlabel('Column Number')
    plt.ylabel('Wavelength Offset from Middle Row');
    plt.ylim(-3,3)
    
    print('\n#####################3\n [a]dd  point, [d]elete point, 1,2,3,[4],5,6 (poly order) \n')    
#    kid = fig.canvas.mpl_connect('key_press_event', onkeypress2)
##    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    plt.draw()
#    if pyversion>=3:
#        input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
#    else:
#        raw_input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
#    return

#######################################################
def  onkeypress2(event):
    global ax2, _ny, _cols, _lambdas,_xs, _order, lines

    xdata,ydata = event.xdata,event.ydata
    print(xdata,ydata)
    print(_cols)
    print(_lambdas)
#    dist = np.sqrt((xdata-np.array(_obj_wave_ds))**2+(ydata-np.array(_LogSensfunc))**2)
#    ii = np.argmin(dist)

    if event.key == 'a' :
        print('a')
#        idd.append(idd[-1]+1)
#        __obj_wave_ds = list(_obj_wave_ds)
#        __obj_wave_ds.append(xdata)
#        _obj_wave_ds = np.array(__obj_wave_ds)
#        __LogSensfunc = list(_LogSensfunc)
#        __LogSensfunc.append(ydata)
#        _LogSensfunc = np.array(__LogSensfunc)
#        ax2.plot(xdata,ydata,'db',ms=10)

    if event.key == 'd' :
        print('d')
#        idd.remove(ii)
#        for i in range(len(_obj_wave_ds)):
#            if i not in idd: nonincl.append(i)

    if event.key in ['1','2','3','4','5','6','7','8','9'] :
        _order=int(event.key)
        print(_order)
    
    # just plot offsets for a few of the rows across the image
    for y in range(10, _ny, 40):
            p = plt.plot(_cols, _lambdas[y, _cols] - _xs[_cols], 'o')
            c = np.polyfit(_cols, _lambdas[y, _cols] - _xs[_cols], _order)
            lines.pop(0).remove()
            lines = ax2.plot(_xs, np.polyval(c, _xs), c=p[0].get_color(), label='row {}'.format(y))

    plt.legend()
    plt.ylim(-3,3)
    plt.xlabel('Column Number')
    plt.ylabel('Wavelength Offset from Middle Row');    
    print('just testing')

#######################################################

def model_sky(params, dx, counts):
#    wav0, a, b, c, d, scale, c0 = params
    wav0, a, b, scale, c0 = params
    
    dtype = []
    dtype.append(('wav', float))
    dtype.append(('flux', float))
    model = np.zeros(counts.shape, dtype=dtype)
#    model['wav'] = wav0 + a * dx + b * dx**2 + c * dx**3 +  d * dx**4
    model['wav'] = wav0 + a * dx + b * dx**2
    model['flux'] = c0 + scale * counts
    
    return model

#######################################################

def get_sky_difference(params, dx, specdata, skyatlas):
    model = model_sky(params, dx, specdata)
    
    # residual
    res = model['flux'] - skyref_interp(model['wav'])
    
    return np.sum(res**2 / np.sqrt(np.abs(model['flux'])))

#######################################################

def plot_sky_model(skyref, model):
    plt.clf()
    plt.plot(model['wav'], model['flux'], label='Extracted Sky Background');
    plt.plot(skyref['wav'], skyref['flux'], label='Reference Sky Spectrum');
    plt.xlim(model['wav'].min(), model['wav'].max());
    plt.ylim(model['flux'].min() - 0.25, model['flux'].max() + 0.25)
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Scaled Counts or Flux')
    plt.legend();

###########################################

def trace(img,dictionary, g0=10, g1=85, g2=3, key=3, verbose=False):
    if 'sky' + str(key) in dictionary[img]:
        sky = dictionary[img]['sky' + str(key)][0].data
        image = dictionary[img]['trimmed' + str(key)][0].data
        nosky = image - sky
         
        # get the a pixel coordinate near the image center
        ny, nx = image.shape
        cy, cx = ny//2, nx//2
        # create 1d arays of the possible x and y values
        xs = np.arange(nx)
        ys = np.arange(ny)
        # pixel coordinates for each pixel
        yvals, xvals = np.indices(image.shape)
         
        # get the median for each row
        profile = np.median(image - sky, axis=1)

        # starting guess for the profile model
        guess = (g0, g1, g2)
        model = get_profile_model(guess, ys)

        # fit for the best model
        params = fmin(get_profile_chisq, guess, args=(ys, profile))
        print("best fit parameters are", params)

        model = get_profile_model(params, ys)

        # fit the profile centered at these columns
        hwidth = 50
        cols = np.arange(hwidth, nx + 1, 2 * hwidth)

        ycenter = np.zeros( (len(cols), 2) )
        for icol, col in enumerate(cols):
             stamp = (image - sky)[:, col - hwidth : col + hwidth]
             profile = np.mean(stamp, axis=1)
             params = fmin(get_profile_chisq, guess, args=(ys, profile))
             ycenter[icol, :] = params[[1]]

        print(ycenter)
        # fit the relation with a polynomial
        ind = 0 # which trace 0 or 1?
        t_order = 3
        trace_c = np.polyfit(cols, ycenter[:, ind], t_order)
        print(trace_c)
        slitpos = yvals - np.polyval(trace_c, yvals)
        peakpos = np.polyval(trace_c, xs)

        dictionary[img]['peakpos' + str(key)] = peakpos
        dictionary[img]['trace' + str(key)] = slitpos
        dictionary[img]['tracefit' + str(key)] = trace_c
        if verbose:
            plt.figure(2)
            fig2 = plt.figure(2)
            fig2.clf()
            ax2 = fig2.add_subplot(2, 1, 1)
            ax22 = fig2.add_subplot(2, 1, 2)
#            fig, axarr = plt.subplots(2, sharex=True)
            ax2.plot(cols, ycenter[:, ind], 'ro')
            ax2.plot(xs, np.polyval(trace_c, xs), 'r')
            ax2.axes.set_ylabel('y-coordinate')
            ax22.plot(cols, ycenter[:, ind] - np.polyval(trace_c, cols), 'ro')
            ax22.axes.set_ylim(-0.5, 0.5)
            ax22.axes.set_ylabel('Fit Residual (pixels)')
            ax2.set_xlabel('Column Number');
            if pyversion>=3:        
                input('trace completed')
            else:
                raw_input('trace completed')
    return dictionary

##################################################


def extract(img,dictionary, key=3, edgeleft=30, edgeright=30, othertrace=None, shift=0, verbose=False):
    extract = False
    _grism = dictionary[img]['GRATENAM']
    _slit = dictionary[img]['SLMSKNAM']   
    setup = (_grism,_slit)
    _dir = '_'.join(setup)
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
    if not os.path.isdir(_dir + '/' + str(key)):
        os.mkdir(_dir + '/' + str(key))
    
    if verbose:
        print('dimension ',key)
        print(img)
        
    if othertrace:
        if 'trace' + str(key) in dictionary[othertrace]:
            slitpos = dictionary[othertrace]['trace' + str(key)]
            trace_c = dictionary[othertrace]['tracefit' + str(key)]
            peak = dictionary[othertrace]['peakpos' + str(key)]
            extract = True
        else:
            print('Warning: ',othertrace,' do not have a trace')

    if 'trace' + str(key) in dictionary[img]:
        slitpos = dictionary[img]['trace' + str(key)]
        trace_c = dictionary[img]['tracefit' + str(key)]
        peak = dictionary[img]['peakpos' + str(key)]
        extract = True
        
    if extract is False:
        print('\n###  Error: trace not found')
        return dictionary
    else:
        sky = dictionary[img]['sky' + str(key)]
        image = dictionary[img]['trimmed' + str(key) ]
        nosky = image - sky
        
        # get the a pixel coordinate near the image center
        ny, nx = image.shape
        cy, cx = ny//2, nx//2
        # create 1d arays of the possible x and y values
        xs = np.arange(nx)
        ys = np.arange(ny)
        
        # pixel coordinates for each pixel
        yvals, xvals = np.indices(image.shape)
                  
        # normalize to the pixel brightness at the trace center
        yinds = (np.round(np.polyval(trace_c, xs))).astype(int)
        normed = nosky / nosky[yinds, xs]

        # get 1D arrays with the positions along the slit and the normalized counts
        pos = slitpos.flatten()
        counts = normed.flatten()
         
        # sort by slit position
        sort_inds = pos.argsort()
        pos, counts = pos[sort_inds], counts[sort_inds]

        print(pos)
        # fit a spline to model the spatial profile
        t = np.linspace(pos.min() + 2, pos.max() - 2, ny // 2) # spline knot points
        profile_spl = LSQUnivariateSpline(pos, counts, t)

        # remove outliers and re-fit
        diff = counts - profile_spl(pos)
        sample = sigma_clip(diff)
        w = ((np.abs(diff) / sample.std()) < 3) & np.isfinite(diff)
        profile_spl = LSQUnivariateSpline(pos[w], counts[w], t)
        
        # create the profile image
        profile_image = profile_spl(slitpos)

        #profile image
        _out = fits.ImageHDU(data=profile_image)
        fits.writeto(_dir + '/' + str(key)  + '/' + re.sub('.fits','',img) + '_profile_' +\
                     str(key) + '.fits', _out.data,header=_out.header,overwrite='yes')

                    
        # de-weight negative values in provile_image
        profile_image[profile_image < 0] = 0
        
        # select which rows to sum
        w = (slitpos > (-1) * edgeleft) & (slitpos < edgeright )
        ymin, ymax = yvals[w].min(), yvals[w].max()
        print(ymin, ymax)

        ###########################################################
        ######  replacing with my simple basic extraction
        ######
        zero = nosky-nosky
        for r in range(nx):
            zero[int(peak[r])-edgeleft:int(peak[r])+edgeright,r]=1

        sumsource  = nosky * zero
        sumsky = sky * zero
        spec_basic = sumsource.sum(axis=0)
        skybg_basic = sumsky.sum(axis=0)


        
        # calculate the weighted average (for each column)
        spec_opt = (nosky * profile_image)[ymin:ymax, :].sum(axis=0) / profile_image.sum(axis=0)
        
        # calculate the bias factor needed to scale the average to a sum
        bias_factor = np.median(spec_basic / spec_opt)
        spec_opt *= bias_factor

        if verbose:
            plt.figure(2)
            plt.clf()
            fig2 = plt.figure(2)
            ax1 = fig2.add_subplot(2, 1, 1)
            mean, median, std = sigma_clipped_stats(nosky)
            ax1.imshow(nosky, vmin = median - 2*std, vmax = median + 2*std)
            ax2 = fig2.add_subplot(2, 1, 2)
            mean, median, std = sigma_clipped_stats(profile_image)
            ax2.imshow(profile_image, vmin = median - 2*std, vmax = median + 2*std)
            if pyversion>=3:
                input('optimal extraction shown on figure 2')
            else:
                raw_input('optimal extraction shown on figure 2')
                
        # same for the sky background
        skybg_opt = (sky * profile_image)[ymin:ymax, :].sum(axis=0) / profile_image.sum(axis=0)
        bias_factor_sky = np.median(skybg_basic / skybg_opt)
        skybg_opt *= bias_factor_sky
        skybg_opt -= np.min(skybg_opt)
        dictionary[img]['spec_basic' + str(key)]= spec_basic
        dictionary[img]['spec_opt' + str(key)]= spec_opt
        dictionary[img]['skybg_opt' + str(key)]= skybg_opt

        if verbose:
            # plot the extracted spectrum
            plt.clf()
            plt.plot(xs, spec_basic, label='basic extraction')
            plt.plot(xs, spec_opt, label='optimal extraction')
            plt.legend()
            if pyversion>=3:
                input('extraction completed')
            else:
                raw_input('extraction completed')                
    return dictionary
                    

##################################################    

def fit_another(xs, cols, hwidths, fitparams, i, plot=True):
    col = cols[i]
    hwidth = hwidths[i]
    
    # find the closest matching index with a valid fit
    inds = np.arange(cols.size)
    w = np.isfinite(fitparams[:, 0]) & (inds != i)
    iref = inds[w][np.argmin(np.abs(inds[w] - i))]
    print("reference index is ", iref)

    # get the wavelength guess
    if w.sum() == 0:
        print("not enough data to make guess...bye!")
        return
    if w.sum() < 4:
        guess = fitparams[iref, :].copy()
        guess[0] = guess[0] + guess[1] * (cols[i] - cols[iref])
    else:
        if w.sum() > 9:
            order = 3
        elif w.sum() > 6:
            order = 2
        else:
            order = 1
        print("order is", order)
        cwav = np.polyfit(cols[w], fitparams[w, 0], order)
        cdwav = np.polyfit(cols[w], fitparams[w, 1], order)
        cscale = np.polyfit(cols[w], fitparams[w, 3], order)
        guess = (np.polyval(cwav, col), np.polyval(cdwav, col), 0.0, np.polyval(cscale, col), 0.0)
    print("guess is", guess)

    w = (xs > (cols[i] - hwidths[i])) & (xs < (cols[i] + hwidths[i]))
    output = fmin(get_sky_difference, guess, args=(xs[w] - cols[i], bgspec[w], skyref)
                  , full_output=True, disp=False, maxiter=10000)
    if output[4] == 0:
        fitparams[i, :] = output[0]
        print(output[0])

        if plot:
            model = model_sky(output[0], xs[w] - col, bgspec[w])
            plot_sky_model(skyref, model)
            plt.title('Fit to Section {}'.format(i))
    else:
        print("fit failed!")

########################################################
###########################################################################
def readstandard(standardfile):
    import deimos
    import numpy as np
    import string, os

    if os.path.isfile(standardfile):
        listastandard = standardfile
    elif standardfile[0] == '/':
        listastandard = standardfile
    else:
        listastandard = _path[0] + '/standard/' + standardfile
    f = open(listastandard, 'r')
    liststd = f.readlines()
    f.close()
    star, ra, dec = [], [], []
    magnitude = []
    for i in liststd:
        if i[0] != '#':
            star.append(i.split()[0])
            _ra = (i.split()[1]).split(':')
            _dec = (i.split()[2]).split(':')
            ra.append((float(_ra[0]) + ((float(_ra[1]) + (float(_ra[2]) / 60.)) / 60.)) * 15)
            if '-' in str(_dec[0]):
                dec.append((-1) * (np.abs(float(_dec[0])) + ((float(_dec[1]) + (float(_dec[2]) / 60.)) / 60.)))
            else:
                dec.append(float(_dec[0]) + ((float(_dec[1]) + (float(_dec[2]) / 60.)) / 60.))
            try:
                magnitude.append(str.split(i)[3])
            except:
                magnitude.append(999)
    return np.array(star), np.array(ra), np.array(dec), np.array(magnitude)

###########################################################################

def _mag2flux(wave, mag, zeropt=48.60):
    '''
    Convert magnitudes to flux units. This is important for dealing with standards
    and files from IRAF, which are stored in AB mag units. To be clear, this converts
    to "PHOTFLAM" units in IRAF-speak. Assumes the common flux zeropoint used in IRAF

    Parameters
    ----------
    wave : 1d numpy array
        The wavelength of the data points
    mag : 1d numpy array
        The magnitudes of the data
    zeropt : float, optional
        Conversion factor for mag->flux. (Default is 48.60)

    Returns
    -------
    Flux values!
    '''

    c = 2.99792458e18 # speed of light, in A/s
    flux = 10.0**( (mag + zeropt) / (-2.5) )
    return flux * (c / wave**2.0)

###########################################################################

def DefFluxCal(obj_wave, obj_flux, stdstar='', mode='spline', polydeg=4, exptime =1, airmass=1,
               display=False, interactive=False):
    """

    Parameters
    ----------
    obj_wave : 1-d array
        The 1-d wavelength array of the spectrum

    obj_flux : 1-d array
        The 1-d flux array of the spectrum

    stdstar : str
        Name of the standard star file to use for flux calibration. You
        must give the subdirectory and file name, for example:

        >>> sensfunc = DefFluxCal(wave, flux, mode='spline', stdstar='spec50cal/feige34.dat')  # doctest: +SKIP

        If no standard is set, or an invalid standard is selected, will
        return array of 1's and a warning. A list of all available
        subdirectories and objects is available on the wiki, or look in
        pydis/resources/onedstds/

    mode : str, optional
        either "linear", "spline", or "poly" (Default is spline)

    polydeg : float, optional
        set the order of the polynomial to fit through (Default is 9)

    display : bool, optional
        If True, plot the down-sampled sensfunc and fit to screen (Default
        is False)

    Returns
    -------
    sensfunc : 1-d array
        The sensitivity function for the standard star

    """
    stdstar2 = stdstar.lower()
    std_dir = os.path.join(os.path.dirname(_path[0]),
                           'deimos','resources', 'onedstds')

    if os.path.isfile(os.path.join(std_dir, stdstar2)):
        std_wave, std_mag, std_wth = np.genfromtxt(os.path.join(std_dir, stdstar2),
                                                   skip_header=1, unpack=True)
        # standard star spectrum is stored in magnitude units
        std_flux = _mag2flux(std_wave, std_mag)

        std_flux = atmoexp_correction(std_wave, std_flux, exptime, airmass, site='mauna', verbose = True)
        
        # Automatically exclude these obnoxious lines...
        balmer = np.array([6563, 4861, 4341], dtype='float')

        # down-sample (ds) the observed flux to the standard's bins
        obj_flux_ds = []
        obj_wave_ds = []
        std_flux_ds = []
        for i in range(len(std_wave)):
            rng = np.where((obj_wave >= std_wave[i] - std_wth[i] / 2.0) &
                           (obj_wave < std_wave[i] + std_wth[i] / 2.0))
            IsH = np.where((balmer >= std_wave[i] - std_wth[i] / 2.0) &
                           (balmer < std_wave[i] + std_wth[i] / 2.0))

            # does this bin contain observed spectra, and no Balmer line?
            if (len(rng[0]) > 1) and (len(IsH[0]) == 0):
                # obj_flux_ds.append(np.sum(obj_flux[rng]) / std_wth[i])
                obj_flux_ds.append( np.nanmean(obj_flux[rng]) )
                obj_wave_ds.append(std_wave[i])
                std_flux_ds.append(std_flux[i])


        # the ratio between the standard star flux and observed flux
        # has units like erg / counts
        ratio = np.abs(np.array(std_flux_ds, dtype='float') /
                       np.array(obj_flux_ds, dtype='float'))


        # interp calibration (sensfunc) on to object's wave grid
        # can use 3 types of interpolations: linear, cubic spline, polynomial

        # if invalid mode selected, make it spline
        if mode not in ('linear', 'spline', 'poly'):
            mode = 'spline'
            print("WARNING: invalid mode set in DefFluxCal. Changing to spline")
            
        # actually fit the log of this sensfunc ratio
        # since IRAF does the 2.5*log(ratio), everything in mag units!
        LogSensfunc = np.log10(ratio)

        if interactive==True:
            again=True
            while again ==True:
                if pyversion>=3:
                    mode = input('which mode [linear/spline/poly] [spline]? ')
                else:
                    mode= raw_input('which mode [linear/spline/poly] [spline]? ')
                    
                sensfunc2 = fitsens(obj_wave, obj_wave_ds, LogSensfunc, mode, polydeg, obj_flux, std_wave, std_flux, obj_flux_ds)
                if pyversion>=3:
                    again = input('again [y/n/]? [y] ')
                else:
                    again = raw_input('again [y/n]? [y] ')
                if not again: again = True
                if again in ['Yes','y','yes','YES']:
                    again = True

        else:
            sensfunc2 = fitsens(obj_wave, obj_wave_ds, LogSensfunc, mode, polydeg, obj_flux,std_wave, std_flux, obj_flux_ds)
    else:
        sensfunc2 = np.zeros_like(obj_wave)
        print('ERROR: in DefFluxCal no valid standard star file found at ')
    print(os.path.join(std_dir, stdstar2))

    return 10**sensfunc2

####################################

def fitsens(obj_wave, obj_wave_ds, LogSensfunc, mode, polydeg0, obj_flux,std_wave, std_flux, obj_flux_ds):
    global idd, _obj_wave, _LogSensfunc, _polydeg0, _mode, lines, lines2,lines3, _obj_wave_ds, ax2, ax22, fig,\
        _obj_flux, _std_wave, _std_flux, nonincl, ax3, _obj_flux_ds, _sensfunc2
    fig = plt.figure(1)
    plt.clf()
    ax2 = fig.add_subplot(3, 1, 1)
    ax22 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    _obj_wave = obj_wave
    _obj_flux = obj_flux
    _std_wave = std_wave 
    _std_flux = std_flux
    _obj_wave_ds  = obj_wave_ds
    _obj_flux_ds  = obj_flux_ds
    _LogSensfunc = LogSensfunc
    idd = list(range(len(obj_wave_ds)))
    _mode = mode
    _polydeg0 = polydeg0
    nonincl = []

    
    if not mode:
        mode='spline'
        
    if mode=='linear':
        sensfunc2 = np.interp(obj_wave, obj_wave_ds, LogSensfunc)
    elif mode=='spline':
        spl = UnivariateSpline(obj_wave_ds, LogSensfunc, ext=0, k=2 ,s=0.0025)
        sensfunc2 = spl(obj_wave)
    elif mode=='poly':
        if pyversion>=3:
            polydeg0 = input('polydeg ? [4] ' )
        else:
            polydeg0 = raw_input('polydeg ? [4] ')
        if not polydeg0: polydeg0 = 4
        
        fit = np.polyfit(obj_wave_ds, LogSensfunc, float(polydeg0))
        sensfunc2 = np.polyval(fit, obj_wave)

    _sensfunc2 = sensfunc2
    _mode = mode
    
    x0 = np.min(_obj_wave)
    x1 = np.max(_obj_wave)
    
    ax2.plot(_obj_wave_ds, _LogSensfunc, 'ko', label='sensfunc')
    lines =    ax2.plot(_obj_wave, _sensfunc2, '-b', label='interpolated sensfunc')
    ax2.set_xlabel('Wavelength')
    ax2.set_ylabel('log Sensfunc')

    lines2 = ax22.plot(obj_wave, obj_flux*(10**_sensfunc2),'k',
             label='applied sensfunc')
    ax22.plot(std_wave, std_flux, 'ro', alpha=0.5, label='standard flux')


    lines3 = ax3.plot(_obj_wave, _obj_flux*(10**_sensfunc2),'k',
             label='applied sensfunc')

    ax3.plot(_obj_wave_ds, _obj_flux_ds*(10**LogSensfunc), 'bo', label='downsample observed')

    ax2.set_xlim([x0, x1])
    ax22.set_xlim([x0, x1])
    ax3.set_xlim([x0, x1])
    
    plt.legend()
    plt.draw()

    print('\n#####################3\n [a]dd  point, [d]elete point, 1,2,3,[4],5,6 (poly order) \n')    
    
    kid = fig.canvas.mpl_connect('key_press_event', onkeypress)
#    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    plt.draw()
    if pyversion>=3:
        input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
    else:
        raw_input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
    return _sensfunc2

####################################
def onkeypress(event):
    global idd, _obj_wave, _LogSensfunc, _polydeg0, _mode, lines, lines2, lines3, _obj_wave_ds, ax2, ax22, fig,\
        _obj_flux, _std_wave, _std_flux, nonincl, ax3, _obj_flux_ds, _sensfunc2


    xdata,ydata = event.xdata,event.ydata
    dist = np.sqrt((xdata-np.array(_obj_wave_ds))**2+(ydata-np.array(_LogSensfunc))**2)
    ii = np.argmin(dist)

    if event.key == 'a' :
        idd.append(idd[-1]+1)
        __obj_wave_ds = list(_obj_wave_ds)
        __obj_wave_ds.append(xdata)
        _obj_wave_ds = np.array(__obj_wave_ds)
        __LogSensfunc = list(_LogSensfunc)
        __LogSensfunc.append(ydata)
        _LogSensfunc = np.array(__LogSensfunc)
        ax2.plot(xdata,ydata,'db',ms=10)

    if event.key == 'd' :
        idd.remove(ii)
        for i in range(len(_obj_wave_ds)):
            if i not in idd: nonincl.append(i)

    if event.key in ['1','2','3','4','5','6','7','8','9'] :
        _polydeg0=int(event.key)
        print(_polydeg0)

    print(_mode)
    if _mode=='linear':
        sensfunc2 = np.interp(_obj_wave, _obj_wave_ds[idd], _LogSensfunc[idd])
    elif _mode=='spline':
        spl = UnivariateSpline(np.array(_obj_wave_ds)[idd], np.array(_LogSensfunc)[idd], ext=0, k=2 ,s=0.0025)
        sensfunc2 = spl(_obj_wave)
    elif _mode=='poly':
        print('type the order of the polyfit in the figure')
        fit = np.polyfit(np.array(_obj_wave_ds)[idd], np.array(_LogSensfunc)[idd], float(_polydeg0))
        sensfunc2 = np.polyval(fit, _obj_wave)
        
    _sensfunc2 = sensfunc2 
    ax2.plot(_obj_wave_ds, _LogSensfunc,'ok')
    ax2.plot(np.array(_obj_wave_ds)[nonincl], np.array(_LogSensfunc)[nonincl],'ow')
    lines.pop(0).remove()
    lines2.pop(0).remove()
    lines3.pop(0).remove()
    
    lines =  ax2.plot(_obj_wave, _sensfunc2, '-b', label='interpolated sensfunc')

    lines2 = ax22.plot(_obj_wave, _obj_flux*(10**_sensfunc2),'k',
             label='applied sensfunc')
    ax22.plot(_std_wave, _std_flux, 'ro', alpha=0.5, label='standard flux')

    lines3 = ax3.plot(_obj_wave, _obj_flux*(10**_sensfunc2),'k',
             label='applied sensfunc')
#    ax3.plot(np.array(_obj_wave_ds)[idd], np.array(_obj_flux_ds)[idd]*(10**np.array(_LogSensfunc)[idd]), 'bo', label='downsample observed')
    ax3.plot(np.array(_obj_wave_ds)[nonincl], np.array(_obj_flux_ds)[nonincl]*(10**np.array(_LogSensfunc)[nonincl]), 'ro', label='')
    x0 = np.min(_obj_wave)
    x1 = np.max(_obj_wave)
    ax2.set_xlim([x0, x1])
    ax22.set_xlim([x0, x1])
    ax3.set_xlim([x0, x1])
    
    plt.draw()
    
###############################################################################


def checkwavelength_arc(xx1, yy1, xx2, yy2, xmin, xmax, inter=True):
    import numpy as np

    minimo = max(min(xx1), min(xx2)) + 50
    massimo = min(max(xx1), max(xx2)) - 50
    yy1 = [0 if e < 0 else e for e in np.array(yy1)]
    yy2 = [0 if e < 0 else e for e in np.array(yy2)]
    _shift, integral = [], []
    for shift in range(-500, 500, 1):
        xxnew = xx1 + shift / 10.
        yy2interp = np.interp(xxnew, xx2, yy2)
        yy2timesyy = yy2interp * yy1
        xxcut = np.compress((np.array(xxnew) >= minimo) & (
            np.array(xxnew) <= massimo), np.array(xxnew))
        yycut = np.compress((np.array(xxnew) >= minimo) & (
            np.array(xxnew) <= massimo), np.array(yy2timesyy))
        integrale = np.trapz(yycut, xxcut)
        integral.append(integrale)
        _shift.append(shift / 10.)
    result = _shift[integral.index(max(integral))]
    if inter:
        # import matplotlib as mpl
        #   mpl.use("TKAgg")
        plt.figure(3)
        plt.clf()
        ratio = np.trapz(yy1, xx1) / np.trapz(yy2, xx2)
        print(ratio)
        yy3 = np.array(yy2) * float(ratio)
        xx4 = xx1 + result
        plt.plot(xx1, yy1, label='spectrum')
        plt.plot(xx2, yy3, label='reference sky')
        plt.plot(xx4, yy1, label='shifted spectrum')
        plt.legend(numpoints=1, markerscale=1.5)
        if xmin != '' and xmax != '':
            plt.xlim(xmin, xmax)
    return result

##################################################################

def readtrace(ascifile):
    #
    #  input: filename
    #  output:
    #     astropy ascii table,
    #     dictionary  of parameters
    #
    #import string
    from astropy.io import ascii
    aa = ascii.read(ascifile)
    bb = [str.split(i,'=') for i in aa.meta['comments']]
    meta = {i[0]:i[1] for i in bb}
    # ascii use the first line as title, but I want to file to be easy to plot
    # also without reading it with ascii
    aa = np.genfromtxt(ascifile)
    return aa,meta

#################################################
def writetrace(trace,meta,tablename,output):
    #
    #  input:
    #        1d array
    #        meta = dictionary of parameters
    #        table name
    #        output filename
    #
    parameters = ''.join([' %s= %s\n' % (line,meta[line]) for line in meta])[:-1]
    cc = '%s\n %s ' %(tablename,parameters)
#    dd =[tablename] + ['# %s %s' % (line,meta[line]) for line in meta]+list(trace)
    np.savetxt(output,trace, header= parameters)
    
###################################################
def summary(dictionary):
    print('#'*20 + '\n')
    print('IMG          OBJECT   KEY   TRIM  SKY   TRACE  EXTRACTED  WAVE  FLUX  STD  RESPONSE')
    for img in dictionary:
        if dictionary[img]['type']=='object':
            for key in ['3','7']:
                tt    = 'trimmed' + str(key) in dictionary[img]
                ss    = 'nosky' + str(key) in dictionary[img]
                trace = 'peakpos_' + str(key) in dictionary[img]
                ex    = 'spec_opt'  +str(key) in dictionary[img]
                wav   = 'wave' + str(key) in dictionary[img]
                flux  = 'spec_flux' + str(key) in dictionary[img]
                response  = 'response' + str(key) in dictionary[img]
                std  = 'std' in dictionary[img]
                line = '%s  %s %s  %s  %s  %s %s  %s  %s  %s  %s' % (img,dictionary[img]['OBJECT'],key, tt,ss,trace,ex,wav,flux,std,response)
                print(line)
    print('#'*20 + '\n')

###################################################################################

def onkeypress1(event):
    global _xx,_yy, center, lower, upper, l1,l2,u1,u2, line1, line2, line3, line5, fig, gs, ax1, ax2,line6,\
        line7,line8, nx,linel1,linel2,lineu1,lineu2
    
    xdata,ydata = event.xdata,event.ydata
    
    if event.key == '6' :
        lower = xdata
    if event.key == '7' :
        upper = xdata
    if event.key == '1' :
        l1 = xdata
    if event.key == '2' :
        l2 = xdata
    if event.key == '3' :
        u1 = xdata
    if event.key == '4' :
        u2 = xdata
    if event.key == 'c' :
        center = xdata

    
    line1.pop(0).remove()
    line2.pop(0).remove()
    line3.pop(0).remove()
    line5.pop(0).remove()
    line6.pop(0).remove()
    line7.pop(0).remove()
    line8.pop(0).remove()
    linel1.pop(0).remove()
    linel2.pop(0).remove()
    lineu1.pop(0).remove()
    lineu2.pop(0).remove()
    

    line5 = ax1.plot(_yy,_xx,'-b')
    line1 = ax1.plot([l1,l2],[0,0],'-k')
    line2 = ax1.plot([u1,u2],[0,0],'-k')
    line3 = ax1.plot([lower,upper],[1,1],'-k')
    line6 = ax1.plot([center],[1],'or')
    line7 = ax2.plot([lower,lower],[0,nx],'--r')
    line8 = ax2.plot([upper,upper],[0,nx],'--r')
    linel1 =  ax2.plot([l1,l1],[0,nx],'--y')
    linel2 = ax2.plot([l2,l2],[0,nx],'--y')
    lineu1 = ax2.plot([u1,u1],[0,nx],'--y')
    lineu2 = ax2.plot([u2,u2],[0,nx],'--y')
    
def interactive_extraction(dictionary,img, key, listobjects, nsky0, interactive=True):
    global _dictionary, _img, _key, line1, line2, line3, line5, line6, gs, ax1, ax2,\
        line40,line41,line42,line43, line7, _nsky1, bottom,top
    
    import string
    _grism = dictionary[img]['GRATENAM']
    _slit = dictionary[img]['SLMSKNAM']   
    setup = (_grism,_slit)
    _dir = '_'.join(setup)
    # define global varaiable 
    _dictionary = dictionary
    _img = img
    _key = key
    _nsky1 = nsky0
    bottom = None
    top = None
    

    
    if 'trimflat' + str(_key) in _dictionary[_img]:                           
        image = _dictionary[_img]['trimflat' + str(_key) ]
    elif 'trimmed' + str(_key) in _dictionary[_img]:
        image = _dictionary[_img]['trimmed' + str(_key) ][0].data
                                    
    sky = _dictionary[_img]['sky' + str(_key)]
    nosky = image - sky
    ny, nx = nosky.shape
    xs = np.arange(nx)
    peak = _dictionary[_img]['peakpos_' + str(_key)]

    fig = plt.figure(3,figsize=(7,9))
    plt.clf()
    gs = gridspec.GridSpec(nrows=3, 
                       ncols=2, 
                       figure=fig, 
                       width_ratios= [1, 1],
                       height_ratios=[1, 1, 1],
                       wspace=0.3,
                       hspace=0.3)
    ny, nx = image.shape
    sample = sigma_clip(image)
    vmin = sample.mean() - 1 * sample.std()
    vmax = sample.mean() + 3 * sample.std()
    yvals, xvals = np.indices(image.shape)
    extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
    
    ax2 = fig.add_subplot(gs[0:2, 0:2])
    ax2.imshow(image, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)
    line1 = ax2.plot(xs,peak,'-r')
    line2 = ax2.plot(xs,peak + float(_dictionary[_img]['aplow_'+str(_key)]),'-w')
    line3 = ax2.plot(xs,peak + float(_dictionary[_img]['aphigh_'+str(_key)]),'-w')
    bkg = np.array(string.split(_dictionary[_img]['bckgrintervals_' + str(_key)],','),float)
    line40 = ax2.plot(xs,peak + bkg[0],'-y')
    line41 = ax2.plot(xs,peak + bkg[1],'-y')
    line42 = ax2.plot(xs,peak + bkg[2],'-y')
    line43 = ax2.plot(xs,peak + bkg[3],'-y')

    ax1 = fig.add_subplot(gs[2, 0:2])
    if 'spec_basic'+str(_key) in _dictionary[_img]:
        spec_basic = _dictionary[_img]['spec_basic' + str(_key)]
        spec_opt = _dictionary[_img]['spec_opt' + str(_key)]
        line5 = ax1.plot(xs, spec_opt, '-r', label='optimal extraction')
        line7 = ax1.plot(xs, spec_basic, '-b', label='')
        line6 = ax1.plot(xs, spec_basic, '-b', label='basic extraction')
        ax1.set_xlabel('pixels')
        ax1.set_ylabel('counts')
        ax1.legend()

    if interactive:
        othertrace = None
        _shift=0
        if pyversion>=3:
            answ = input('Do you want to use a different trace [1,2,3,4,5] [n] ?\n (see figure 2)\n[n] use object frame,\n1,2,3,4,5 (trace of different object ')
        else:
            answ = raw_input('Do you want to use a different trace [1,2,3,4,5] [n] ?\n (see figure 2)\n[n] use object frame,\n1,2,3,4,5 (trace of different object ')
                                         
        if not answ:
            answ='n'
        if answ in ['No','N','n','NO']:
            othertrace = None
        else:
            othertrace = listobjects[int(answ)]
            print(othertrace)
            
        if othertrace is not None:
            _dictionary[_img]['peakpos_' + str(_key)] = _dictionary[othertrace]['peakpos_' + str(_key)]
            peak = _dictionary[othertrace]['peakpos_' + str(_key)]
            line1.pop(0).remove()
            line1 = ax2.plot(xs,peak,'-g')
        
        
        kid = fig.canvas.mpl_connect('key_press_event',onkeypress5)
        plt.draw()
        if pyversion>=3:
            input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
        else:
            raw_input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
        
        
    # take the trace from the other file but the rest of parameters from the target
    meta ={
        'aplow':               _dictionary[_img]['aplow_' + str(_key)],
        'bckgrfunc':           _dictionary[_img]['bckgrfunc_' + str(_key)],
        'bckgr_low_reject':    _dictionary[_img]['bckgr_low_reject_' + str(_key)],
        'displine':            _dictionary[_img]['displine_' + str(_key)],
        'aphigh':              _dictionary[_img]['aphigh_' + str(_key)],
        'bckgrfunc_iraforder': _dictionary[_img]['bckgrfunc_iraforder_' + str(_key)],
        'coeffs':              _dictionary[_img]['coeffs_' + str(_key)],
        'bckgrintervals':      _dictionary[_img]['bckgrintervals_' + str(_key)],
        'bckgr_niterate':      _dictionary[_img]['bckgr_niterate_' + str(_key)],
        'bckgr_high_reject':   _dictionary[_img]['bckgr_high_reject_' + str(_key)],
    }
    
    trace = dictionary[_img]['peakpos_'+str(_key)]
    _dictionary[_img]['peakpos_'+str(_key)] = trace
    imgtrace = re.sub('.fits','',_img) + '_' + str(_key) + '_trace.ascii'
    output = _dir + '/' + str(_key)  + '/' + imgtrace                                
    # write the new trace for this object
    writetrace(trace,meta, 'trace', output)

    return _dictionary

def onkeypress5(event):
    global _dictionary, _img, _key, line1, line2, line3, line5, line6, gs, ax1, ax2,\
        line40,line41,line42,line43,line7, _nsky1, bottom,top
    import string
    
    xdata,ydata = event.xdata,event.ydata
    print(xdata,ydata)

    if 'trimflat' + str(_key) in _dictionary[_img]:                           
        image = _dictionary[_img]['trimflat' + str(_key) ]
    elif 'trimmed' + str(_key) in _dictionary[_img]:
        image = _dictionary[_img]['trimmed' + str(_key) ][0].data
                                    
    sky = _dictionary[_img]['sky' + str(_key)]
    nosky = image - sky
    ny, nx = nosky.shape
    xs = np.arange(nx)
    peak = _dictionary[_img]['peakpos_' + str(_key)]
    peakinterp = interp1d(xs,peak)            
    ny, nx = image.shape
    sample = sigma_clip(image)
    vmin = sample.mean() - 1 * sample.std()
    vmax = sample.mean() + 3 * sample.std()
    yvals, xvals = np.indices(image.shape)
    extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
    
    line1.pop(0).remove()
    line2.pop(0).remove()
    line3.pop(0).remove()
    line40.pop(0).remove()
    line41.pop(0).remove()
    line42.pop(0).remove()
    line43.pop(0).remove()
    
        
    bkg = np.array(string.split(_dictionary[_img]['bckgrintervals_' + str(_key)],','),float)
    if event.key == '6' :
        _dictionary[_img]['aplow_' + str(_key)] = ydata - peakinterp(xdata)
    if event.key == '7' :
        _dictionary[_img]['aphigh_' + str(_key)] = ydata - peakinterp(xdata)
        print('7')
    if event.key == '1' :
        bkg[0] = ydata - peakinterp(xdata)
        _dictionary[_img]['bckgrintervals_' + str(_key)] = ', '.join([str(i) for i in bkg])
    if event.key == '2' :
        bkg[1] = ydata - peakinterp(xdata)
        _dictionary[_img]['bckgrintervals_' + str(_key)] = ', '.join([str(i) for i in bkg])
    if event.key == '3' :
        bkg[2] = ydata - peakinterp(xdata)
        _dictionary[_img]['bckgrintervals_' + str(_key)] = ', '.join([str(i) for i in bkg])
    if event.key == '4' :
        bkg[3] = ydata - peakinterp(xdata)
        _dictionary[_img]['bckgrintervals_' + str(_key)] = ', '.join([str(i) for i in bkg])
    if event.key == 'c' :
        shift = ydata - peakinterp(xdata)
        peak = peak + shift
        _dictionary[_img]['peakpos_' + str(_key)] = peak

        
    line1 = ax2.plot(xs,peak,'-r')
    line2 = ax2.plot(xs,peak + float(_dictionary[_img]['aplow_'+str(_key)]),'-w')
    line3 = ax2.plot(xs,peak + float(_dictionary[_img]['aphigh_'+str(_key)]),'-w')

    bkg = np.array(string.split(_dictionary[_img]['bckgrintervals_' + str(_key)],','),float)
    line40 = ax2.plot(xs,peak + bkg[0],'-y')
    line41 = ax2.plot(xs,peak + bkg[1],'-y')
    line42 = ax2.plot(xs,peak + bkg[2],'-y')
    line43 = ax2.plot(xs,peak + bkg[3],'-y')

    ## write nosky file from dictionary
    readnoise = 16
    gain = 1
    apmedfiltlength = 61 # no idea what is it
    colfitorder, scattercut = 15, 25  # no idea what is it
    othertrace = None
    _shift = 0
    if _nsky1:
        print('\n##### Warning: extract on the trimmed image instead of the sky')
        _rawdataname = 'trimmed'
    else:
        _rawdataname = 'nosky'

    if 'spec_basic'+str(_key) in _dictionary[_img]:
        spec_basic = _dictionary[_img]['spec_basic' + str(_key)]
        spec_opt = _dictionary[_img]['spec_opt' + str(_key)]
        line5.pop(0).remove()
        line6.pop(0).remove()
        line5 = ax1.plot(xs, spec_opt,'-r' ,label='optimal extraction')
        line6 = ax1.plot(xs, spec_basic, '-b', label='basic extraction')
        ax1.set_xlabel('pixels')
        ax1.set_ylabel('counts')
        ax1.legend()


    if event.key in ['r','c','1','2','3','4','6','7']:
        from deimos import irafext
        spec_opt1, spec_basic1, skybg_opt1, spec_var1 = irafext.opextract_new(_img, 0, 0, False, 1,\
                                                                              readnoise, gain, apmedfiltlength,
                                                                              colfitorder, scattercut,
                                                                              colfit_endmask=10,
                                                                              diagnostic= False,
                                                                              production= True,\
                                                                              other = othertrace, shift=_shift,
                                                                              dictionary= _dictionary, key= _key,
                                                                              rawdataname = _rawdataname,
                                                                              bckhigh = False, bcklow = False)

        line7.pop(0).remove()
        line7 = ax1.plot(xs, spec_opt1, '-g', label='new extraction')

    if event.key =='b':
        bottom = ydata
    if event.key =='t':
        top = ydata
        
    if bottom and top:
        ax1.set_ylim(bottom,top)
        
#        # add exstraction to the dictionary
#        #  iraf dimensions (to be checked)
#        #  1 optimal extraction
#        #  2 basic extraction
#        #  3 sky
#        #  4 errors
#        _dictionary[img]['spec_basic' + str(key)]= spec_basic
#        _dictionary[img]['spec_opt' + str(key)]= spec_opt
#        _dictionary[img]['skybg_opt' + str(key)]= skybg_opt
#        _dictionary[img]['spec_var' + str(key)]= spec_var
        



        
############################################
        
        
    
def profilespec(data,dispersion):
    global _xx,_yy, center, lower, upper, l1,l2,u1,u2, line1, line2, line3, line5, fig, gs, ax1, ax2, line6,\
        line7,line8, nx,linel1,linel2,lineu1,lineu2
    print("\n##################\n 1 = bg1\n 2 = bg2\n 3 = bg3\n 4 = bg4\n 6 = lower\n 7 = upper\n c = center")
    fig = plt.figure(1,figsize=(7,9))
    plt.clf()
    gs = gridspec.GridSpec(nrows=3, 
                       ncols=2, 
                       figure=fig, 
                       width_ratios= [1, 1],
                       height_ratios=[1, 1, 1],
                       wspace=0.3,
                       hspace=0.3)

    # upper plot
    datat = data.transpose()
    ny, nx = data.shape
    sample = sigma_clip(data)
    vmin = sample.mean() - 1 * sample.std()
    vmax = sample.mean() + 3 * sample.std()
    yvals, xvals = np.indices(datat.shape)
    extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
    ax2 = fig.add_subplot(gs[0:2, 0:2])
    ax2.imshow(datat, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)

    
    if dispersion is None:
        xx = data.mean(axis=1)
    else:
        xx = data[:,dispersion-50:dispersion+50].mean(axis=1)
    xx=  (xx - np.min(xx))/(np.max(xx)-np.min(xx)) + 1e-3
    yy = np.arange(len(xx))

    _xx = xx
    _yy = yy

    high = 1.
    fwhm = 5.
    _cent = len(_yy)/2.
    # starting guess for the profile model
    guess = (high, _cent, fwhm)
    guessbound = [(0.1,1.9),(_cent-30, _cent+30),(fwhm/3.,fwhm*2)]
    params1 = minimize(get_profile_chisq, guess, args=(_yy, _xx), bounds=guessbound)

    centfit = params1['x'][1]
    fwhmfit = params1['x'][2]
    if centfit!= _cent:
        center = centfit
        fwhm = np.min([fwhmfit , 15])
        
    else:
        center = _yy[np.argmax(_xx)]
        fwhm = 7

        
    lower = center - fwhm * 2.5
    upper = center + fwhm * 2.5
    l1 = np.max([center - fwhm * 5, 5])
    l2 = np.max([center - fwhm *4,10])
    u1 = np.min([center + fwhm * 4, len(_xx)-10])
    u2 = np.min([center + fwhm * 5, len(_xx)-5])

    ax1 = fig.add_subplot(gs[2, 0:2])
    
    line5 = ax1.plot(_yy,_xx,'-b')
    line1 = ax1.plot([l1,l2],[0,0],'-k')
    line2 = ax1.plot([u1,u2],[0,0],'-k')
    line3 = ax1.plot([lower,upper],[1,1],'-k')
    line6 = ax1.plot(center,1,'or')
    line7 = ax2.plot([lower,lower],[0,nx],'--r')
    line8 = ax2.plot([upper,upper],[0,nx],'--r')
    linel1 =  ax2.plot([l1,l1],[0,nx],'--y')
    linel2 = ax2.plot([l2,l2],[0,nx],'--y')
    lineu1 = ax2.plot([u1,u1],[0,nx],'--y')
    lineu2 = ax2.plot([u2,u2],[0,nx],'--y')
    
    kid = fig.canvas.mpl_connect('key_press_event',onkeypress1)
    plt.draw()
    if pyversion>=3:
        input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
    else:
        raw_input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
    return center, lower, upper, l1,l2,u1,u2
    
def poly(x, y, order, rej_lo, rej_hi, niter):

    # x = list of x data
    # y = list of y data
    # order = polynomial order
    # rej_lo = lower rejection threshold (units=sigma)
    # rej_hi = upper rejection threshold (units=sugma)
    # niter = number of sigma-clipping iterations

    npts = []
    iiter = 0
    iterstatus = 1

    # sigma-clipping iterations
    rejected=[]
    while iiter < niter and iterstatus > 0:
        iterstatus = 0
        tmpx = []
        tmpy = []
        npts.append(len(x))
        coeffs = polyfit(x, y, order)
        fit = polyval(coeffs, x)

        # calculate sigma of fit

        sig = 0
        for ix in range(npts[iiter]):
            sig = sig + (y[ix] - fit[ix]) ** 2
        sig = math.sqrt(sig / (npts[iiter] - 1))

        # point-by-point sigma-clipping test
        for ix in range(npts[iiter]):
            if y[ix] - fit[ix] < rej_hi * sig and fit[ix] - y[ix] < rej_lo * sig:
                tmpx.append(x[ix])
                tmpy.append(y[ix])
            else:
                rejected.append([x[ix],y[ix]])
                iterstatus = 1
        x = tmpx
        y = tmpy
        iiter += 1

    # coeffs = best fit coefficients
    # iiter = number of sigma clipping iteration before convergence
    return coeffs, iiter, rejected

########################################

def tracenew(img, dictionary, key, step, verbose, polyorder, sigma, niteration,rawdataname='nosky'):
    if rawdataname=='nosky':
        data = dictionary[img][rawdataname + str(key)]
    else:
        data = dictionary[img][rawdataname + str(key)][0].data
    dispersion= None
    if verbose:
        plt.clf()
        image_plot(data,2,dictionary[img]['OBJECT'])
        if pyversion>=3:
            dispersion = input('where do you want to look for the object profile [[a]ll / 300] ? [a] ')
        else:
            dispersion = raw_input('where do you want to look for the object profile [[a]ll / 300] ? [a] ')
        if not dispersion: dispersion= 'a'
        if dispersion in ['a','N','NO','n']: dispersion= None
        else:
            dispersion=int(dispersion)
    center, lower, upper, l1,l2,u1,u2 = profilespec(data,dispersion)
    if verbose:
        print(center, lower, upper, l1,l2,u1,u2)

    ny, nx = data.shape
    # create 1d arays of the possible x and y values
    xs = np.arange(nx)
    ys = np.arange(ny)
         
    # get the median for each row
    xx0 = data.mean(axis=1)
    # the 0.01 is because does not like the zero value
    profile =  (xx0 - np.min(xx0)+0.01)/np.max(xx0)

    high = 1.
    fwhm = np.sqrt(center - lower)
    # starting guess for the profile model
    guess = (high,center, fwhm)
    guessbound = [(0.1,1.9),(center-10,center+10),(fwhm/3.,fwhm*2)]

    loop = np.arange(int(step/2),len(data[1]),step)
    centerv,highv,fwhmv=[],[],[]
    for ii in loop:
        xx1 = data[:, ii - int(step/2.): ii + int(step/2.)].mean(axis=1)
        xx1 =  (xx1 - np.min(xx1)+0.01)/np.max(xx1)
        params1 = minimize(get_profile_chisq, guess, args=(ys, xx1),bounds=guessbound)
        if verbose:
            print("best fit parameters are", params1)
        model1 = get_profile_model(params1['x'], ys)
        centerv.append(params1['x'][1])
        highv.append(params1['x'][0])
        fwhmv.append(params1['x'][2])
        
    polypar = np.polyfit(loop,centerv,polyorder)
    peakpos = np.polyval(polypar, xs)
    polypar1,niter,rejected = poly(loop, centerv, polyorder, sigma, sigma, niteration)

    peakpos1 = np.polyval(polypar1, xs)
    if verbose:
        image_plot(data,3)
        plt.plot(loop,centerv,'or')
        plt.plot(xs,peakpos,'-g')
        plt.plot(xs,peakpos1,'-c')
        plt.title(dictionary[img]['OBJECT'])
        for line in rejected:
            plt.plot(line[0],line[1],'om')
        if pyversion>=3:
            input('stop here')
        else:
            raw_input('stop here')
        plt.clf()
        
    meta={}
    meta['aplow']=lower-center
    meta['bckgrfunc']='chebyshev'
    meta['bckgr_low_reject']= 5

    if dispersion:
        meta['displine']=dispersion
    else:
        meta['displine']= 1000
        
    meta['aphigh']= upper - center
    meta['bckgrfunc_iraforder']= 1
    # this coeff are not used since we give directly the peakpos
    meta['coeffs'] = [2.0, 4.0, 10.0, 4090.0, -3.961599, -2.403275, 2.146498, -0.08873626]
    meta['bckgrintervals']= [[l1-center,l2-center],[u1-center,u2-center]]
    meta['bckgr_niterate']= 5
    meta['bckgr_high_reject']= 3

    _grism = dictionary[img]['GRATENAM']
    _slit = dictionary[img]['SLMSKNAM']   
    setup = (_grism,_slit)
    _dir = '_'.join(setup)
    imgtrace = re.sub('.fits','',img) + '_' + str(key) + '_trace.ascii'
    output = _dir + '/' + str(key)  + '/' + imgtrace
    writetrace(peakpos1,meta,'trace',output)

    for key1 in meta:
        print(meta[key1],key1)
        if key1 in ['aplow','displine','aphigh']:
            dictionary[img][key1 + '_' + str(key)] = float(meta[key1])
        else:    
            dictionary[img][key1 + '_' + str(key)] = re.sub('\[?]?','', str(meta[key1]))
            
    dictionary[img]['peakpos_'+str(key)] = peakpos1
    
    return peakpos1,centerv,highv,fwhmv,dictionary

##########################################################################
def smoothListGaussian(list, degree=5):
    # run a gaussian smooth to a list
    # last points are all the same to keep the same size 
    window = degree*2-1
    weight = np.array([1.0]*window)
    weightGauss = []
    for i in range(window):
        i = i-degree+1
        frac = i/float(window)
        gauss = 1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight = np.array(weightGauss)*weight
    smoothed = [0.0]*(len(list)-window)
    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(list[i:i+window])*weight)/sum(weight)

    smoothed = smoothed + smoothed[-1:]*int(window)
    return smoothed

############################################################
def findslit(img,key, verbose=False,cut=None):
    hdu= fits.open(img)
    x = hdu[key].data
    prof = x.mean(axis=0)
    prof = prof/np.max(prof)
    xx = np.arange(len(prof))
    prof = smoothListGaussian(list(prof), degree=5)
    prof = np.array(prof)
    if cut is None:
        cut = np.mean(prof)
    start = 0
    end = len(prof)
    point=[]
    prof1 = prof
    xx1 = xx
    mode = 'g'
    while end-start> 100:
        if mode =='g':
            if len(xx1[prof1>cut]):
                start = xx1[prof1>cut][0]
                mode = 'm'
            else:
                break
        else:
            if len(xx1[prof1>cut]):
                start = xx1[prof1<cut][0]
                mode = 'g'
            else:
                break
        point.append(start)
        prof1 = prof1[(xx1>start)]
        xx1 = xx1[(xx1>start)]

    yy = np.zeros(len(point))+cut
    if verbose:
        plt.clf()
        plt.plot(xx,prof,'-r')
        plt.plot(point,yy,'ob')

    ss = np.array(point[:-1])[np.diff(point)>100] + 20
    ee = np.array(point[1:])[np.diff(point)>100] - 20
    return list(zip(ss,ee))

#################################################

def atmoexp_correction(wavestd,fluxstd,exptime,airmass,site='mauna', verbose = False):
    exptime = 1
    import deimos
    if site =='mauna':
        extinction = _path[0] + '/resources/extinction/mauna.dat'
        data = np.genfromtxt(extinction)
        aae, yye = zip(*data)
        aae, yye = np.array(aae, float), np.array(yye, float)
        atm_std = np.interp(wavestd, aae, yye)
        aircorr = 10 ** (0.4 * np.array(atm_std) * airmass)
        fluxstd_corr = (fluxstd / exptime) * aircorr
        if verbose:
            plt.clf()
            plt.plot(wavestd,fluxstd,'-r')
            plt.plot(wavestd,fluxstd_corr,'-b')
            if pyversion>=3:
                input('stop here')
            else:
                raw_input('stop here')
        return fluxstd_corr

##########################################

def flatcombine2(flatlist, verbose = False, response = True, Saxis=0):
    """
    input: list of flats
    output: 
        - combined flat (median)
        - normalized flat
    """
    files = flatlist
    for i in range(0,len(files)):
        #hdu_i = fits.open(files[i])
#        im_i = hdu_i[0].data
        if (i==0):
            all_data = files[i] 
        elif (i>0):
            all_data = np.dstack( (all_data, files[i]))

    # do median across whole stack of flat images
    flat_stack = np.nanmedian(all_data, axis=2)

    if verbose:
        plt.figure(2)
        plt.clf()
        print('combined flat')
        image_plot(flat_stack,frame=2,_title='merged flat')
#        plt.imshow(flat_stack)

    if response:
        print('normalize flat')
        xdata = np.arange(all_data.shape[1]) # x pixels
        flat_1d = convolve(flat_stack.mean(axis=Saxis), Box1DKernel(40))
        otherdirection = convolve(flat_stack.mean(axis=1), Box1DKernel(5))
        otherdirection = otherdirection/np.max(otherdirection)
        flat_2d = flat_stack.mean(axis=Saxis)
        flat = flat_stack
        
        if Saxis==0:
            for i in range(flat_stack.shape[Saxis]):
                flat[i,:] = flat_stack[i,:] / (flat_1d  * otherdirection[i])
        else:
            for i in range(flat_stack.shape[Saxis]):
                flat[:,i] = flat_stack[:,i] / (flat_1d  * otherdirection[i])
                
        ### remve outlier in the flat
        flat[flat>1.1]=1.1
        flat[flat<0.9]=0.9

        if verbose:
#            print('here')
#            plt.figure(1)
#            plt.clf()
#            plt.plot(xdata,flat_2d,'r-')
#            plt.plot(xdata,flat_1d,'b-')
            image_plot(flat_stack,frame=1,_title='normalized flat')
            
#            plt.figure(3)
#            plt.clf()
#            plt.plot(xdata,flat_2d/flat_1d,'r-')
#            ds9 = pyds9.DS9('deimos')
#            ds9.set('frame 1')
#            ds9.set('scale zscale');
#            ds9.set_np2arr(flat)
    else:
        flat = None
        
    return flat_stack, flat

######################################
