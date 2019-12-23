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
from astropy.stats import sigma_clip
from scipy.optimize import fmin
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import pickle
from astropy.stats import sigma_clipped_stats
from deimos import __path__ as _path
import sys
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

def checkalldata(directory=False):
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
        print(img,dictionary[img]['type'],dictionary[img]['OBJECT'],dictionary[img]['OBSTYPE'])
        if dictionary[img]['type'] is None:
            skip.append(img) 
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
            print(setup)
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
            
            for key in ['3','7']:
                imgtrimmed = re.sub('.fits','',img) + '_' + str(key) + '_trimmed.fits'
                imgnosky   = re.sub('.fits','',img) + '_' + str(key) + '_nosky.fits'
                imgtrace   = re.sub('.fits','',img) + '_' + str(key) + '_trace.ascii'
                imgex      = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_ex.ascii'
                imgflux    = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_flux.ascii'
                imgwave    = re.sub('.fits','',img) + '_' + str(key) +  '_' + dictionary[img]['OBJECT'] + '_wave.ascii'
        
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgtrimmed):
                    print('load trimmed image')
                    hdu = fits.open(_dir + '/' + str(key)  + '/' + imgtrimmed)
                    dictionary[img]['trimmed' + str(key)] = hdu[0].data
        
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgnosky):
                    print('load nosky image')
                    hdu = fits.open(_dir + '/' + str(key)  + '/' + imgnosky)
                    dictionary[img]['nosky' + str(key)] = hdu[0].data
        
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgtrace):
                    print('load trimmed data')
                    aa, meta = readtrace(_dir + '/' + str(key) + '/' + imgtrace)
                    
                    for key1 in meta:
                        if key1 in ['aplow','displine','aphigh']:
                            dictionary[img][key1 + '_' + str(key)] = float(meta[key1])
                        else:    
                            dictionary[img][key1 + '_' + str(key)] = re.sub('\[?]?','', meta[key1])
                        dictionary[img]['peakpos_'+str(key)] = aa
                        
                else:
                    print('nothing found',imgtrace)

                
                from astropy.io import ascii                    
                if os.path.isfile(_dir + '/' + str(key) + '/' + imgflux):
                    aa = ascii.read(_dir + '/' + str(key) + '/' + imgflux)
                    print(aa)
                    if key==7:
                        dictionary[img]['spec_basic' + str(key)] = aa['spec_basic'][::-1]
                        dictionary[img]['spec_opt' + str(key)] = aa['spec_opt'][::-1]
                        dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt'][::-1]
                        dictionary[img]['spec_var' + str(key)] = aa['spec_var'][::-1]
                        dictionary[img]['mysky' + str(key)] = aa['mysky'][::-1]
                        dictionary[img]['mybasic' + str(key)] = aa['mybasic'][::-1]
                        dictionary[img]['wave' + str(key)] = aa['wave'][::-1]
                        dictionary[img]['arcspec' + str(key)] = aa['arcspec'][::-1]
                        dictionary[img]['response' + str(key)] = aa['response'][::-1]
                        dictionary[img]['spec_flux' + str(key)] = aa['spec_flux'][::-1]
                    else:
                        dictionary[img]['spec_basic' + str(key)] = aa['spec_basic']
                        dictionary[img]['spec_opt' + str(key)] = aa['spec_opt']
                        dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt']
                        dictionary[img]['spec_var' + str(key)] = aa['spec_var']
                        dictionary[img]['mysky' + str(key)] = aa['mysky']
                        dictionary[img]['mybasic' + str(key)] = aa['mybasic']
                        dictionary[img]['wave' + str(key)] = aa['wave']
                        dictionary[img]['arcspec' + str(key)] = aa['arcspec']
                        dictionary[img]['response' + str(key)] = aa['response']
                        dictionary[img]['spec_flux' + str(key)] = aa['spec_flux']
                    
                elif os.path.isfile(_dir + '/' + str(key) + '/' + imgwave):
                    aa = ascii.read(_dir + '/' + str(key) + '/' + imgwave)
                    print(aa)
                    print(imgwave)
                    dictionary[img]['spec_basic' + str(key)] = aa['spec_basic']
                    dictionary[img]['spec_opt' + str(key)] = aa['spec_opt']
                    dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt']
                    dictionary[img]['spec_var' + str(key)] = aa['spec_var']
                    dictionary[img]['mysky' + str(key)] = aa['mysky']
                    dictionary[img]['mybasic' + str(key)] = aa['mybasic']
                    dictionary[img]['wave' + str(key)] = aa['wave']
                    dictionary[img]['arcspec' + str(key)] = aa['arcspec']

                      
                elif os.path.isfile(_dir + '/' + str(key) + '/' + imgex):
                    aa = ascii.read(_dir + '/' + str(key) + '/' + imgex)
                    dictionary[img]['spec_basic' + str(key)] = aa['spec_basic']
                    dictionary[img]['spec_opt' + str(key)] = aa['spec_opt']
                    dictionary[img]['skybg_opt' + str(key)] = aa['skybg_opt']
                    dictionary[img]['spec_var' + str(key)] = aa['spec_var']
                    dictionary[img]['mysky' + str(key)] = aa['mysky']
                    dictionary[img]['mybasic' + str(key)] = aa['mybasic']

                    
    for img in skip:
        del dictionary[img]
    return dictionary, setup_object, setup_arc, setup_flat

########################################################################

def trim_rotate_split(setup_object,setup_flat,setup_arc,dictionary, setup, force=False, verbose=False):
    _dir = '_'.join(setup)
    ############ split files
    for img in setup_object[setup] + setup_flat[setup]+setup_arc[setup]:
        for key in [3,7]:
            trimimage = False
            imgtrimmed = re.sub('.fits','',img) + '_' + str(key) + '_trimmed.fits'
            if os.path.isfile(_dir + '/' + str(key) + '/' + imgtrimmed) and force is False:
                if verbose:
                    print('trimmed image already there')
                trimimage = False
            else:
                trimimage = True

            if trimimage:
                if key==3:
                    xmin,xmax = 710,860
                else:
                    xmin,xmax = 1205,1350
          
          
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
                dictionary[img]['trimmed'+str(key)] = trimmed

                if not os.path.isdir(_dir):
                    os.mkdir(_dir)
                if not os.path.isdir(_dir + '/' + str(key)):
                    os.mkdir(_dir + '/' + str(key))

                hdu = dictionary[img]['trimmed' + str(key)]
                _out = fits.ImageHDU(data=hdu.data, header=hdu.header)
                fits.writeto(_dir + '/' + str(key)  + '/' + imgtrimmed, _out.data,header=_out.header,overwrite='yes')
            else:
                if verbose:
                    print('read trimmed image from file')
                hdu = fits.open(_dir + '/' + str(key)  + '/' + imgtrimmed)
                dictionary[img]['trimmed' + str(key)] = hdu[0].data
    return dictionary

######################################################################

def makeflat(setup_flat,dictionary,key):
    masterflat3 = False
    masterflat7 = False
    ######### make master flat 3 
    flatlist = []
    for img in setup_flat[key]:
        if 'trimmed3' in dictionary[img]:
            flatlist.append(dictionary[img]['trimmed3'])
            masterflat3 = np.mean(flatlist,axis=0)
        else:
            print('ERROR: Flat not trimmed3')

    ######### make master flat 7
    flatlist = []
    for img in setup_flat[key]:
        if 'trimmed7' in dictionary[img]:
            flatlist.append(dictionary[img]['trimmed7'])
            masterflat7 = np.mean(flatlist,axis=0)
        else:
            print('ERROR: Flat not trimmed7')
    return masterflat3, masterflat7

###############################################################

def image_plot(image,frame=3):
    # determine the image pixel distribution (used for displaying below)
    sample = sigma_clip(image)
    vmin = sample.mean() - 1 * sample.std()
    vmax = sample.mean() + 3 * sample.std()
    yvals, xvals = np.indices(image.shape)
    extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
    plt.figure(frame)
    plt.imshow(image, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)
    plt.xlabel('Column Number')
    plt.ylabel('Row Number');

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
    image = dictionary[img0]['trimmed'+str(ext)].data                
    # this is an arc, we do not need to mask
    skymask = np.ones(image.shape, dtype=bool)
    
    # show the mask
    if verbose:
        plt.figure(3)
        plt.clf()
        plt.imshow(skymask, origin='lower', aspect='auto');
        if pyversion>=3:
            input('stop')
        else:
            raw_input('stop')
            
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
    print(dxs)
    
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

        # just plot offsets for a few of the rows across the image
        plt.clf()
        order = 2
        for y in range(10, ny, 40):
            p = plt.plot(cols, lambdas[y, cols] - xs[cols], 'o')
            c = np.polyfit(cols, lambdas[y, cols] - xs[cols], order)
            plt.plot(xs, np.polyval(c, xs), c=p[0].get_color(), label='row {}'.format(y))

        plt.legend()
        plt.xlabel('Column Number')
        plt.ylabel('Wavelength Offset from Middle Row');

    if pyversion>=3:        
        input('stop')
    else:
        raw_input('stop')
    return lambdas

################################################3

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
        sky = dictionary[img]['sky' + str(key)]
        image = dictionary[img]['trimmed' + str(key)]
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
            #print(len(ymin),len(xs))
            #ax1.plot(xs,ymin,'-c')
            #ax1.plot(xs,ymax,'-c')
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
                magnitude.append(string.split(i)[3])
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

def DefFluxCal(obj_wave, obj_flux, stdstar='', mode='spline', polydeg=4,
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
    print(std_dir)
    print(stdstar2)
    if os.path.isfile(os.path.join(std_dir, stdstar2)):
        std_wave, std_mag, std_wth = np.genfromtxt(os.path.join(std_dir, stdstar2),
                                                   skip_header=1, unpack=True)
        # standard star spectrum is stored in magnitude units
        std_flux = _mag2flux(std_wave, std_mag)

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
            sensfunc2 = fitsens(obj_wave, obj_wave_ds, LogSensfunc, mode, polydeg0, obj_flux,std_wave, std_flux)
    else:
        sensfunc2 = np.zeros_like(obj_wave)
        print('ERROR: in DefFluxCal no valid standard star file found at ')
    print(os.path.join(std_dir, stdstar2))

    return 10**sensfunc2

####################################

def fitsens(obj_wave, obj_wave_ds, LogSensfunc, mode, polydeg0, obj_flux,std_wave, std_flux, obj_flux_ds):
    global idd, _obj_wave, _LogSensfunc, _polydeg0, _mode, lines, lines2,lines3, _obj_wave_ds, ax2, ax22, fig,\
        _obj_flux, _std_wave, _std_flux, nonincl, ax3, _obj_flux_ds
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
    idd = range(len(obj_wave_ds))
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

    
    ax2.plot(_obj_wave_ds, _LogSensfunc, 'ko', label='sensfunc')
    lines =    ax2.plot(_obj_wave, sensfunc2, '-b', label='interpolated sensfunc')
    ax2.set_xlabel('Wavelength')
    ax2.set_ylabel('log Sensfunc')

    lines2 = ax22.plot(obj_wave, obj_flux*(10**sensfunc2),'k',
             label='applied sensfunc')
    ax22.plot(std_wave, std_flux, 'ro', alpha=0.5, label='standard flux')


    lines3 = ax3.plot(_obj_wave, _obj_flux*(10**sensfunc2),'k',
             label='applied sensfunc')
    print(_obj_wave_ds)
    print(len(_obj_flux_ds))
    print(len(sensfunc2))
    ax3.plot(_obj_wave_ds, _obj_flux_ds*(10**LogSensfunc), 'bo', label='downsample observed')
    
    plt.legend()
    plt.show()
        
    kid = fig.canvas.mpl_connect('key_press_event',onkeypress)
#    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    plt.draw()
    raw_input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')

    return sensfunc2

####################################
def onkeypress(event):
    global idd, _obj_wave, _LogSensfunc, _polydeg0, _mode, lines, lines2, lines3, _obj_wave_ds, ax2, ax22, fig,\
        _obj_flux, _std_wave, _std_flux, nonincl, ax3, _obj_flux_ds

    
    xdata,ydata = event.xdata,event.ydata
    dist = np.sqrt((xdata-np.array(_obj_wave_ds))**2+(ydata-np.array(_LogSensfunc))**2)
    ii = np.argmin(dist)
    
    
    if event.key == 'a' :
        idd.append(idd[-1]+1)
        print(_obj_wave_ds)
        print(xdata)
        __obj_wave_ds = list(_obj_wave_ds)
        __obj_wave_ds.append(xdata)
        _obj_wave_ds = np.array(__obj_wave_ds)
        __LogSensfunc = list(_LogSensfunc)
        __LogSensfunc.append(ydata)
        _LogSensfunc = np.array(__LogSensfunc)
        print(_obj_wave_ds)
        ax2.plot(xdata,ydata,'db',ms=10)

    if event.key == 'd' :
        idd.remove(ii)
        for i in range(len(_obj_wave_ds)):
            if i not in idd: nonincl.append(i)

    if event.key in ['1','2','3','4','5','6'] :
        _polydeg0=int(event.key)
        print _polydeg0
            
    print(nonincl)
    print(idd)
    print('####',event.key)


    if _mode=='linear':
        sensfunc2 = np.interp(_obj_wave, _obj_wave_ds[idd], _LogSensfunc[idd])
    elif _mode=='spline':
        spl = UnivariateSpline(np.array(_obj_wave_ds)[idd], np.array(_LogSensfunc)[idd], ext=0, k=2 ,s=0.0025)
        sensfunc2 = spl(_obj_wave)
    elif _mode=='poly':
        print('type the order of the polyfit in the figure')
        fit = np.polyfit(np.array(_obj_wave_ds)[idd], np.array(_LogSensfunc)[idd], float(_polydeg0))
        sensfunc2 = np.polyval(fit, _obj_wave)

    ax2.plot(_obj_wave_ds, _LogSensfunc,'ok')
    ax2.plot(np.array(_obj_wave_ds)[nonincl], np.array(_LogSensfunc)[nonincl],'ow')
    lines.pop(0).remove()
    lines2.pop(0).remove()
    lines3.pop(0).remove()
    
    lines =  ax2.plot(_obj_wave, sensfunc2, '-b', label='interpolated sensfunc')

    lines2 = ax22.plot(_obj_wave, _obj_flux*(10**sensfunc2),'k',
             label='applied sensfunc')
    ax22.plot(_std_wave, _std_flux, 'ro', alpha=0.5, label='standard flux')

    lines3 = ax3.plot(_obj_wave, _obj_flux*(10**sensfunc2),'k',
             label='applied sensfunc')
#    ax3.plot(np.array(_obj_wave_ds)[idd], np.array(_obj_flux_ds)[idd]*(10**np.array(_LogSensfunc)[idd]), 'bo', label='downsample observed')
    ax3.plot(np.array(_obj_wave_ds)[nonincl], np.array(_obj_flux_ds)[nonincl]*(10**np.array(_LogSensfunc)[nonincl]), 'ro', label='')
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
    import string
    from astropy.io import ascii
    aa = ascii.read(ascifile)
    bb = [string.split(i,'=') for i in aa.meta['comments']]
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
    parameters = ''.join([' %s= %s\n' % (line,meta[line]) for line in meta])[:-2]
    print(parameters)
    cc = '%s\n %s ' %(tablename,parameters)
#    dd =[tablename] + ['# %s %s' % (line,meta[line]) for line in meta]+list(trace)
    np.savetxt(output,trace, header= parameters)
    
###################################################
