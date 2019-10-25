import astropy
import numpy
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
from astropy.stats import sigma_clip
from scipy.optimize import fmin
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import pickle

poly_arc = {3: np.array([  1.03773471e-05,   5.78487274e-01,   4.45847046e+03]),
            7: np.array([  2.30350858e-05,  -7.64099597e-01,   9.79141140e+03])}

plt.ion()


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

###############################################################################

def checkalldata(directory=False):
    if directory is not False:
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
    for img in dictionary:
        print(img,dictionary[img]['type'],dictionary[img]['OBJECT'],dictionary[img]['OBSTYPE'])
        if dictionary[img]['type'] is None:
            print(dictionary[img])
            answ = input('what is it?')
    
        _grism = dictionary[img]['GRATENAM']
        _slit = dictionary[img]['SLMSKNAM']        
    
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
            
    return dictionary, setup_object, setup_arc, setup_flat

########################################################################

def trim_rotate_split(setup_object,setup_flat,setup_arc,dictionary,key):
    xmin3,xmax3 = 710,860
    xmin7,xmax7 = 1205,1350
    ############ split files
    for img in setup_object[key] + setup_flat[key]+setup_arc[key]:
#        img3=re.sub('.fits','',img) + '_3.fits'
#        img7=re.sub('.fits','',img) + '_7.fits' 
        _header3 = dictionary[img]['fits'][3].header
        _header7 = dictionary[img]['fits'][7].header
        _data3 = np.transpose(dictionary[img]['fits'][3].data)
        _data7 = np.transpose(dictionary[img]['fits'][7].data)

        for ll in ['DATASEC','DETSIZE','DETSEC']:
            del _header3[ll]
            del _header7[ll]
        
        science3 = CCDData(data=_data3,header=_header3,unit=u.adu)
        science7 = CCDData(data=_data7,header=_header7,unit=u.adu)
        # add header from the 
        _header3['exptime'] = dictionary[img]['EXPTIME']
        _header3['MJD-OBS'] = dictionary[img]['MJD-OBS']
        _header3['OBJECT']  = dictionary[img]['OBJECT']
        _header3['OBSTYPE'] = dictionary[img]['OBSTYPE']
        _header3['AIRMASS'] = dictionary[img]['AIRMASS']
        _header3['RA']      = dictionary[img]['RA']
        _header3['DEC']     = dictionary[img]['DEC']
        
        _header7['exptime'] = dictionary[img]['EXPTIME']
        _header7['MJD-OBS'] = dictionary[img]['MJD-OBS']
        _header7['OBJECT']  = dictionary[img]['OBJECT']
        _header7['OBSTYPE'] = dictionary[img]['OBSTYPE']
        _header7['AIRMASS'] = dictionary[img]['AIRMASS']
        _header7['RA']      = dictionary[img]['RA']
        _header7['DEC']     = dictionary[img]['DEC']

        #  trim images 
        trimmed3 = ccdproc.trim_image(science3, fits_section='[:,' + str(xmin3) + ':' + str(xmax3) + ']')
        trimmed7 = ccdproc.trim_image(science7, fits_section='[:,' + str(xmin7) + ':' + str(xmax7) + ']')
        dictionary[img]['trimmed3'] = trimmed3
        dictionary[img]['trimmed7'] = trimmed7
    return dictionary

######################################################################

def makeflat(setup_flat,dictionary,key):
    masterflat3 = False
    masterflat7 = False
    ######### make master flat 3 
    flatlist = []
    for img in setup_flat[key]:
        if 'trimmed3' in dictionary[img]:
            flatlist.append(dictionary[img]['trimmed3'].data)
            masterflat3 = np.mean(flatlist,axis=0)
        else:
            print('ERROR: Flat not trimmed3')

    ######### make master flat 7
    flatlist = []
    for img in setup_flat[key]:
        if 'trimmed7' in dictionary[img]:
            flatlist.append(dictionary[img]['trimmed7'].data)
            masterflat7 = np.mean(flatlist,axis=0)
        else:
            print('ERROR: Flat not trimmed7')
    return masterflat3, masterflat7

###############################################################

def image_plot(image):
    # determine the image pixel distribution (used for displaying below)
    sample = sigma_clip(image)
    vmin = sample.mean() - 1 * sample.std()
    vmax = sample.mean() + 3 * sample.std()
    yvals, xvals = np.indices(image.shape)
    extent = (xvals.min(), xvals.max(), yvals.min(), yvals.max())
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
    fig,ax = plt.subplots()
    ax.plot(ys, rowaverage)
    ax.set_xlabel('Row Number (y-coordinate)'), plt.ylabel('Normalized Row Average')
    plt.grid();
    # find the rows with object light
    objrows = ys[rowaverage > objectcut]

    rangeobj = {3:'70,100', 7:'55,90'}
    if interactive:
        obj = input('give object position [eg ' + str(rangeobj[key]) + ']')
        if not obj:
            obj = rangeobj[key]
        start,end= obj.split(',')
        objrows = np.arange(int(start),int(end))
    else:
        # add some margin to object rows
        ngrow = 5 # number of rows to include above and below object rows
        newobjrows = []
        for row in objrows:
            newobjrows.extend([row + i for i in np.arange(-ngrow, ngrow + 1)])
        objrows = np.unique(newobjrows)
        
    # mask to mark sky rows
    skymask = np.ones(image.shape, dtype=bool)
    skymask[objrows, :] = False
    
    # also exclude bad rows
    badrows = ys[rowaverage < -0.05]
    skymask[badrows, :] = False
    
    # rows with mostly sky background light
    skyrows = ys[skymask.mean(axis=1) == 1]

    # median (unmasked) sky spectrum and standard deviation
    medspec = np.median(image[skyrows, :], axis=0)
    stdspec = np.std(image[skyrows, :], axis=0, ddof=1)
    
    # exclude deviant pixels from the skymask
    pull = (image - medspec) / stdspec
    w = pull > 5
    skymask[w] = False

    if interactive:
        plt.close()
        plt.clf()
        plt.figure(figsize=(15, 3))
        plt.imshow(skymask, origin='lower', aspect='auto');

    plt.close()
    return objrows, skymask
            
###################################################

def retify_frame(img0,ext=3, verbose=False):            
    image = dictionary[img0]['trimmed'+str(ext)].data                
    # this is an arc, we do not ned to mask
    skymask = np.ones(image.shape, dtype=bool)
    
    # show the mask
    if verbose:
        plt.close()
        plt.figure(figsize=(15, 3))
        plt.imshow(skymask, origin='lower', aspect='auto');
        input('stop')

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
        plt.close()
        
        # plot stamp values against column numbers
        plt.close()
        plt.clf()
        plt.plot(xs_stamp.ravel(), stamp.ravel(), 'r.');
        plt.xlabel('Column Number'), plt.ylabel('Counts');
        input('stop')
        plt.close()
    
    # pixel offsets from the refernece pixel
    dxs = xs_stamp - col
    dys = ys_stamp - row

    # parameter guess
    guess = (0, 0)

    # get the wavelength offsets and plot vs. counts
    dls = get_dl_model(guess, dxs, dys)
    plt.close()
    plt.clf()
    plt.plot(dls.ravel(), stamp.ravel(), 'r.')
    plt.xlabel('Wavelength Offset')
    plt.ylabel('Counts');
    input('stop')

    # fit a spline to the data and plot
    x, y, spl = get_profile_spl(dls, stamp)

    
    plt.clf()
    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(x, y, 'r.')
    axarr[0].plot(x, spl(x))
    axarr[1].plot(x, y - spl(x), 'r.')
    plt.ylim(-200, 200)
    plt.xlabel('Wavelength Offset');
    input('stop')
    plt.close()
    
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
        pickle.dump(lambdas, open('lambdas' + str(ext) + '.dat', 'wb'))

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
    input('stop')
    return lambdas

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
    plt.plot(model['wav'], model['flux'], label='Extracted Sky Background');
    plt.plot(skyref['wav'], skyref['flux'], label='Reference Sky Spectrum');
    plt.xlim(model['wav'].min(), model['wav'].max());
    plt.ylim(model['flux'].min() - 0.25, model['flux'].max() + 0.25)

    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Scaled Counts or Flux')
    plt.legend();

###########################################
    
def trace(img,dictionary, g0=10, g1=85, g2=3, key=3):
    if 'sky' + str(key) in dictionary[img]:
        sky = dictionary[img]['sky' + str(key)].data
        image = dictionary[img]['trimmed' + str(key)].data
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
         
        # fit the relation with a polynomial
        ind = 0 # which trace 0 or 1?
        t_order = 3
        trace_c = np.polyfit(cols, ycenter[:, ind], t_order)
        fig, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(cols, ycenter[:, ind], 'ro')
        axarr[0].plot(xs, np.polyval(trace_c, xs), 'r')
        axarr[0].axes.set_ylabel('y-coordinate')
        axarr[1].plot(cols, ycenter[:, ind] - np.polyval(trace_c, cols), 'ro')
        axarr[1].axes.set_ylim(-0.5, 0.5)
        axarr[1].axes.set_ylabel('Fit Residual (pixels)')
        plt.xlabel('Column Number');
        input('trace completed')

        slitpos = yvals - np.polyval(trace_c, yvals)
        
        dictionary[img]['trace' + str(key)] = slitpos
        dictionary[img]['tracefit' + str(key)] = trace_c
    return dictionary

##################################################    

def extract(img,dictionary, key=3):
    if 'trace' + str(key) in dictionary[img]:
        sky = dictionary[img]['sky' + str(key)].data
        image = dictionary[img]['trimmed' + str(key) ].data
        nosky = image - sky
        
        # get the a pixel coordinate near the image center
        ny, nx = image.shape
        cy, cx = ny//2, nx//2
        # create 1d arays of the possible x and y values
        xs = np.arange(nx)
        ys = np.arange(ny)
        # pixel coordinates for each pixel
        yvals, xvals = np.indices(image.shape)
         
        slitpos = dictionary[img]['trace' + str(key)]
        trace_c = dictionary[img]['tracefit' + str(key)]
         
        # normalize to the pixel brightness at the trace center
        yinds = (np.round(np.polyval(trace_c, xs))).astype(int)
        normed = nosky / nosky[yinds, xs]

        # get 1D arrays with the positions along the slit and the normalized counts
        pos = slitpos.flatten()
        counts = normed.flatten()
         
        # sort by slit position
        sort_inds = pos.argsort()
        pos, counts = pos[sort_inds], counts[sort_inds]
         
        # fit a spline to model the spatial profile
        t = np.linspace(pos.min() + 2, pos.max() - 2, ny // 2) # spline knot points
        profile_spl = LSQUnivariateSpline(pos, counts, t)

        # remove outliers and re-fit
        diff = counts - profile_spl(pos)
        sample = sigma_clip(diff)
        w = ((np.abs(diff) / sample.std()) < 5) & np.isfinite(diff)
        profile_spl = LSQUnivariateSpline(pos[w], counts[w], t)
        
        # create the profile image
        profile_image = profile_spl(slitpos)
        
        # de-weight negative values in provile_image
        profile_image[profile_image < 0] = 0
        
        # select which rows to sum
        w = (slitpos > -10) & (slitpos < 10)
        ymin, ymax = yvals[w].min(), yvals[w].max()
    
        # calculate the sum
        spec_basic = nosky[ymin:ymax, :].sum(axis=0)
        
        # sky background
        skybg_basic = np.array(sky)[ymin:ymax, :].sum(axis=0)
         
        # calculate the weighted average (for each column)
        spec_opt = (nosky * profile_image)[ymin:ymax, :].sum(axis=0) / profile_image.sum(axis=0)
        
        # calculate the bias factor needed to scale the average to a sum
        bias_factor = np.median(spec_basic / spec_opt)
        spec_opt *= bias_factor
        
        # same for the sky background
        skybg_opt = (sky * profile_image)[ymin:ymax, :].sum(axis=0) / profile_image.sum(axis=0)
        bias_factor_sky = np.median(skybg_basic / skybg_opt)
        skybg_opt *= bias_factor_sky
        skybg_opt -= np.min(skybg_opt)
 
        # plot the extracted spectrum
        plt.clf()
        plt.plot(xs, spec_basic, label='basic extraction')
        plt.plot(xs, spec_opt, label='optimal extraction')
        plt.legend()
        dictionary[img]['spec_basic' + str(key)]= spec_basic
        dictionary[img]['spec_opt' + str(key)]= spec_opt
        dictionary[img]['skybg_opt' + str(key)]= skybg_opt
        input('extraction completed')
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
    
#######################################################
#            # apply flat                
#            if masterflat3 is not False:
#                # get the a pixel coordinate near the image center
#                ny, nx = masterflat3.shape
#                cy, cx = ny//2, nx//2
#                # create 1d arays of the possible x and y values
#                xs = np.arange(nx)
#                ys = np.arange(ny)
#                # pixel coordinates for each pixel
#                yvals, xvals = np.indices(masterflat3.shape)
#                from scipy.interpolate import LSQBivariateSpline, LSQUnivariateSpline

#                lambdafit = np.zeros(masterflat3.shape)
#                x = lambdafit.ravel()
#                y = masterflat3.ravel()
#                weights = np.sqrt(np.abs(y)) 
#                wsort = x.argsort()
#                x, y, weights = x[wsort], y[wsort], weights[wsort]
#                t = np.linspace(x.min() + 1, x.max() - 1, nx)
#                spl = LSQUnivariateSpline(x, y, t, weights)
#                plt.plot(x, spl(x))
#                plt.xlabel('Wavelength ($\AA$)')
#                plt.ylabel('Counts');
#                input('stop')
###############################################################
    
###############################################################
