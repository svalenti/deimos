from matplotlib import pylab as plt
plt.ion()
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
import sys
from scipy.optimize import fmin
pyversion = sys.version_info[0]
import deimos
import os

poly_arc = {3: np.array([  1.03773471e-05,   5.78487274e-01,   4.45847046e+03]),
#            7: np.array([  2.30350858e-05,  -7.64099597e-01,   9.79141140e+03]),
#            7: np.array([ -1.39838149e-13,   1.83212231e-09,  -8.83011172e-06, -6.28779911e-01,   9.64233695e+03])}
            7: np.array([ -1.39838149e-13,   1.83212231e-09,  -8.83011172e-06, -6.28779911e-01,   9.64533695e+03])}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def get_profile_model(params1, ys):
    a1, cy1, sigma1 = params1
    p1 = np.exp(-(ys - cy1)**2 / 2 / sigma1**2)
    p1 /= p1.max()
    return a1 * p1 

def get_profile_chisq(params2, ys, profile):
    model = get_profile_model(params2, ys)
    return np.sum( (profile - model)**2 / np.sqrt(np.abs(profile)) ) / (profile.size - len(params2))

def fitline(xx,yy,center,amplitude=1,sigma=3,verbose=True):
    guess = [amplitude,float(center),sigma]
    params3 = fmin(get_profile_chisq, guess, args=(xx, yy))
    model = get_profile_model(params3, xx)
    if verbose:
        plt.clf()
        plt.plot(xx,yy,'-b',linewidth=3)
        plt.plot(xx,model,'-r',linewidth=3)
        print(params3)
    return params3

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None, title=True):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    title : bool or string, optional (default = True)
        if True, show standard title. If False or empty string, doesn't show
        any title. If string, shows string as title.
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    >>> detect_peaks(x, show=True, ax=axs[0], threshold=0.5, title=False)
    >>> detect_peaks(x, show=True, ax=axs[1], threshold=1.5, title=False)
    Version history
    ---------------
    '1.0.6':
        Fix issue of when specifying ax object only the first plot was shown
        Add parameter to choose if a title is shown and input a title
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
            no_ax = True
        else:
            no_ax = False

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        if title:
            if not isinstance(title, str):
                mode = 'Valley detection' if valley else 'Peak detection'
                title = "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"% \
                        (mode, str(mph), mpd, str(threshold), edge)
            ax.set_title(title)
        # plt.grid()
        if no_ax:
            plt.show()

##################################
# add def model_lamp_std

# add crossmatch definition


def model_lamp(params, wave, counts):
    wav0, scale, c0 = params
    
    dtype = []
    dtype.append(('wav', float))
    dtype.append(('flux', float))
    model = np.zeros(counts.shape, dtype=dtype)
    model['wav'] = wav0 +  wave #+ b * dx**2
    model['flux'] = c0 + scale * counts
    return model

def model_lamp_std(params, wave, counts):
    wav0,  c0 = params
    dtype = []
    dtype.append(('wav', float))
    dtype.append(('flux', float))
    model = np.zeros(counts.shape, dtype=dtype)
    model['wav'] = wav0 +  wave #+ b * dx**2
    model['flux'] = c0  * counts
    return model

def checkwithtelluric(wave, flux, key, ref_filename, guess = (1.0, 1.001), verbose=False):
    from astropy.io import ascii
    from astropy.table import QTable
    if key == 7:
        which_side = 'red'
    if key == 3:
        which_side = 'blue'
    std = QTable([wave,flux], names=('wave', 'flux'))
    std.sort('wave')
    stdwave = std['wave']
    stdflux = std['flux']
    
    stdflux = stdflux-stdflux.min()
    stdflux = stdflux/stdflux.max()
    
    # this will be the array with atmospheric lines removed
    stdwave_cut = stdwave
    stdflux_cut = stdflux
    # cut the atmoshperic lines
    if which_side == 'red':
        atm_range = [[7150, 7420],[7580,7730],[7840,8450]] #red
    if which_side == 'blue':  
        #atm_range = [[6250,6340],[6850,7100],[7150, 7420],[7580,7730],[7840,8450]]
        atm_range = [[6250,6340],[6850,6990]]
    for j in atm_range:
        ww = (stdwave_cut < j[0]) | (stdwave_cut > j[1])
        stdwave_cut = stdwave_cut[ww]
        stdflux_cut = stdflux_cut[ww]
        
    # read the reference sky
    hdu = fits.open(ref_filename)
    y = hdu[0].data
    x = np.arange(len(y))
    A = hdu[0].header['CRVAL1']
    B = hdu[0].header['CDELT1']
    # use headers to get the wavelength calibration
    sky_wave = A +B *x #+ 100
    sky_flux =  y
    if which_side == 'red':
        ss = (sky_wave > 6500) & (sky_wave < 9200)
    if which_side == 'blue':
        ss = (sky_wave > 4500) & (sky_wave < 7200)
    sky_flux = 1 - sky_flux[ss]
    sky_wave = sky_wave[ss]
    sky_flux[sky_flux<=0] = 1e-5

    # inteprolate the array
    # after removing the telluric I need to inteprolate along the cut to get the same file dimention 
    flux_interp = interp1d(stdwave_cut, stdflux_cut, bounds_error=False )
    new_stdflux = flux_interp(stdwave)

    # the atmospheric file is usually 1 everywhwere 
    atmo = stdflux/new_stdflux
    atmo[atmo<0]=0
    if which_side == 'red':
        gg = (stdwave < 8500) #red
    if which_side == 'blue':
        gg = (stdwave > 5000)
    atmwave=stdwave[gg]
    atmflux = 1 - atmo[gg]
    atmflux[atmflux<=0] = 1e-5

    model = model_lamp_std(guess, atmwave, atmflux)
    
    # the get_lamp_difference file takes the inteprolate model file as input 
    atmomodel_interp = interp1d(sky_wave, sky_flux, bounds_error=False)
    
    # run the minization giving the interpolated atmospheric file, the initial parameter and the 
    bestparams = fmin(get_lamp_difference_std, guess, args=(atmwave, atmflux, atmomodel_interp), maxiter=10000, disp = False)
    # this should be the best parameters for shift and sclae (c)
    print(bestparams)
    shift, scalefactor = bestparams[0],bestparams[1]
    print ('myshift: '+str(shift))

    if verbose:
        plt.figure(2)
        fig2 = plt.figure(2)
        fig2.clf()
        # compare the reference spectrum and the extracted sky spectrum
        ax2 = fig2.add_subplot(2, 1, 1)
        ax22 = fig2.add_subplot(2, 1, 2)
        ax2.plot(atmwave, atmflux,'-r')
        ax2.axes.set_ylabel('Flux Density ($10^{16} f_{\lambda}$)')
        ax2.axes.set_xlabel('Wavelength ($\AA$)')
        ax2.plot(atmwave, atmflux,'-b')
        ax2.plot(sky_wave, sky_flux,'-r')
        ax2.plot(atmwave+bestparams[0], atmflux,'-g')
        
        # plot the extracted sky spectrum 
        ax22.plot(wave, flux)
        ax22.axes.set_ylabel('Counts')
        ax22.axes.set_xlabel('wavelenght');            
        if pyversion>=3:
            input('stop std')
        else:
            raw_input('stop std')
            
    return bestparams[0],bestparams[1]
###########################################################


def get_lamp_difference(params, wave, flux, skyref_interp):
    model = model_lamp(params, wave, flux)
    
    # residual
    res = model['flux'] - skyref_interp(model['wav'])
    
    return np.sum(res**2 / np.sqrt(np.abs(model['flux'])))

def get_lamp_difference_std(params, wave, flux, skyref_interp):
    model = model_lamp_std(params, wave, flux)
    
    # residual
    res = model['flux'] - skyref_interp(model['wav'])
    
    return np.sum(res**2 / np.sqrt(np.abs(model['flux'])))

###########################################################

def fitlamp(lampixel, refpeak, lampeak, deg, pixel, flux, skyref):
    global _pixel, _flux, idd, _lampixel, _refpeak, _deg, nonincl, fig, ax1, _params5, _num,  _line, _line3,\
        ax2, ax3,  _skyref_wav, _skyref_flux, _line2, _line4, _line5
        
    _pixel= pixel
    _flux= flux
    _lampixel = lampixel
    _refpeak = refpeak
    _deg = deg
    _num = 0
    _skyref_wav = skyref['wav']
    _skyref_flux = skyref['flux']
    
    
    idd = list(range(len(_lampixel)))
    nonincl = []

    _params5 = np.polyfit(_lampixel[idd], _refpeak[idd], _deg )
    p2 = np.poly1d(_params5)    
    
    fig = plt.figure(1)
    plt.clf()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    _line = ax1.plot(p2(_lampixel),_refpeak-p2(_lampixel),'.r')
    _line3 = ax1.plot(np.array(p2(_lampixel))[nonincl], (_refpeak-p2(_lampixel))[nonincl],'oc')
    ax1.set_xlim(np.min(p2(_pixel)),np.max(p2(_pixel)))

    
    ax2.plot(_skyref_wav, _skyref_flux,'-b')
    ax2.plot(p2(_pixel), _flux,'-r')
    ax2.plot(p2(_lampixel),np.ones(len(_lampixel)),'|b')
    ax2.plot(_refpeak,np.ones(len(_lampixel)),'|r')
    ax2.set_ylim(0,1.1)
    ax2.set_xlim(np.min(p2(_pixel)),np.max(p2(_pixel)))

    
    ax3.plot(_skyref_wav, _skyref_flux,'-b')
    _line2 = ax3.plot(p2(_pixel), _flux,'-r')
    ax3.set_xlim(p2(_lampixel[_num]) - 50, p2(_lampixel)[_num] + 50)
    ax3.set_ylim(0,1.1)
    _line4 = ax3.plot(p2(_lampixel[_num]),[1],'|b', label = 'ref. lamp detection')
    _line5 = ax3.plot(_refpeak[_num],[1],'|r', label = 'lamp detection')
    plt.legend()
    
    kid = fig.canvas.mpl_connect('key_press_event', onkeycazzo)
    plt.draw()
    if pyversion>=3:
        input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
    else:
        raw_input('left-click mark bad, right-click unmark, <d> remove. Return to exit ...')
    return _params5

def onkeycazzo(event):
    global _pixel,_flux, idd, _lampixel, _refpeak, _deg, nonincl, fig, ax1, _params5, _num, _line, _line3,\
        ax2, ax3,  _skyref_wav, _skyref_flux,  _line2, _line4, _line5
    
    xdata,ydata = event.xdata,event.ydata

    _params5 = np.polyfit(_lampixel[idd], _refpeak[idd], _deg )
    p2 = np.poly1d(_params5)        

    dist = np.sqrt((xdata-np.array(p2(_lampixel)))**2+(ydata - (np.array(_refpeak)- p2(_lampixel)))**2)
    ii = np.argmin(dist)
    _num = ii

    if event.key == 'a' :
        idd.append(idd[-1]+1)
        __lampixel = list(_lampixel)
        __lampixel.append(xdata)
        _lampixel = np.array(__lampixel)
        __refpeak = list(_refpeak)
        __refpeak.append(ydata)
        _refpeak = np.array(__refpeak)
        ax1.plot(xdata,ydata,'ob')
    if event.key == 'd' :
        idd.remove(ii)
        _num = ii
        for i in range(len(_lampixel)):
            if i not in idd: nonincl.append(i)
    if event.key == 'c' :
        _num = ii

    if event.key in ['1','2','3','4','5','6','7','8','9'] :
        _deg = int(event.key)

    _params5 = np.polyfit(_lampixel[idd], _refpeak[idd], _deg )
    p2 = np.poly1d(_params5)        
                
    _line.pop(0).remove()
    _line3.pop(0).remove()
    _line = ax1.plot(p2(_lampixel),_refpeak-p2(_lampixel),'.r')
    _line3 = ax1.plot(np.array(p2(_lampixel))[nonincl], (_refpeak-p2(_lampixel))[nonincl],'oc')
    ax1.set_xlim(np.min(p2(_pixel)),np.max(p2(_pixel)))

    ax2.plot(_skyref_wav, _skyref_flux,'-b')
    ax2.plot(p2(_pixel), _flux,'-r')
    ax2.plot(p2(_lampixel),np.ones(len(_lampixel)),'|b')
    ax2.plot(_refpeak,np.ones(len(_lampixel)),'|r')
    ax2.set_ylim(0,1.1)
    ax2.set_xlim(np.min(p2(_pixel)),np.max(p2(_pixel)))

    _line2.pop(0).remove()
    _line4.pop(0).remove()
    _line5.pop(0).remove()
    ax3.plot(_skyref_wav, _skyref_flux,'-b')
    _line2 = ax3.plot(p2(_pixel), _flux,'-r')
    ax3.set_xlim(p2(_lampixel[_num]) - 50, p2(_lampixel)[_num] + 50)
    ax3.set_ylim(0,1.1)
    _line4 = ax3.plot(p2(_lampixel[_num]),[1],'|b', label = 'ref. lamp detection')
    _line5 = ax3.plot(_refpeak[_num],[1],'|r', label = 'lamp detection')
    plt.legend()
    plt.draw()

#################################

def wavesolution(reference, pixel, flux, key, radius, edge, initial_solution, deg, initial_shift=0.1):
    #  read template spectrum 
    hdu  = fits.open(reference)
    dd =hdu[1].data
    xx,yy = zip(*dd)
    # normalize spectrum
    yy = yy/np.max(yy)

    skyref = {'wav': np.array(xx),
              'flux': np.array(yy)}
    # we must interpolate the reference spectrum to the model wavelengths
    skyref_interp = interp1d(skyref['wav'], skyref['flux'], bounds_error=False)


##
#    I don't understand what is happening here. if I write the file and read it again it works?!?!?!
#    
##
    imgout = 'arc_test1.ascii'
    np.savetxt(imgout, np.c_[pixel, flux], header='pixel  flux')
    # read asci extraction 
    data = np.genfromtxt('arc_test1.ascii')    
    pixel , flux= zip(*data)

    flux = flux - np.min(flux)
    flux = flux/np.max(flux)

    # use initial solution
    p = np.poly1d(initial_solution)
    wave = p(pixel)
    pixel = np.array(pixel)
    
    params = (initial_shift, 1.000001, 0.0)
    model = model_lamp(params, wave, flux)

    # compute the shift between ref and observed lamp
    guess = (initial_shift,  1.00001, 0.0)
    bestparams = fmin(get_lamp_difference, guess, args=(wave, flux, skyref_interp), maxiter=10000)
    shift = bestparams[0]
    print('###',shift)
    model = model_lamp(bestparams, wave, flux)

    # measure the peaks of the observed lamp 
    peaks= detect_peaks(flux, mph = 0.05,edge='rising',mpd=2)

    # remove peaks too close eachother 
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    close_peaks = np.diff(peaks)<10
    close_peaks_index = get_indexes(True, close_peaks)
    index_to_remove = close_peaks_index  + [i+1 for i in close_peaks_index] 
    peaks = np.delete(peaks,index_to_remove)

    # measure the peaks of the reference lamp 
    peaksref= detect_peaks(skyref['flux'], mph = 0.05,edge='rising',mpd=2)

    # remove peaks too close eachother 
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    close_peaks = np.diff(peaksref)<10
    close_peaks_index = get_indexes(True, close_peaks)
    index_to_remove = close_peaks_index  + [i+1 for i in close_peaks_index] 
    peaksref = np.delete(peaksref,index_to_remove)

    skyref['flux_peaks'] = skyref['flux'][peaksref]
    skyref['wav_peaks'] = skyref['wav'][peaksref] 

    #####################
    wave0  = wave + shift

    refpeak=[]
    lampeak=[]
    lampixel=[]
    for i,j in enumerate(skyref['wav_peaks']):
        nearest_to_reference,nearest_to_reference_idx = find_nearest(wave[peaks] + shift,  skyref['wav_peaks'][i])
        if np.abs(nearest_to_reference - skyref['wav_peaks'][i]) < radius:
            ww = [( wave0 < nearest_to_reference + edge) & ( wave0 > nearest_to_reference - edge)]
            params_fit = fitline(pixel[ww],flux[ww], peaks[nearest_to_reference_idx], 1.,  3., verbose=False)
            peak_minus_fit = np.abs(p(params_fit[1]) + shift -nearest_to_reference)
            if peak_minus_fit< radius:
                # wavelength in the reference arc
                refpeak.append(skyref['wav_peaks'][i])
        
                # wavelength in the observed arc
                lampeak.append(nearest_to_reference)
        
                # pixel in the observed arc
                lampixel.append(params_fit[1])

    lampixel = np.array(lampixel)
    refpeak = np.array(refpeak)
    lampeak = np.array(lampeak)
    finalsolution = fitlamp(lampixel, refpeak, lampeak, deg, pixel, flux, skyref)
    return finalsolution

##############################################################################################################################

def checkshift(img, dictionary, key, wave, arc, arcspec, sky, skyref, skyref_interp, setup, verbose = True ):
    dictionary[img]['arcfile' + str(key)]= arc[key]
    dictionary[img]['arcspec' + str(key)] = arcspec
    flux = dictionary[img]['spec_opt' + str(key)]
    _dir = '_'.join(setup)
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
#        bestparams = fmin(deimos.deimoswave.get_lamp_difference, guess, args=(wave0, sky0, skyref_interp), maxiter=10000)
        bestparams = fmin(get_lamp_difference, guess, args=(wave0, sky0, skyref_interp), maxiter=10000)
        if (dxa > 0).all() is False:
            shift = bestparams[0]
        else:
            shift = (-1) * bestparams[0]
    
        print('shift the spectrum of ',shift)        
        #  wavelength calibration in the database
        wave = wave + shift
            
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
        ref_filename = os.path.join(deimos.__path__[0]+'/resources/sky/','std_telluric.fits')
        imgout = 'std_'+ _dir + '_' + str(key) + '.ascii'
        np.savetxt(imgout, np.c_[wave, flux ], header='wave  flux ')        
#        shift, scalefactor = deimos.deimoswave.checkwithtelluric(wave, flux , key, ref_filename, guess=(0.001,1.0001), verbose=True)
        shift, scalefactor = checkwithtelluric(wave, flux , key, ref_filename, guess=(0.001,1.0001), verbose=True)
        print ('myshift: '+str(shift))
        print('shift the spectrum of ',shift)        
        #  wavelength calibration in the database
        wave = wave + shift
    return dictionary, wave
