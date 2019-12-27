####
import numpy as np
import numpy.polynomial.legendre as leg
import numpy.polynomial.chebyshev as cheb
from astropy.io import fits
import scipy.ndimage as nd
from astropy import modeling
from datetime import datetime
from matplotlib import pylab as plt
import sys
import re
import string

    
# Class to read and interpret IRAF aperture database files.

class aperture_params:
    """parameters read from an IRAF ap database file."""
    def __init__(self, froot="", dispaxis=2):
        if froot:
            froot = froot.replace(".fits", "")
            reading_coeffs = False  # coeffs get parsed differently.
            ncoeffs = 0
            self.coeffs = []
            reading_background = False
            with open("database/ap%s" % froot, "r") as apf:
               for l in apf:
                  if l[0] != '#':
                     if 'background' in l:
                        reading_background = True
                     if 'axis' in l:
                         reading_background = False
                     if 'curve' in l:
                         reading_coeffs = True
                         reading_background = False # just in case.
                     x = l.split()
                     if x: # no blank lines
                         if 'image' in x[0]:
                            self.imroot = x[1]
                         if 'aperture' in x[0]:
                             self.apnum = int(x[1])
                             if self.apnum > 1:
                                 print("Warning -- apnum > 1!")
                                 print("Multiple apertures are not implemented.")
                         if 'enter' in x[0]:
                             if dispaxis == 2:
                                 self.center = float(x[1])
                                 self.displine = float(x[2])
                             else:
                                 self.center = float(x[2])
                                 self.displine = float(x[1])
                         if 'low' in x[0] and 'reject' not in x[0]:
                             # print(l, x)
                             if dispaxis == 2:
                                 self.aplow = float(x[1])
                                 self.lowline = float(x[2])
                             else:
                                 self.aplow = float(x[2])
                                 self.lowline = float(x[1])
                         if 'high' in x[0] and 'reject' not in x[0]:
                             if dispaxis == 2:
                                 self.aphigh = float(x[1])
                                 self.highline = float(x[2])
                             else:
                                 self.aphigh = float(x[2])
                                 self.highline = float(x[1])
                         if reading_background:
                             if 'sample' in x[0]:  # this is not consistently formatted.  Ugh.
                                 self.bckgrintervals = []
                                 if len(x) == 2:   # lower and upper joined by comma and no space
                                     y = x[1].split(',')
                                 else :            # lower and upper parts space-separated.
                                     y = x[1:]
                                 for yy in y:
                                     z = yy.split(':')
                                     # print(z)
                                     self.bckgrintervals.append([float(z[0]), float(z[1])])
                             if 'function' in x[0]:
                                 self.bckgrfunc = x[1]
                             if 'order' in x[0]:
                                 self.bckgrfunc_iraforder = int(x[1])
                             if 'niterate' in x[0]:
                                 self.bckgr_niterate = int(x[1])
                             # rejecting both low and high pixels is a bit awkward later.
                             # low pixels are not much of a problem, usually, so didn't
                             # implement a low-reject scheme.
                             #if 'low_reject' in x[0]:
                             #    self.bckgr_low_reject = float(x[1])
                             if 'high_reject' in x[0]:
                                 self.bckgr_high_reject = float(x[1])
                         if reading_coeffs and 'curve' not in l:
                             self.coeffs.append(float(x[0]))

        else:
           print("Need a valid image name.")

    # These were all done neatly with f-string but reverted for
    # compatibility with python 2.7.  Even after
    # from  __future__ import print_statement
    # f-strings will not parse in pyhon 2.7.

    def repeat_back(self):
        print("root ", self.imroot, ", aperture ", self.apnum)
        print("center ", self.center, " displine ", self.displine)
        print("aplow ", self.aplow, ", aphigh ", self.aphigh)
        print("lowline ", self.lowline, "  highline ", self.highline)
        print("bckgrfunc ", self.bckgrfunc, "iraf_order ", self.bckgrfunc_iraforder)
        print("niterate ", self.bckgr_niterate, " high_reject ", self.bckgr_high_reject)
        print("bckgr intervals:")
        for b in self.bckgrintervals:
           print(" start ", b[0], " end ", b[1])
        print("coeffs:")
        for c in self.coeffs:
           print(c)

    def evaluate_curve(self, pixlims=None):
        ic = irafcurve(self.coeffs)
        # ic.repeat_back()
        y = ic.evaluate_by1(pixlims)
        # curve is relative to center so need to add center to get real value.
        return self.center + y

# May want to reuse irafcurve if I write code to do ident,
# reident, or other things.

class irafcurve:
    """A fit generated by the iraf curvefit routines, e.g. a
       an aptrace and possibly a wavelenght fit (not tested yet tho')."""

    def __init__(self, fitparams):
        # handed a list or tuple of firparams (which includes flags
        # for type of fit) sets up the irafcurve instance.

        # should be an int but it i'nt sometimes
        typecode = int(fitparams[0] + 0.001)

        if typecode == 1: self.curvetype = 'chebyshev'
        elif typecode == 2: self.curvetype = 'legendre'
        elif typecode == 3: self.curvetype = 'spline3'
        else:
            print("Unknown fit type: ", fitparams[0])

        # the 'iraforder' is not the conventional order; it's
        # the number of coefficients for a polynomial, so a
        # a straight line has iraforder = 2.  For a spline3 it is
        # the number of spline segments.

        self.iraforder = int(fitparams[1] + 0.001)

        self.xrange = (fitparams[2], fitparams[3])
        self.span = self.xrange[1] - self.xrange[0]
        self.sumrange = self.xrange[0] + self.xrange[1]

        self.fitcoeffs = np.array(fitparams[4:])

        # Numpy provides built-in legendre and chebyshev apparatuses that make
        # this trivial.  The numpy spline3 is apparently oriented toward interpolation,
        # and wasn't as easy to adapt, though I'm probably missing something.  My own
        # spline3 stuff works correctly though it's more awkward.

        if self.curvetype == 'legendre':
            self.lpoly = leg.Legendre(self.fitcoeffs, domain = [self.xrange[0], self.xrange[1]])

        if self.curvetype == 'chebyshev':
            self.chpoly = cheb.Chebyshev(self.fitcoeffs, domain = [self.xrange[0], self.xrange[1]])

    def repeat_back(self):
        # be sure the fit read correctly
        print("curvetype ", self.curvetype, " iraforder ", self.iraforder)
        print("xrange ", self.xrange)
        print("span, sumrange ", self.span, self.sumrange)
        print("coeffs ", self.fitcoeffs)

    def evalfit(self, x):  # evaluate fit for an array of x-values.
        # translated from C, taking advantage of array arithmetic.

        if self.curvetype == 'spline3':

            # this is by far the most complicated case.

            xnorm = (2. * x - self.sumrange) / self.span
            splcoo = self.iraforder * (x - self.xrange[0]) / self.sumrange
            jlo = splcoo.astype(int)  # this is 0 for pixels in first segment, 1 for 2nd etc
            a = (jlo + 1) - splcoo    # these are basically x-values referred to segment boundaries
            b = splcoo - jlo

            # make four blank arrays

            coef0 = np.zeros(xnorm.shape)
            coef1 = np.zeros(xnorm.shape)
            coef2 = np.zeros(xnorm.shape)
            coef3 = np.zeros(xnorm.shape)

            # fill the arrays piecewise with the appropriate
            # spline coefficients.  Then the evaluation can be
            # done entirely with array arithmentic.

            for i in range(self.iraforder):
                np.place(coef0, jlo == i, self.fitcoeffs[i])
                np.place(coef1, jlo == i, self.fitcoeffs[i+1])
                np.place(coef2, jlo == i, self.fitcoeffs[i+2])
                np.place(coef3, jlo == i, self.fitcoeffs[i+3])

            y = coef0 * a ** 3 + coef1 * (1. + 3. * a * (1 + a * b)) + \
                                 coef2 * (1. + 3. * b * (1 + a * b)) + \
                                 coef3 * b ** 3

            return y

        elif self.curvetype == "legendre":
            return self.lpoly(x)

        elif self.curvetype == "chebyshev":
            return self.chpoly(x)

    def evaluate_by1(self, pixlims=None): # evaluates curve for every pixel in range.
        if pixlims == None:
            firstpix = int(self.xrange[0] + 0.001)
            lastpix = int(self.xrange[1] + 0.001)
        else:
            firstpix = pixlims[0]
            lastpix = pixlims[1]
        pixarr = np.arange(firstpix, lastpix + 1, 1)
        return self.evalfit(pixarr)

####

def fake_multispec_data(arrlist):
   # takes a list of 1-d numpy arrays, which are
   # to be the 'bands' of a multispec, and stacks them
   # into the format expected for a multispec.  As of now
   # there can only be a single 'aperture'.

   return np.expand_dims(np.array(arrlist), 1)

############################################################################################

### START MAIN TASK.  Get the input file.
def opextract(imroot, firstlinetoplot, lastlinetoplot, plot_sample, DISPAXIS, readnoise, gain, apmedfiltlength,
              colfitorder, scattercut, colfit_endmask=10, diagnostic=False, production=False, other=None, shift=0):

    if '.fits' in imroot:
        imroot = imroot.replace(".fits", "")
    if '.fit' in imroot:
        imroot = imroot.replace(".fit", "")
        
    if other:
        apparams = aperture_params(froot=other, dispaxis=DISPAXIS)
    else:
        apparams = aperture_params(froot=imroot, dispaxis=DISPAXIS)

    if diagnostic:
        apparams.repeat_back()

    hdu = fits.open(imroot + '.fits')
    hdr = hdu[0].header
    rawdata = hdu[0].data

    # If dispersion does not run along the columns, transpose
    # the data array.  Doing this once here means we can assume
    # dispersion is along columns for the rest of the program.

    if DISPAXIS != 2:
        rawdata = rawdata.T

    # compute a variance image from the original counts, using
    # the specified read noise (in electrons) and gain
    # (in electrons per ADU).  These can be set with command
    # line arguments or (more likely) set as defaults.

    # The algorithm wants variance in data numbers, so divide
    # and square.

    readvar = (readnoise / gain) ** 2  # read noise as variance in units of ADU

    # variance image data is derived from bias-subbed data.
    varimage = readvar + rawdata / gain

    # Creating zero arrays for processed data should be much
    # faster than building them row by row, e.g. for the
    # background-subtracted data:
    subbeddata = np.zeros(rawdata.shape)

    rootpi = np.sqrt(np.pi)  # useful later.

    # Compute aperture and background limits using the
    # parameters read from the database file.

    apcent = apparams.evaluate_curve(pixlims=(0, rawdata.shape[0] - 1))

    # IRAF is one-indexed, so subtract 1 to make it zero-indexed.
    apcent -= 1

    # adding shift
    apcent = apcent + shift
    
    # four arrays give limits of background sections

    bckgrlim1 = apcent + apparams.bckgrintervals[0][0]
    bckgrlim2 = apcent + apparams.bckgrintervals[0][1]
    bckgrlim3 = apcent + apparams.bckgrintervals[1][0]
    bckgrlim4 = apcent + apparams.bckgrintervals[1][1]

    # arrays of limits for aperture
    aplimlow = apcent + apparams.aplow
    aplimhigh = apcent + apparams.aphigh
    # convert to integers for later use
    aplimlowint = np.round(aplimlow).astype(int)
    aplimhighint = np.round(aplimhigh).astype(int)

    lowestap = aplimlowint.min()
    highestap = aplimhighint.max()   # extreme ends of aperture range

    if diagnostic:
        print("lowestap ", lowestap, " highestap ", highestap)

    # Now compute and load the background spectrum by fitting
    # rows one by one.  Start with a zero array:

    ### NOTE that only Legendre fits are implemented in this version. ###

    bckgrspec = np.zeros(apcent.shape)

    # crossdisp is the grid of pixel numbers for
    # fit to background, and to form the x-array
    # for optional plots.

    crossdisp = np.array(range(rawdata.shape[1]))

    # take background fit parameters from input
    # file if they loaded right.

    try:
        niterations = apparams.bckgr_niterate
        low_rej = apparams.bckgr_low_reject
        high_rej = apparams.bckgr_high_reject
    except:
        niterations = 3
        low_rej = 2.
        high_rej = 2.

    print(niterations,low_rej,high_rej)
    # fit and subtract the background.  The region
    # fitted is on each side of the program object.
    # Only legendre fits have been tested.

    for lineindex in range(rawdata.shape[0]):

        ldata = rawdata[:][lineindex]

        # index limits for low side and high side background windows
        ind1  = int(bckgrlim1[lineindex])
        ind2  = int(bckgrlim2[lineindex])
        ind3  = int(bckgrlim3[lineindex])
        ind4  = int(bckgrlim4[lineindex])

        # grab subarrays for low and high side and join

        xlo = crossdisp[ind1:ind2]
        ylo = ldata[ind1:ind2]
        xhi = crossdisp[ind3:ind4]
        yhi = ldata[ind3:ind4]

        xtofit = np.hstack((xlo, xhi))
        ytofit = np.hstack((ylo, yhi))

        # fit and iterate to get rid of bad pixels.

        for iteration in range(niterations):

            # use legendre order from input if function is a leg

            if apparams.bckgrfunc == 'legendre':
                legcoefs = leg.legfit(xtofit, ytofit,
                   apparams.bckgrfunc_iraforder - 1)
            else:  # or default to 2nd order leg if func is something else.
                legcoefs = leg.legfit(xtofit, ytofit, 2)
            fit = leg.legval(xtofit, legcoefs)
            residuals = ytofit - fit
            stdev = np.std(residuals)
            # fancy indexing!
            keepindices = abs(residuals) < high_rej * stdev
            xtofit = xtofit[keepindices]
            ytofit = ytofit[keepindices]

        # Subtract the fit from this line, and store in subbeddta

        subbeddata[lineindex] = rawdata[lineindex] - leg.legval(crossdisp, legcoefs)

        # Keep the 1-d background spec at the center of the image.
        # later this is scaled up to the 'effective' width of the optimally
        # extracted spectrum and written out to the multispec.

        bckgrspec[lineindex] = leg.legval(apcent[lineindex], legcoefs)

    # If keeping diagnostics, write a sky-subtracted image.

    if diagnostic:
        # create a new hdu object around subbeddata
        hduout = fits.PrimaryHDU(subbeddata)
        # copy header stuff
        hdrcopy = hdr.copy(strip = True)
        hduout.header.extend(hdrcopy, strip=True, update=True,
            update_first=False, useblanks=True, bottom=False)
        # and write it.
        hduout.writeto(imroot + "_sub.fits", overwrite=True)

        print("Background-subbed image written to '%s_sub.fits'" % (imroot))

        # Now that we have hduout, write the variance image
        # simply by substituting the data and writing.

        hduout.data = varimage
        hduout.writeto(imroot + "_var.fits", overwrite=True)

        print("Variance image written to '%s_var.fits'" % (imroot))

    # PROFILE FINDING

    # Creates an image of the stellar spectrum profile
    # normalized row by row, i.e, Stetson's "P_lambda".

    # Start by median-filtering the subbed array parallel to the
    # dispersion; this will remove CRs and smooth a bit.

    smootheddata = nd.median_filter(subbeddata, size=(apmedfiltlength, 1), mode='nearest')

    if diagnostic:
        # write out median-smoothed array for diagnostic.
        hduout.data = smootheddata
        hduout.writeto(imroot + "_medfilt.fits", overwrite=True)
        print("Medium-filtered image written to '%s_medfilt.fits'" % (imroot))

    # Find the whole x-range over which we'll extract.  We'll
    # fit only those columns.

    aprange = range(lowestap, highestap+1, 1)

    # Get the range of pixels along the dispersion.
    # OSMOS data needs extreme ends masked a bit.

    pixrange = np.arange(0, smootheddata.shape[0], 1.)

    firstfitpix = colfit_endmask
    lastfitpix = smootheddata.shape[0] - colfit_endmask
    fitrange = np.arange(firstfitpix, lastfitpix, 1.)

    # This array will contain the normalized 2-d aperture.

    apdata = np.zeros(smootheddata.shape)

    # go column by column (parallel to dispersion), and fit
    # a polynomial to the smoothed spec in that column.

    # the order needs to be set high enough to follow
    # odd bumps for the echelle chip.
    # now removed to hard-coded param list
    # colfitorder = 15

    for i in aprange:

        # Diagnostics gives a nice plot of the columns and their fits, which can
        # be very enlightening.  First, the smoothed data:

        if diagnostic:
            plt.plot(pixrange, smootheddata[:, i])

        legcoefs = leg.legfit(fitrange, smootheddata[firstfitpix:lastfitpix, i], colfitorder)

        thisfit = leg.legval(pixrange, legcoefs)
        # plot fit for diagnostic.
        if diagnostic:
            plt.plot(pixrange, thisfit)
        apdata[:, i] = thisfit

    # mask values less than zero
    # this may bias spec in very low s/n situations, but
    # it saves lots of trouble.

    apdata[apdata < 0.] = 0.

    # normalize across dispersion to create Horne's profile
    # estimate 'P'.  This is redundant as the aperture is later
    # restricted to those within the pixel limits for that row,
    # but 'norm' is useful later.

    norm = np.sum(apdata, axis=1)

    # if there are no pixels, norm is zero.  Replace those with
    # ones to avoid NaNs in the divided array.

    norm = np.where(norm == 0., 1., norm)

    # show accumulated graphs, which are of the fits of aperture
    # along the dispersion together with the median-smoothed
    # spectrum.

    if diagnostic:
        plt.title("Smoothed column data and poly fits.")
        plt.xlabel("Pixel along dispersion")
        plt.ylabel("Counts (not normalized)")
        plt.show()

    # finally, normalize the aperture so sum across
    # dispersion is 1, making Horne's "P".
    # (again, this is redundant)

    nn = norm.reshape(rawdata.shape[0], 1)
    apdata = apdata / nn

    # Do something rational to normalize the
    # sky background.  Let's try this:
    # - get the width of the spectrum as a sigma (from a
    #     gaussian fit)   at a spot where it's likely to be
    #     strong -- the displine from the aperture file is
    #     likely to be good.
    # - multiply sigma by root-pi.  This is like an 'effective
    #     width' of the aperture -- it's basically how much
    #     sky effectively falls in the aperture.
    # - Use this width to renormalize the sky spec, which is
    #     per-pixel.
    # - This whole exercise may be silly, but it's reasonably
    #   cheap.

    goodapline = apdata[int(apparams.displine), aprange]

    # use the nice astropy Gaussian fit routines to fit profile.

    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude = np.max(goodapline), mean = np.median(aprange), stddev=1.)
    fittedgau = fitter(model, aprange, goodapline)
    skyscalefac = rootpi * fittedgau.stddev.value

    if diagnostic:

        # diagnostic to show fidicual profile.

        plt.plot(aprange, goodapline)
        plt.plot(aprange, fittedgau(aprange))
        plt.title("Fiducial profile (from row %5.0f)." % (apparams.displine))
        plt.show()

    #
    # Here comes the MAIN EVENT, namely extraction of the
    # spectrum and so on.
    #

    # Create 1-d arrays for the results.

    optimally_extracted = np.zeros(rawdata.shape[0])
    straight_sum = np.zeros(rawdata.shape[0])
    sigma_spec = np.zeros(rawdata.shape[0])

    # keep a summation of the non-optimally-extracted spectra
    # for later renormalization.

    cr_corrected_overall_flux = 0.

    # keep track of how many pixels are rejected.
    nrej_pixel = 0       # total number of 2d pixels affected
    corrected_specpts = 0   # total number of spectral points affected
    n_profiles_substituted = 0 # and number of profiles replaced with
               # Gaussian fits because of terrible S/N.

    # This optimal extraction process is very tricky.  For
    # development it's useful to look at the detailed calculation
    # for several rows.  I've built in the ability to
    # plot a range of rows and print out some data on them.
    # This is invoked now with command-line arguments.

    # We'll compute a statistic for how badly a pixel deviates from
    # the expected profile and ignore those that are bad enough.

    # This maximum accepted 'badness' statistic is set elsewhere now.
    # scattercut = 25.   # Horne's suggested value is 25.

    # Finally, extract line-by-line:

    for lineindex in range(rawdata.shape[0]):

        # are we plotting this line out (etc.)?
        # This logic is separate from "--diagnostic".
        showdiagn = False
        if plot_sample and lineindex >= firstlinetoplot and lineindex <= lastlinetoplot:
            showdiagn = True

        # Compute straight sum of sky-subbed data in aperture, without pixel
        # rejection.  In principle we could edit the CRs out of the straight sum but
        # that would require interpolating across them somehow.

        # Keep a copy of data in the aperture for later renormalization of the optimal
        # extraction.

        in_ap_data = subbeddata[lineindex, aplimlowint[lineindex]:aplimhighint[lineindex]]

        straight_sum[lineindex] = np.sum(in_ap_data)

        # I'm getting bad results in very poor S/N parts of the spectrum
        # where the aperture isn't well-defined even after all that smoothing.
        # Here's an attempt to fix this -- in cases where the S/N is terrible,
        # replace the aperture with the gaussian fit to the supposedly good line,
        # recentered.  Here we can re-use the gaussian profile fit computed
        # earlier for the "fiducial" line, shifted to the local aperture
        # center from aptrace.  This ignores any profile variations along the
        # dispersion, but the spec is barely detected anyway.

        # The norm array from before has preserved the cross-dispersion sums
        # pre-normalization.  Use this to decide whether to substitute the
        # fit for the empirical profile.

        if norm[lineindex] < readnoise / gain:  # basically nothing there.
            apmodel = modeling.models.Gaussian1D(amplitude =
                 rootpi/fittedgau.stddev.value, mean = apcent[lineindex],
                 stddev = fittedgau.stddev.value)
            apdata[lineindex] = apmodel(range(apdata.shape[1]))
            n_profiles_substituted = n_profiles_substituted + 1

        # 'pixind' is the array of pixel numbers to be included.
        # When pixels are rejected, the mechanism used is to delete
        # their index from pixind.

        # To start, include only pixels where the aperture is
        # at least positive and a bit.

        pixind0 = np.where(apdata[lineindex] > 0.001)
        # this weirdly returned a tuple that would not do fancy indexing:
        pixinds = np.array(pixind0)

        # Include only those pixels that are within the aperture
        # in this row.
        pixinds = pixinds[pixinds >= aplimlowint[lineindex]]
        pixinds = pixinds[pixinds <= aplimhighint[lineindex]]

        # renormalize apdata to the values within the aperture.
        # This is assumed to contain 'all the light'.

        validsum = np.sum(apdata[lineindex, pixinds])
        apdata[lineindex, pixinds] = apdata[lineindex, pixinds] / validsum

        worst_scatter = 10000. # initialize high to make the loop start.

        largest_valid_stat = 40. # this isn't used yet.
        iteration = 0

        # "while" loop to iterate CR rejection. Second
        # condtion guards against case of no valid aperture points.

        while worst_scatter > scattercut and pixinds.size > 0:

            # Horne eq'n (8):
            numerator = np.sum(apdata[lineindex, pixinds] * subbeddata[lineindex, pixinds] / varimage[lineindex, pixinds])
            denominator = np.sum(apdata[lineindex, pixinds] ** 2 / varimage[lineindex, pixinds])
            optimally_extracted[lineindex] = numerator/denominator

            # Horne eq'n (9) for variance, square-rooted to get sigma:
            sigma_spec[lineindex] = np.sqrt(1. / (np.sum(apdata[lineindex, pixinds] ** 2 / varimage[lineindex, pixinds])))

            # The procedure for eliminating cosmic rays and other discrepant profile points
            # follows; it's taken from Horne's article, page 614, top right.

            # compute Horne's measure of anomalous pixels due to CRs or whatever.

            # NOTE that an inaccurate profile estimate will lead to spurious 'bad' pixels.
            # May want to put in something to relax the rejection criterion for bright objects.

            scatter_array = ((subbeddata[lineindex, pixinds] - optimally_extracted[lineindex] * apdata[lineindex, pixinds])**2 / varimage[lineindex, pixinds])

            # array of S/Ns to assess validity of stat model - not yet used.
            sn_array = subbeddata[lineindex, pixinds] / np.sqrt(varimage[lineindex, pixinds])

            if showdiagn:   # examine the fit in this row in detail.
                print("scatter_array ", scatter_array, " shape ", scatter_array.shape)
                print("sn_array", sn_array)

            worst_scatter = np.max(scatter_array)

            if worst_scatter > scattercut:   # reject bad pixels

                # find and delete bad pixel.  This will fail if there are two equal
                # values of scatter_array, but they are floats so the chance of this
                # happening is minuscule.

                index_of_worst = np.where(scatter_array == worst_scatter)[0][0]
                pixinds = np.delete(pixinds, index_of_worst)

                if showdiagn:
                    print("worst: ", worst_scatter, "killed index ", index_of_worst)

                # Also edit out the high point from the in_ap_data so it doesn't skew the
                # later overall renormalization too badly.

                bad_point_value = subbeddata[lineindex, index_of_worst]
                in_ap_data = in_ap_data[in_ap_data != bad_point_value]

                # re-normalize the remaining aperture points.
                # *** This was an error!! ***  Just omit the point, and keep normalization.
                # validsum = np.sum(apdata[lineindex, pixinds])
                # apdata[lineindex, pixinds] = apdata[lineindex, pixinds] / validsum

                # keep track of how many pixels were rejected, and how
                # many spectral points are affected.

                nrej_pixel += 1
                if iteration == 0:
                    corrected_specpts += 1

            if len(pixinds) == 0:  # Uh-oh -- out of pixels!
                worst_scatter = 0.  # will kick us out of loop.
                optimally_extracted[lineindex] = 0.

            iteration += 1

        if len(pixinds) == 0:  # can be zero because aperture is all zero.
            optimally_extracted[lineindex] = 0.
            sigma_spec[lineindex] = 10.  # arbitrary

        # accumulate sum of flux in non-rejected straight sum points.
        cr_corrected_overall_flux += np.sum(in_ap_data)

        # plot some sample lines for diagnostic if indicated.

        if showdiagn:
            lowx = aplimlowint[lineindex]   #brevity
            highx = aplimhighint[lineindex]
            plrange = range(lowx - 15, highx + 15)
            # plot aperture profile * estimate
            plt.plot(plrange, apdata[lineindex, plrange] * optimally_extracted[lineindex])
            # and also the actual sky-subtracted data.
            plt.plot(plrange, subbeddata[lineindex, plrange])

            # also plot vertical bars at pixel limits, and dots at pixels that were used.
            plt.plot((lowx, lowx), (-10, 50))
            plt.plot((highx, highx), (-10, 50))
            pixpl = np.zeros(pixinds.shape[0])
            plt.plot(pixinds, pixpl, 'bo')
            plt.title("Line %d  optextr %8.2f " % (lineindex, optimally_extracted[lineindex]))
            plt.show()

    if diagnostic:
        # write aperture image (as amended by extraction) for a diagnostic.
        hduout.data = apdata
        hduout.writeto(imroot + "_aperture.fits", overwrite=True)
        print("Normalized aperture image written to '%s_aperture.fits'" % imroot)
        print("(These diagnostic images are purely for your dining and dancing")
        print("pleasure, and can be safely deleted.)")
        print(" ")

    # Finally, normalize the optimally extracted spec to
    # the cr-rejected straight sum.

    normfac = cr_corrected_overall_flux / np.sum(optimally_extracted)
    if diagnostic:
        print("overall flux %8.0f, sum of optimal extr. %8.0f, norm. fac %7.5f" %
            (cr_corrected_overall_flux, np.sum(optimally_extracted), normfac))
    optimally_extracted *= normfac

    # EXTRACTION IS COMPLETE!

    # WRITE OUT AS A MULTISPEC FITS FILE.

    ultimate = rawdata.shape[0] - 1      # last and second-to-last indices
    penultimate = rawdata.shape[0] - 2

    if DISPAXIS == 2:

        # For modspec data -- and presumably for other column-dispersion ---

        # Comparison with previous extractions show an off-by-one!
        # Never could figure out why, so shift output arrays by one
        # pixel with np.roll, and repeat the last pixel.
        # Could also do this with indexing I'm thinking that
        # the implementation of np.roll is likely to be faster (?).

        ultimate = rawdata.shape[0] - 1
        penultimate = rawdata.shape[0] - 2

        out1 = np.roll(optimally_extracted, -1)
        out1[ultimate] = out1[penultimate]

        out2 = np.roll(straight_sum, -1)
        out2[ultimate] = out2[penultimate]

        out3 = np.roll(bckgrspec * skyscalefac, -1)
        out3[ultimate] = out3[penultimate]

        out4 = np.roll(sigma_spec, -1)
        out4[ultimate] = out4[penultimate]

    else:  # OSMOS data (dispaxis = 1) doesn't have this issue
        # Code to fix a bad pixel at high end left in place commented
        # out in case it's needed
        out1 = optimally_extracted
        # out1[ultimate] = out1[penultimate] # fudge the last pixel.
        out2 = straight_sum
        # out2[ultimate] = out2[penultimate] # fudge the last pixel.
        out3 = bckgrspec * skyscalefac
        out4 = sigma_spec

    print(imroot, ": rejected ", nrej_pixel, " pixels, affecting ", corrected_specpts, " spectral points.")
    return out1, out2, out3, out4


##########################################################################
def dvex():
    dv = {}
    dv['line'] = {'600ZD': 300, 'Gr11': 430, 'Gr13': 200, 'GR': 150, 'GB': 430}
    dv['std'] = {'_t_order': 6, '_t_niter': 5, '_t_sample': '*', '_t_nlost': 20, '_width': 10, '_radius': 10,
                 '_weights': 'variance',
                 '_nsum': 30, '_t_step': 10, '_t_nsum': 10, '_lower': -10, '_upper': 10, '_b_sample': '-40:-20,20:40',
                 '_resize': 'no'}
    dv['obj'] = {'_t_order': 4, '_t_niter': 5, '_t_sample': '*', '_t_nlost': 20, '_width': 10, '_radius': 10,
                 '_weights': 'variance',
                 '_nsum': 40, '_t_step': 10, '_t_nsum': 10, '_lower': -5, '_upper': 5, '_b_sample': '-25:-15,15:25',
                 '_resize': 'yes'}
    dv['test'] = {'_t_order': 4, '_t_niter': 5, '_t_sample': '*', '_t_nlost': 20, '_width': 10, '_radius': 10,
                 '_weights': 'none',
                 '_nsum': 40, '_t_step': 10, '_t_nsum': 10, '_lower': -5, '_upper': 5, '_b_sample': '-25:-15,15:25',
                 '_resize': 'yes'}
    return dv

#################################################################
def repstringinfile(filein, fileout, string1, string2):
    import re
    f = open(filein, 'r')
    ss = f.readlines()
    f.close()
    f = open(fileout, 'w')
    for n in range(len(ss)):
        if string1 in ss[n]:
            f.write(re.sub(string1, string2, ss[n]))
        else:
            f.write(ss[n])
    f.close()


##################################

##############################################################################
def delete(listfile):
    import os
    import string
    import re
    import glob

    if listfile[0] == '@':
        ff = open(listfile[1:])
        files = ff.readlines()
        imglist = []
        for ff in files:
            ff = re.sub(' ', '', ff)
            if not ff == '\n' and ff[0] != '#':
                ff = re.sub('\n', '', ff)
                imglist.append(ff)
    elif ',' in listfile:
        imglist = str.split(listfile, sep=',')
    else:
        imglist = [listfile]
    lista = []
    for _file in imglist:
        lista = lista + glob.glob(_file)
    if lista:
        for _file in lista:
            try:
                os.system('rm ' + _file)
            except:
                pass


###############################################################

def extractspectrum(dictionary,img, key, _ext_trace=False, _dispersionline=False, _interactive=True, _type='obj', _force=False):
    import deimos
    import glob
    import os
    import string
    import sys
    import re
    import datetime
    import numpy as np
    from astropy.io import fits
    print(_interactive)
    print(_ext_trace)
    
    imgout = re.sub('.fits','',img) + '_' + str(key) + '_nosky.fits'

    os.environ["PYRAF_BETA_STATUS"] = "1"
    _extinctdir = 'direc$standard/extinction/'
    _extinction = 'lasilla2.txt'
    _observatory = 'lasilla'
    
    from pyraf import iraf
    iraf.set(direc=deimos.__path__[0] + '/')
    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.specred(_doprint=0)
    toforget = ['specred.apall', 'specred.transform']
    for t in toforget:
        iraf.unlearn(t)
        
    iraf.specred.dispaxi = 1
    dv = dvex()
    hdr = dictionary[img]['fits'][0].header
    _gain = 1
    _rdnoise = 1
    _grism = dictionary[img]['GRATENAM']
    
    if _force==True:
        _new = 'yes'
        _extract = 'yes'
    else:
        if not os.path.isfile('database/ap' + re.sub('.fits', '', imgout)): 
            _new = 'yes'
            _extract = 'yes'
        else:
            if _interactive in ['Yes', 'yes', 'YES', 'y', 'Y',True]:
                answ = 'x'
                while answ not in ['y', 'n','no','yes']:
                    answ = raw_input('\n### do you want to trace again [[y]/n] ? ')
                    if not answ:
                        answ = 'y'
                    if answ == 'y':
                        _new, _extract = 'yes', 'yes'
                    else:
                        _new, _extract = 'yes', 'no'
            else:
                _new, _extract = 'no', 'no'
            
    if _extract == 'yes':
        #delete(imgex)
        ############  chose where to show the profile
        if _dispersionline:
            question = 'yes'
            while question == 'yes':
                deimos.deimosutil.image_plot(directory[img]['nosky'],2)
                dist = raw_input(
                    '\n### At which line do you want to trace the spectrum [' + str(dv['line'][_grism]) + '] ? ')
                if not dist:
                    dist = 400
                try:
                    dist = int(dist)
                    question = 'no'
                except:
                    print('\n### input not valid, try again:')
        else:
            dist = dv['line'][_grism]

        ######################   
        if _ext_trace in ['yes', 'Yes', 'YES', True]:
            print('add here the use of different trace')           
        else:
            _reference = ''
            _find = 'yes'
            _recenter = 'yes'
            _resize = dv[_type]['_resize']
            _edit = 'yes'
            _trace = 'yes'            
            _fittrac = 'yes'
            iraf.specred.mode = 'q'
            _mode = 'q'
            _review = 'yes'
            iraf.specred.apall.b_niterate=3
            iraf.specred.apall.b_function='legendre'
            iraf.specred.aptrace.niterate=10
            iraf.specred.aptrace(imgout, referen=_reference, interactive=_interactive,\
                                 find=_find, recenter=_recenter, resize=_resize,\
                                 edit=_edit, trace=_trace, fittrace=_fittrac,\
                                 line=dist,step=dv[_type]['_t_step'],\
                                 nlost=dv[_type]['_t_nlost'], function='legendre',\
                                 order=dv[_type]['_t_order'], sample=dv[_type]['_t_sample'],\
                                 naverage = 1, niter=10,\
                                 #dv[_type]['_t_niter'],\
                                 low_reject=4, high_reject=4,\
                                 nsum=dv[_type]['_t_nsum'],\
                                 mode=_mode)


            ####### read iraf database and add trace in the dictionary 
            apparams = deimos.irafext.aperture_params(froot=imgout, dispaxis=1)
            # data.shape[1] is for dispersion axes 1
            # data.shape[0] is for dispersion axes 0
            peakpos = apparams.evaluate_curve(pixlims=(0, dictionary[img]['trimmed'+str(key)].shape[1] - 1))
            peakpos -= 1

            # add trace to the dictionary
            dictionary[img]['peakpos_' + str(key)] = peakpos

            #add iraf parameters to the dictionary
            meta={}
            meta['coeffs'] = apparams.coeffs
            meta['bckgrintervals'] = apparams.bckgrintervals
            meta['aplow'] = apparams.aplow
            meta['aphigh'] = apparams.aphigh
            meta['bckgr_niterate'] = dv[_type]['_t_niter'] #apparams.bckgr_niterate
            meta['bckgr_low_reject'] = 5 #apparams.bckgr_low_reject
            meta['bckgr_high_reject'] = apparams.bckgr_high_reject
            meta['bckgrfunc'] = apparams.bckgrfunc
            meta['bckgrfunc_iraforder'] = apparams.bckgrfunc_iraforder
            meta['displine'] = apparams.displine

            for key1 in meta:
                if key1 in ['aplow','displine','aphigh']:
                    dictionary[img][key1 + '_' + str(key)] = float(meta[key1])
                else:    
                    dictionary[img][key1 + '_' + str(key)] = re.sub('\[?]?','', str(meta[key1]))
                        
            # write the trace file
            _grism = dictionary[img]['GRATENAM']
            _slit = dictionary[img]['SLMSKNAM']   
            setup = (_grism,_slit)
            _dir = '_'.join(setup)
            imgtrace = re.sub('.fits','',img) + '_' + str(key) + '_trace.ascii'
            output = _dir + '/' + str(key)  + '/' + imgtrace
            deimos.deimosutil.writetrace(peakpos,meta,'trace',output)
    else:
        print('trace already done')
    return dictionary
##############################

### START MAIN TASK.  Get the input file.
def opextract_new(img, firstlinetoplot, lastlinetoplot, plot_sample, DISPAXIS, readnoise, gain, apmedfiltlength,
                  colfitorder, scattercut, colfit_endmask=10, diagnostic=False, production=False,
                  other=None, shift=0, dictionary= None, key = 3):
    # reading aperture from the database instead of the iraf file
    #
    #
    #
    imroot = re.sub('.fits','',img) + '_' + str(key) + '_nosky'
    if '.fits' in imroot:
        imroot = imroot.replace(".fits", "")
        if '.fit' in imroot:
            imroot = imroot.replace(".fit", "")

    if dictionary:
        rawdata = dictionary[img]['nosky' + str(key)]
        
        # if other is defined use trace from other
        if other:
            if other not in dictionary:
                print('error image not in the dictionary')
            else:
                # use trace from a different file
                img = other
                
        if img not in dictionary:
            sys.exit('error image not in the dictionary')
        else:
            coeffs = np.array(str.split(dictionary[img]['coeffs_' + str(key)],','),float)
            bckgrintervals = np.reshape(np.array(str.split(dictionary[img]['bckgrintervals_' + str(key)],','),float),(2,2))
            aplow = float(dictionary[img]['aplow_' + str(key)])
            aphigh = float(dictionary[img]['aphigh_' + str(key)])
            bckgr_niterate = int(dictionary[img]['bckgr_niterate_' + str(key)])
            bckgr_low_reject = dictionary[img]['bckgr_low_reject_' + str(key)]
            bckgr_high_reject = float(dictionary[img]['bckgr_high_reject_' + str(key)])
            bckgrfunc = re.sub(" ","",dictionary[img]['bckgrfunc_' + str(key)])
            bckgrfunc_iraforder = int(dictionary[img]['bckgrfunc_iraforder_' + str(key)])
            displine = float(dictionary[img]['displine_' + str(key)])
    else:
        print('read from iraf')
#        imroot = re.sub('.fits','',img) + '_' + str(key) + '_nosky'
        if dictionary:
            hdu = dictionary[img]['nosky' + str(key)]
            _out = fits.ImageHDU(data=hdu)
            fits.writeto(imroot+'.fits', _out.data,overwrite='yes')
        else:
            print('this does not work if the nosky image is not already there')
        
        hdu = fits.open(imroot + '.fits')
        hdr = hdu[0].header
        rawdata = hdu[0].data
                
        if other:
            # use trace from a different file
            apparams = aperture_params(froot=other, dispaxis=DISPAXIS)
        else:
            apparams = aperture_params(froot=imroot, dispaxis=DISPAXIS)

        if diagnostic:
            apparams.repeat_back()

        coeffs               = apparams.coeffs
        bckgrintervals       = apparams.bckgrintervals
        aplow                = apparams.aplow
        aphigh               = apparams.aphigh             
        bckgr_niterate       = apparams.bckgr_niterate     
        bckgr_low_reject     = 5 # apparams.bckgr_low_reject   
        bckgr_high_reject    = apparams.bckgr_high_reject  
        bckgrfunc            = apparams.bckgrfunc          
        bckgrfunc_iraforder  = apparams.bckgrfunc_iraforder
        displine             = apparams.displine            
        

    # If dispersion does not run along the columns, transpose
    # the data array.  Doing this once here means we can assume
    # dispersion is along columns for the rest of the program.

    if DISPAXIS != 2:
        rawdata = rawdata.T

    # compute a variance image from the original counts, using
    # the specified read noise (in electrons) and gain
    # (in electrons per ADU).  These can be set with command
    # line arguments or (more likely) set as defaults.

    # The algorithm wants variance in data numbers, so divide
    # and square.

    readvar = (readnoise / gain) ** 2  # read noise as variance in units of ADU

    # variance image data is derived from bias-subbed data.
    varimage = readvar + rawdata / gain

    # Creating zero arrays for processed data should be much
    # faster than building them row by row, e.g. for the
    # background-subtracted data:
    subbeddata = np.zeros(rawdata.shape)

    rootpi = np.sqrt(np.pi)  # useful later.

    # Compute aperture and background limits using the
    # parameters read from the database file.

    if dictionary:
        apcent = dictionary[img]['peakpos_'+str(key)] 
    else:
        apcent = apparams.evaluate_curve(pixlims=(0, rawdata.shape[0] - 1))
        # IRAF is one-indexed, so subtract 1 to make it zero-indexed.
        apcent -= 1
    
    # adding shift
    apcent = apcent + shift
    
    # four arrays give limits of background sections
    print(bckgrintervals)
    bckgrlim1 = apcent + bckgrintervals[0][0]
    bckgrlim2 = apcent + bckgrintervals[0][1]
    bckgrlim3 = apcent + bckgrintervals[1][0]
    bckgrlim4 = apcent + bckgrintervals[1][1]

    # arrays of limits for aperture
    aplimlow = apcent + aplow
    aplimhigh = apcent + aphigh
    # convert to integers for later use
    aplimlowint = np.round(aplimlow).astype(int)
    aplimhighint = np.round(aplimhigh).astype(int)

    lowestap = aplimlowint.min()
    highestap = aplimhighint.max()   # extreme ends of aperture range

    if diagnostic:
        print("lowestap ", lowestap, " highestap ", highestap)

    # Now compute and load the background spectrum by fitting
    # rows one by one.  Start with a zero array:

    ### NOTE that only Legendre fits are implemented in this version. ###

    bckgrspec = np.zeros(apcent.shape)

    # crossdisp is the grid of pixel numbers for
    # fit to background, and to form the x-array
    # for optional plots.

    crossdisp = np.array(range(rawdata.shape[1]))

    # take background fit parameters from input
    # file if they loaded right.

    try:
        niterations = bckgr_niterate
        low_rej = bckgr_low_reject
        high_rej = bckgr_high_reject
    except:
        niterations = 3
        low_rej = 2.
        high_rej = 2.

    # it seems that does not work with 0 interactions
    # and pyraf seems also to not save correctly the number of interaction
    if niterations ==0:
        niterations = 10

    print(niterations,low_rej,high_rej)
    # fit and subtract the background.  The region
    # fitted is on each side of the program object.
    # Only legendre fits have been tested.

    for lineindex in range(rawdata.shape[0]):

        ldata = rawdata[:][lineindex]

        # index limits for low side and high side background windows
        ind1  = int(bckgrlim1[lineindex])
        ind2  = int(bckgrlim2[lineindex])
        ind3  = int(bckgrlim3[lineindex])
        ind4  = int(bckgrlim4[lineindex])

        # grab subarrays for low and high side and join

        xlo = crossdisp[ind1:ind2]
        ylo = ldata[ind1:ind2]
        xhi = crossdisp[ind3:ind4]
        yhi = ldata[ind3:ind4]

        #print(xlo,ylo,xhi,yhi)
        xtofit = np.hstack((xlo, xhi))
        ytofit = np.hstack((ylo, yhi))

        # fit and iterate to get rid of bad pixels.

        for iteration in range(niterations):

            # use legendre order from input if function is a leg
            if bckgrfunc == 'legendre':
                legcoefs = leg.legfit(xtofit, ytofit,
                   bckgrfunc_iraforder - 1)
            else:  # or default to 2nd order leg if func is something else.
                legcoefs = leg.legfit(xtofit, ytofit, 2)
            fit = leg.legval(xtofit, legcoefs)
            residuals = ytofit - fit
            stdev = np.std(residuals)
            # fancy indexing!
            keepindices = abs(residuals) < high_rej * stdev
            xtofit = xtofit[keepindices]
            ytofit = ytofit[keepindices]

        # Subtract the fit from this line, and store in subbeddta

        subbeddata[lineindex] = rawdata[lineindex] - leg.legval(crossdisp, legcoefs)

        # Keep the 1-d background spec at the center of the image.
        # later this is scaled up to the 'effective' width of the optimally
        # extracted spectrum and written out to the multispec.

        bckgrspec[lineindex] = leg.legval(apcent[lineindex], legcoefs)

    # If keeping diagnostics, write a sky-subtracted image.

    if diagnostic:
        # create a new hdu object around subbeddata
        hduout = fits.PrimaryHDU(subbeddata)
        # copy header stuff
        hdrcopy = hdr.copy(strip = True)
        hduout.header.extend(hdrcopy, strip=True, update=True,
            update_first=False, useblanks=True, bottom=False)
        # and write it.
        hduout.writeto(imroot + "_sub.fits", overwrite=True)

        print("Background-subbed image written to '%s_sub.fits'" % (imroot))

        # Now that we have hduout, write the variance image
        # simply by substituting the data and writing.

        hduout.data = varimage
        hduout.writeto(imroot + "_var.fits", overwrite=True)

        print("Variance image written to '%s_var.fits'" % (imroot))

    # PROFILE FINDING

    # Creates an image of the stellar spectrum profile
    # normalized row by row, i.e, Stetson's "P_lambda".

    # Start by median-filtering the subbed array parallel to the
    # dispersion; this will remove CRs and smooth a bit.

    smootheddata = nd.median_filter(subbeddata, size=(apmedfiltlength, 1), mode='nearest')

    if diagnostic:
        # write out median-smoothed array for diagnostic.
        hduout.data = smootheddata
        hduout.writeto(imroot + "_medfilt.fits", overwrite=True)
        print("Medium-filtered image written to '%s_medfilt.fits'" % (imroot))

    # Find the whole x-range over which we'll extract.  We'll
    # fit only those columns.

    aprange = range(lowestap, highestap+1, 1)

    # Get the range of pixels along the dispersion.
    # OSMOS data needs extreme ends masked a bit.

    pixrange = np.arange(0, smootheddata.shape[0], 1.)

    firstfitpix = colfit_endmask
    lastfitpix = smootheddata.shape[0] - colfit_endmask
    fitrange = np.arange(firstfitpix, lastfitpix, 1.)

    # This array will contain the normalized 2-d aperture.

    apdata = np.zeros(smootheddata.shape)

    # go column by column (parallel to dispersion), and fit
    # a polynomial to the smoothed spec in that column.

    # the order needs to be set high enough to follow
    # odd bumps for the echelle chip.
    # now removed to hard-coded param list
    # colfitorder = 15

    for i in aprange:

        # Diagnostics gives a nice plot of the columns and their fits, which can
        # be very enlightening.  First, the smoothed data:

        if diagnostic:
            plt.plot(pixrange, smootheddata[:, i])

        legcoefs = leg.legfit(fitrange, smootheddata[firstfitpix:lastfitpix, i], colfitorder)

        thisfit = leg.legval(pixrange, legcoefs)
        # plot fit for diagnostic.
        if diagnostic:
            plt.plot(pixrange, thisfit)
        apdata[:, i] = thisfit

    # mask values less than zero
    # this may bias spec in very low s/n situations, but
    # it saves lots of trouble.

    apdata[apdata < 0.] = 0.

    # normalize across dispersion to create Horne's profile
    # estimate 'P'.  This is redundant as the aperture is later
    # restricted to those within the pixel limits for that row,
    # but 'norm' is useful later.

    norm = np.sum(apdata, axis=1)

    # if there are no pixels, norm is zero.  Replace those with
    # ones to avoid NaNs in the divided array.

    norm = np.where(norm == 0., 1., norm)

    # show accumulated graphs, which are of the fits of aperture
    # along the dispersion together with the median-smoothed
    # spectrum.

    if diagnostic:
        plt.title("Smoothed column data and poly fits.")
        plt.xlabel("Pixel along dispersion")
        plt.ylabel("Counts (not normalized)")
        plt.show()

    # finally, normalize the aperture so sum across
    # dispersion is 1, making Horne's "P".
    # (again, this is redundant)

    nn = norm.reshape(rawdata.shape[0], 1)
    apdata = apdata / nn

    # Do something rational to normalize the
    # sky background.  Let's try this:
    # - get the width of the spectrum as a sigma (from a
    #     gaussian fit)   at a spot where it's likely to be
    #     strong -- the displine from the aperture file is
    #     likely to be good.
    # - multiply sigma by root-pi.  This is like an 'effective
    #     width' of the aperture -- it's basically how much
    #     sky effectively falls in the aperture.
    # - Use this width to renormalize the sky spec, which is
    #     per-pixel.
    # - This whole exercise may be silly, but it's reasonably
    #   cheap.

    goodapline = apdata[int(displine), aprange]

    # use the nice astropy Gaussian fit routines to fit profile.

    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude = np.max(goodapline), mean = np.median(aprange), stddev=1.)
    fittedgau = fitter(model, aprange, goodapline)
    skyscalefac = rootpi * fittedgau.stddev.value

    if diagnostic:

        # diagnostic to show fidicual profile.

        plt.plot(aprange, goodapline)
        plt.plot(aprange, fittedgau(aprange))
        plt.title("Fiducial profile (from row %5.0f)." % (displine))
        plt.show()

    #
    # Here comes the MAIN EVENT, namely extraction of the
    # spectrum and so on.
    #

    # Create 1-d arrays for the results.

    optimally_extracted = np.zeros(rawdata.shape[0])
    straight_sum = np.zeros(rawdata.shape[0])
    sigma_spec = np.zeros(rawdata.shape[0])

    # keep a summation of the non-optimally-extracted spectra
    # for later renormalization.

    cr_corrected_overall_flux = 0.

    # keep track of how many pixels are rejected.
    nrej_pixel = 0       # total number of 2d pixels affected
    corrected_specpts = 0   # total number of spectral points affected
    n_profiles_substituted = 0 # and number of profiles replaced with
               # Gaussian fits because of terrible S/N.

    # This optimal extraction process is very tricky.  For
    # development it's useful to look at the detailed calculation
    # for several rows.  I've built in the ability to
    # plot a range of rows and print out some data on them.
    # This is invoked now with command-line arguments.

    # We'll compute a statistic for how badly a pixel deviates from
    # the expected profile and ignore those that are bad enough.

    # This maximum accepted 'badness' statistic is set elsewhere now.
    # scattercut = 25.   # Horne's suggested value is 25.

    # Finally, extract line-by-line:

    for lineindex in range(rawdata.shape[0]):

        # are we plotting this line out (etc.)?
        # This logic is separate from "--diagnostic".
        showdiagn = False
        if plot_sample and lineindex >= firstlinetoplot and lineindex <= lastlinetoplot:
            showdiagn = True

        # Compute straight sum of sky-subbed data in aperture, without pixel
        # rejection.  In principle we could edit the CRs out of the straight sum but
        # that would require interpolating across them somehow.

        # Keep a copy of data in the aperture for later renormalization of the optimal
        # extraction.

        in_ap_data = subbeddata[lineindex, aplimlowint[lineindex]:aplimhighint[lineindex]]

        straight_sum[lineindex] = np.sum(in_ap_data)

        # I'm getting bad results in very poor S/N parts of the spectrum
        # where the aperture isn't well-defined even after all that smoothing.
        # Here's an attempt to fix this -- in cases where the S/N is terrible,
        # replace the aperture with the gaussian fit to the supposedly good line,
        # recentered.  Here we can re-use the gaussian profile fit computed
        # earlier for the "fiducial" line, shifted to the local aperture
        # center from aptrace.  This ignores any profile variations along the
        # dispersion, but the spec is barely detected anyway.

        # The norm array from before has preserved the cross-dispersion sums
        # pre-normalization.  Use this to decide whether to substitute the
        # fit for the empirical profile.

        if norm[lineindex] < readnoise / gain:  # basically nothing there.
            apmodel = modeling.models.Gaussian1D(amplitude =
                 rootpi/fittedgau.stddev.value, mean = apcent[lineindex],
                 stddev = fittedgau.stddev.value)
            apdata[lineindex] = apmodel(range(apdata.shape[1]))
            n_profiles_substituted = n_profiles_substituted + 1

        # 'pixind' is the array of pixel numbers to be included.
        # When pixels are rejected, the mechanism used is to delete
        # their index from pixind.

        # To start, include only pixels where the aperture is
        # at least positive and a bit.

        pixind0 = np.where(apdata[lineindex] > 0.001)
        # this weirdly returned a tuple that would not do fancy indexing:
        pixinds = np.array(pixind0)

        # Include only those pixels that are within the aperture
        # in this row.
        pixinds = pixinds[pixinds >= aplimlowint[lineindex]]
        pixinds = pixinds[pixinds <= aplimhighint[lineindex]]

        # renormalize apdata to the values within the aperture.
        # This is assumed to contain 'all the light'.

        validsum = np.sum(apdata[lineindex, pixinds])
        apdata[lineindex, pixinds] = apdata[lineindex, pixinds] / validsum

        worst_scatter = 10000. # initialize high to make the loop start.

        largest_valid_stat = 40. # this isn't used yet.
        iteration = 0

        # "while" loop to iterate CR rejection. Second
        # condtion guards against case of no valid aperture points.

        while worst_scatter > scattercut and pixinds.size > 0:

            # Horne eq'n (8):
            numerator = np.sum(apdata[lineindex, pixinds] * subbeddata[lineindex, pixinds] / varimage[lineindex, pixinds])
            denominator = np.sum(apdata[lineindex, pixinds] ** 2 / varimage[lineindex, pixinds])
            optimally_extracted[lineindex] = numerator/denominator

            # Horne eq'n (9) for variance, square-rooted to get sigma:
            sigma_spec[lineindex] = np.sqrt(1. / (np.sum(apdata[lineindex, pixinds] ** 2 / varimage[lineindex, pixinds])))

            # The procedure for eliminating cosmic rays and other discrepant profile points
            # follows; it's taken from Horne's article, page 614, top right.

            # compute Horne's measure of anomalous pixels due to CRs or whatever.

            # NOTE that an inaccurate profile estimate will lead to spurious 'bad' pixels.
            # May want to put in something to relax the rejection criterion for bright objects.

            scatter_array = ((subbeddata[lineindex, pixinds] - optimally_extracted[lineindex] * apdata[lineindex, pixinds])**2 / varimage[lineindex, pixinds])

            # array of S/Ns to assess validity of stat model - not yet used.
            sn_array = subbeddata[lineindex, pixinds] / np.sqrt(varimage[lineindex, pixinds])

            if showdiagn:   # examine the fit in this row in detail.
                print("scatter_array ", scatter_array, " shape ", scatter_array.shape)
                print("sn_array", sn_array)

            worst_scatter = np.max(scatter_array)

            if worst_scatter > scattercut:   # reject bad pixels

                # find and delete bad pixel.  This will fail if there are two equal
                # values of scatter_array, but they are floats so the chance of this
                # happening is minuscule.

                index_of_worst = np.where(scatter_array == worst_scatter)[0][0]
                pixinds = np.delete(pixinds, index_of_worst)

                if showdiagn:
                    print("worst: ", worst_scatter, "killed index ", index_of_worst)

                # Also edit out the high point from the in_ap_data so it doesn't skew the
                # later overall renormalization too badly.

                bad_point_value = subbeddata[lineindex, index_of_worst]
                in_ap_data = in_ap_data[in_ap_data != bad_point_value]

                # re-normalize the remaining aperture points.
                # *** This was an error!! ***  Just omit the point, and keep normalization.
                # validsum = np.sum(apdata[lineindex, pixinds])
                # apdata[lineindex, pixinds] = apdata[lineindex, pixinds] / validsum

                # keep track of how many pixels were rejected, and how
                # many spectral points are affected.

                nrej_pixel += 1
                if iteration == 0:
                    corrected_specpts += 1

            if len(pixinds) == 0:  # Uh-oh -- out of pixels!
                worst_scatter = 0.  # will kick us out of loop.
                optimally_extracted[lineindex] = 0.

            iteration += 1

        if len(pixinds) == 0:  # can be zero because aperture is all zero.
            optimally_extracted[lineindex] = 0.
            sigma_spec[lineindex] = 10.  # arbitrary

        # accumulate sum of flux in non-rejected straight sum points.
        cr_corrected_overall_flux += np.sum(in_ap_data)

        # plot some sample lines for diagnostic if indicated.

        if showdiagn:
            lowx = aplimlowint[lineindex]   #brevity
            highx = aplimhighint[lineindex]
            plrange = range(lowx - 15, highx + 15)
            # plot aperture profile * estimate
            plt.plot(plrange, apdata[lineindex, plrange] * optimally_extracted[lineindex])
            # and also the actual sky-subtracted data.
            plt.plot(plrange, subbeddata[lineindex, plrange])

            # also plot vertical bars at pixel limits, and dots at pixels that were used.
            plt.plot((lowx, lowx), (-10, 50))
            plt.plot((highx, highx), (-10, 50))
            pixpl = np.zeros(pixinds.shape[0])
            plt.plot(pixinds, pixpl, 'bo')
            plt.title("Line %d  optextr %8.2f " % (lineindex, optimally_extracted[lineindex]))
            plt.show()

    if diagnostic:
        # write aperture image (as amended by extraction) for a diagnostic.
        hduout.data = apdata
        hduout.writeto(imroot + "_aperture.fits", overwrite=True)
        print("Normalized aperture image written to '%s_aperture.fits'" % imroot)
        print("(These diagnostic images are purely for your dining and dancing")
        print("pleasure, and can be safely deleted.)")
        print(" ")

    # Finally, normalize the optimally extracted spec to
    # the cr-rejected straight sum.

    normfac = cr_corrected_overall_flux / np.sum(optimally_extracted)
    if diagnostic:
        print("overall flux %8.0f, sum of optimal extr. %8.0f, norm. fac %7.5f" %
            (cr_corrected_overall_flux, np.sum(optimally_extracted), normfac))
    optimally_extracted *= normfac

    # EXTRACTION IS COMPLETE!

    ultimate = rawdata.shape[0] - 1      # last and second-to-last indices
    penultimate = rawdata.shape[0] - 2

    if DISPAXIS == 2:

        # For modspec data -- and presumably for other column-dispersion ---

        # Comparison with previous extractions show an off-by-one!
        # Never could figure out why, so shift output arrays by one
        # pixel with np.roll, and repeat the last pixel.
        # Could also do this with indexing I'm thinking that
        # the implementation of np.roll is likely to be faster (?).

        ultimate = rawdata.shape[0] - 1
        penultimate = rawdata.shape[0] - 2

        out1 = np.roll(optimally_extracted, -1)
        out1[ultimate] = out1[penultimate]

        out2 = np.roll(straight_sum, -1)
        out2[ultimate] = out2[penultimate]

        out3 = np.roll(bckgrspec * skyscalefac, -1)
        out3[ultimate] = out3[penultimate]

        out4 = np.roll(sigma_spec, -1)
        out4[ultimate] = out4[penultimate]

    else:  # OSMOS data (dispaxis = 1) doesn't have this issue
        # Code to fix a bad pixel at high end left in place commented
        # out in case it's needed
        out1 = optimally_extracted
        # out1[ultimate] = out1[penultimate] # fudge the last pixel.
        out2 = straight_sum
        # out2[ultimate] = out2[penultimate] # fudge the last pixel.
        out3 = bckgrspec * skyscalefac
        out4 = sigma_spec

    print(imroot, ": rejected ", nrej_pixel, " pixels, affecting ", corrected_specpts, " spectral points.")
    return out1, out2, out3, out4

