The DEIMOS pipeline is a python script that can run with python 2.7 and python 3
It should be run from inside a directory with Deimos data.
At the moment only run with the following setup.

MULTISLIT  slit 1 arcsec


Step of the pipeline and output files

- Trim and rotate       file_ext_trimmed.fits
- Flat  Done            no output file    
- sky curvature         lambdas_2_ext.dat,lambdas_3_ext.dat
- sky subtraction       file_ext_nosky.fits
- trace                 file_ext_trace.ascii
- extraction            file_ext_ex_obj.ascii
- wave                  file_ext_wave_obj.ascii
- response             file_ext_response_obj.ascii
- flux                 file_ext_flux_obj.ascii
