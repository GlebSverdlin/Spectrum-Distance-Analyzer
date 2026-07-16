import astropy
from astropy.io import fits
from astropy.table import Table

flux_req = {'Ha' : 6562.8, 'O3_1' : 3726.1, 'O3_2' : 3728.8, 'N2' : 5755}

def get_flux(filename):
    filename='/home/gleb/Astronomy/sdss/spec-1618-53116-0587.fits'
    data = Table.read(filename, format='fits',hdu=1) #important - hdu number depends on where actual spectrum is
    flux = data['flux']
    wave = data['loglam']
    wave=10**wave
    return wave, flux

def get_flux_sel(wave, flux, flux_req):
    flux_sel = []
    for i in wave:

    
