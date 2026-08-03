import astropy
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np


def load_fits(filepath, filetype):
    data = fits.open(filepath)
    # unpacking flux vals for different spectra type:

    match filetype:

        case 'ap':
            flux_raw = data[1].data[0]

            wave = np.logspace(
                data[1].header["CRVAL1"],
                data[1].header["CRVAL1"]
                + data[1].header["CDELT1"] * data[0].header["NWAVE"],
                data[0].header["NWAVE"],
            )

            pixel_mask = data[2].data < 0.1
            #wave_raw = wave_raw[pixel_mask]

        case 'aspcap':
            flux_raw = data[1].data

            wave = np.logspace(
                data[1].header["CRVAL1"],
                data[1].header["CRVAL1"]
                + data[1].header["CDELT1"] * data[3].header["NAXIS"],
                data[1].header["NAXIS"],
            )

            pixel_mask = data[2].data < 0.1

            flux_raw = flux_raw[pixel_mask]

    

    return flux_raw, wave

def normalize_spectrum(flux_raw):
    min = np.min(flux_raw)
    max = np.max(flux_raw)
    diff = max - min

    flux_norm = []

    for i in flux_raw:
        flux = (i-min)/diff
        flux_norm.append(flux)

    return flux_norm

        
