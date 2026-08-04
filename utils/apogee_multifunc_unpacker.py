import astropy
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_splrep


def load_fits(filepath, filetype):
    data = fits.open(filepath)
    # unpacking flux vals for different spectra type:
    match filetype:

        case "ap":
            flux_raw = data[1].data[0]
            wave = np.logspace(
                data[1].header["CRVAL1"],
                data[1].header["CRVAL1"]
                + data[1].header["CDELT1"] * data[0].header["NWAVE"],
                data[0].header["NWAVE"],
            )
            err = data[2].data[0]
            mask = err < 5
            flux_raw = flux_raw[mask]
            wave = wave[mask]
            bitmask = data[3].data[0]
            bitmask = np.ndarray.tolist(bitmask)
            bad_pixels = []
            for i in bitmask:
                if bin(i) != bin(0):
                    bad_pixels.append(bitmask.index(i))
            flux_raw = np.delete(flux_raw, bad_pixels)
            wave = np.delete(wave, bad_pixels)

        case "aspcap":
            flux_raw = data[1].data
            wave = np.logspace(
                data[1].header["CRVAL1"],
                data[1].header["CRVAL1"]
                + data[1].header["CDELT1"] * data[3].header["NAXIS1"],
                data[1].header["NAXIS1"],
            )
            err = data[2].data < 0.3
            flux_raw = flux_raw[err]
            wave = wave[err]

    return flux_raw, wave


def normalize_spectrum(wave, flux_raw, high_percentile=95, s_factor=1e-4):
    mask = np.zeros_like(flux_raw, dtype=bool)
    n_bins = max(1000, len(flux_raw) // 300)
    bins = np.linspace(wave.min(), wave.max(), n_bins + 1)
    for i in range(n_bins):
        bin_mask = (wave >= bins[i]) & (wave < bins[i + 1])
        bin_flux = flux_raw[bin_mask]
        if len(bin_flux) == 0:
            continue
        threshold = np.percentile(bin_flux, high_percentile)
        mask[bin_mask] |= flux_raw[bin_mask] > threshold
    if np.sum(mask) < 10:
        raise ValueError("too few points to fit")
    s = s_factor * (wave.max() - wave.min()) * len(wave)
    spline = make_splrep(wave[mask], flux_raw[mask], s=s)
    continuum = spline(wave)
    continuum = np.clip(continuum, 1e-4, None)
    flux_norm = flux_raw / continuum
    return flux_norm


def plot_spectrum(wave, flux, show):
    plt.figure(figsize=(23, 6))
    plt.xlim(np.min(wave) - 50, np.max(wave) + 50)
    # plt.ylim(0.4, 1.3)
    plt.plot(wave, flux)

    if show:
        plt.show()
