import astropy
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import math
from astroquery.mast import Observations, Mast, MastMissions
from astroquery.simbad import Simbad
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky

k2_table = "/home/gleb/Astronomy/K2_planet_list.csv"
kepler_table = "/home/gleb/Astronomy/Kepler_planet_list.csv"
k2_neg = 'mast_download/k2_neg'
k2_pos = 'mast_download/k2_pos'
kep_neg = 'mast_download/kep_neg'
kep_pos = 'mast_download/kep_pos'

k2 = pd.read_csv(k2_table)
kepler = pd.read_csv(kepler_table)

defFlag_mask = k2["default_flag"] == 1
dispTrue_mask = k2["disposition"] == "CONFIRMED"
dispFalse_mask = k2["disposition"] == "FALSE POSITIVE"
koiDisp_True_mask = kepler["koi_disposition"] == "CONFIRMED"
koiDisp_False_mask = kepler["koi_disposition"] == "FALSE POSITIVE"

k2_targets_positive = k2[defFlag_mask][dispTrue_mask]
k2_targets_negative = k2[defFlag_mask][dispFalse_mask]
kepler_targets_positive = kepler[koiDisp_True_mask]
kepler_targets_negative = kepler[koiDisp_False_mask]

print(f"K2 negative target count: {len(k2_targets_negative)}.")
print(f"K2 positive target count: {len(k2_targets_positive)}.")
print(f"Kepler negative target count: {len(kepler_targets_negative)}.")
print(f"Kepler positive target count: {len(kepler_targets_positive)}.")

allspec = Observations.query_criteria(
    dataproduct_type="spectrum", provenance_name="apogee", target_classification="STAR"
)

allspec_coords = [[allspec["s_ra"]], [allspec["s_dec"]]]

k2_neg_coords = [[k2_targets_negative["ra"]], [k2_targets_negative["dec"]]]
k2_pos_coords = [[k2_targets_positive["ra"]], [k2_targets_positive["dec"]]]
kepler_neg_coords = [[kepler_targets_negative["ra"]], [kepler_targets_negative["dec"]]]
kepler_pos_coords = [[kepler_targets_positive["ra"]], [kepler_targets_positive["dec"]]]


<<<<<<< HEAD
def coordinate_crossmatch(targets, catalog):
    catalog_ra = catalog[0] * u.degree
    catalog_dec = catalog[1] * u.degree
    target_ra = target[0] * u.degree
    target_dec = target[1] * u.degree

    target_coords = SkyCoord(target_ra, target_dec)
    catalog_coords = SkyCoord(catalog_ra, catalog_dec)

    id_target, id_catalog, sep, _ = search_around_sky(
        target_coords[0], catalog_coords[0], 2 * u.arcsec
=======
def coordinate_crossmatch(target_ra, target_dec, catalog_ra, catalog_dec):
    catalog_ra = catalog_ra * u.degree
    catalog_dec = catalog_dec * u.degree
    target_ra = target_ra * u.degree
    target_dec = target_dec * u.degree

    target = SkyCoord(target_ra, target_dec)
    catalog = SkyCoord(catalog_ra, catalog_dec)

    id_target, id_catalog, sep, _ = search_around_sky(
        target[0], catalog[0], 2 * u.arcsec
>>>>>>> origin
    )

    print(f"Found {len(id_target)} matches.")

    return id_target, id_catalog


def obsid_query(id_catalog, catalog):
    obsid_catalog = []
    for i in range(len(id_catalog)):
        match_index = id_catalog[i]
        obsid = catalog["obsid"][match_index - 1 : match_index]
        obsid_catalog.append(obsid[0])

    print(f"Found {len(obsid_catalog)} OBSIDs.")

    return obsid_catalog


def download_spectra(obsid_list, download_path, mrp, curl, flat):
    mrp_only = mrp
    curl_flag = curl
    print(f'Got {len(obsid_list)} spectra for download. Path: {download_path}. Starting...')
    products = Observations.get_product_list(obsid_list)
    products = Observations.filter_products(
        products, mrp_only=mrp_only, dataproduct_type="spectrum"
    )
    manifest = Observations.download_products(
        products,
        mrp_only=mrp_only,
        curl_flag=curl,
        download_dir=download_path,
        flat=flat,
    )
    print(manifest)

<<<<<<< HEAD
id_target, id_catalog = coordinate_crossmatch(k2_pos_coords, allspec_coords)
obsid = obsid_query(id_catalog, allspec)
download_spectra(obsid, k2_pos, True, False, False)
=======
id_target, id_catalog = coordinate_crossmatch(k2_pos_coords[0], k2_pos_coords[1], allspec_coords[0], allspec_coords[1])
obsid = obsid_query(id_catalog, allspec)
#download_spectra(obsid, k2_pos, True, False, True)
>>>>>>> origin

