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
from secret import *

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


def coordinate_crossmatch(target_list, catalog_list):
    catalog_ra = catalog_list[0] * u.degree
    catalog_dec = catalog_list[1] * u.degree
    target_ra = target_list[0] * u.degree
    target_dec = target_list[1] * u.degree

    target = SkyCoord(target_ra, target_dec)
    catalog = SkyCoord(catalog_ra, catalog_dec)

    id_target, id_catalog, sep, _ = search_around_sky(
        target[0], catalog[0], 2 * u.arcsec
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

k2_p_idx, k2_pl = coordinate_crossmatch(k2_pos_coords, allspec_coords)
k2_n_idx, k2_nl = coordinate_crossmatch(k2_neg_coords, allspec_coords)
kep_p_idx, kep_pl = coordinate_crossmatch(kepler_pos_coords, allspec_coords)
kep_n_idx, kep_nl = coordinate_crossmatch(kepler_neg_coords, allspec_coords)

k2_pl_obsid = obsid_query(k2_pl, allspec)
k2_nl_obsid = obsid_query(k2_nl, allspec)
kep_pl_obsid = obsid_query(kep_pl, allspec)
kep_nl_obsid = obsid_query(kep_nl, allspec)

download_spectra(kep_pl_obsid, kep_pos, True, False, False)
download_spectra(kep_nl_obsid, kep_neg, True, False, False)
download_spectra(k2_pl_obsid, k2_pos, True, False, False)
download_spectra(k2_nl_obsid, k2_neg, True, False, False)



