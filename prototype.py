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
print(f"Kepler positive target count: {len(kepler_targets_negative)}.")

allspec = Observations.query_criteria(
    dataproduct_type="spectrum", provenance_name="apogee", target_classification="STAR"
)

allspec_coords = [[allspec["s_ra"]], [allspec["s_dec"]]]

k2_neg_coords = [[k2_targets_negative["ra"]], [k2_targets_negative["dec"]]]
k2_pos_coords = [[k2_targets_positive["ra"]], [k2_targets_positive["dec"]]]
kepler_neg_coords = [[kepler_targets_negative["ra"]], [kepler_targets_negative["dec"]]]
kepler_pos_coords = [[kepler_targets_positive["ra"]], [kepler_targets_positive["dec"]]]


def coordinate_crossmatch(target_ra, target_dec, catalog_ra, catalog_dec):
    self.catalog_ra = catalog_ra * u.degree
    self.catalog_dec = catalog_dec * u.degree
    self.target_ra = target_ra * u.degree
    self.target_dec = target_dec * u.degree

    target = SkyCoord(target_ra, target_dec)
    catalog = SkyCoord(catalog_ra, catalog_dec)

    id_target, id_catalog, sep, _ = search_around_sky(
        target[0], catalog[0], 2 * u.arcsec
    )

    print(f"Found {len(idx_target)} matches.")

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
    self.obsid_list = obsid_list
    self.download_path = download_path
    mrp_only = mrp
    curl_flag = curl

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
