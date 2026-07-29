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

k2_table = '/home/gleb/Astronomy/K2_planet_list.csv'
kepler_table = '/home/gleb/Astronomy/Kepler_planet_list.csv'

k2 = pd.read_csv(k2_table)
kepler = pd.read_csv(kepler_table)
defFlag_mask = k2['default_flag']==1
dispTrue_mask = k2['disposition']=='CONFIRMED'
dispFalse_mask = k2['disposition']=='FALSE POSITIVE'
koiDisp_True_mask = kepler['koi_disposition']=='CONFIRMED'
koiDisp_False_mask = kepler['koi_disposition']=='FALSE POSITIVE'


allspec = Observations.query_criteria(dataproduct_type = 'spectrum', provenance_name = 'apogee', target_classification = 'STAR')
k2_targets_positive = k2[defFlag_mask][dispTrue_mask]
k2_targets_negative = k2[defFlag_mask][dispFalse_mask]

allspec_coords = [[],[]]
allspec_coords = [[allspec['s_ra']],[allspec['s_dec']]]

k2_neg_coords = [[k2_targets_negative['ra']],[k2_targets_negative['dec']]]

ra_catalog = allspec_coords[0]*u.degree
dec_catalog = allspec_coords[1]*u.degree
ra_target = k2_neg_coords[0]*u.degree
dec_target = k2_neg_coords[1]*u.degree

target = SkyCoord(ra_target, dec_target)
catalog = SkyCoord(ra_catalog, dec_catalog)

idx_target, idx_obs, sep, _ = search_around_sky(
    target[0],
    catalog[0],
    2*u.arcsec
)

print(f"found {len(idx_target)} matches:")
for i in range(len(idx_target)):
    print(idx_target[i], ' : ', idx_obs[i])

k2_negative_obsid = [[],[]]
for i in range(len(idx_target)):
    match_index_target = idx_target[i]
    match_index_catalog = idx_obs[i]
    obj = k2_targets_negative['hostname'][match_index_target-1:match_index_target]
    obsid = allspec['obsid'][match_index_catalog-1:match_index_catalog]
    print(f"Object: {obj}. Observation: {obsid[0]}")
    k2_negative_obsid[0].append(obj)
    k2_negative_obsid[1].append(obsid[0])

for j in range(len(k2_negative_obsid[0])):
    print(f"Object: {k2_negative_obsid[0][j]}, MAST OBSID: {k2_negative_obsid[1][j]}")

target_obsids = []
for i in k2_negative_obsid[1]:
    target_obsids.append(i)

products = Observations.get_product_list(target_obsids)
products = Observations.filter_products(products, mrp_only=True, dataproduct_type='spectrum')
manifest = Observations.download_products(products, mrp_only=True, curl_flag=True, download_dir='mast_download')
print(manifest)




