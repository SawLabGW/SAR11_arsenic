#!/usr/bin/env python

__author__ = "Jimmy Saw"

"""
usage example:

cd ~/cgrb_workspace/arsenic/SAR11

osu_plot_TARA_arsenate_reductases_map.py \
    -p moll \
    -c ../../TARAOCEANS/OM.CompanionTables.txt \
    -x sorted_SAR11_LMWP_ArsC_B109_abundances.txt \
    -e LMWP_ArsC \
    -n ../../TARAOCEANS/OM.CompanionTables.W8.txt \
    -z yes -d tara \
    -r l -i ../WOA/woa13_all_p00mn01.csv -o test2

"""

import sys
import argparse
import operator
import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cmocean
import seaborn as sns
from operator import itemgetter
#from mpl_toolkits.basemap import Basemap
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from tqdm import tqdm

## hack to solve Basemap/conda install problem on newer Mac OS
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap


def get_coords(comp_table, count_table, enz):
    """
    Get GPS coordinates of TARA samples
    :param comp_table:
    :param count_table:
    :param enz:
    :return:
    """
    df = pd.read_csv(comp_table, sep="\t", header=0)
    lons = df['Longitude [degrees East]']
    lats = df['Latitude [degrees North]']

    samples_dict = {}

    for a, b, c, d in zip(df['Sample label [TARA_station#_environmental-feature_size-fraction]'],
                          df['Longitude [degrees East]'], df['Latitude [degrees North]'], df['Sampling depth [m]']):
        samples_dict[a] = (str(b), str(c), d)

    coords = []
    for a, b in zip(lons, lats):
        coords.append(','.join([str(a), str(b)]))

    unique_coords = list(set(coords))

    coor_dicts = {}
    for u in unique_coords:
        coor_dicts[u] = [0, 0, ""] #gene abundance, enzyme abundance, biome

    cdf = pd.read_csv(count_table, sep="\t", header=0)

    for a, b, c, d, e in zip(cdf['bins'], cdf['biome'], cdf['depth'], cdf['genes'], cdf[enz]):
        if a in samples_dict:
            coords = ','.join([samples_dict[a][0], samples_dict[a][1]])
            if coords in unique_coords:
                coor_dicts[coords][0] += d
                coor_dicts[coords][1] += e
                coor_dicts[coords][2] = b
        else:
            print a, "not in samples_dict"

    #print "len(coor_dicts) =", len(coor_dicts) #there are 141 unique lat, lon coordinate pairs
    return coor_dicts

def get_nearest_phosphate_conc(x, y, lons, lats, phos):
    """
    Gather phosphate concentrations from nearest GPS coordinates
    :param x:
    :param y:
    :param lons:
    :param lats:
    :param phos:
    :return:
    """
    dists = []
    concs = []

    for i, j, k in zip(lons, lats, phos):
        dist = np.sqrt(np.absolute(i-x)**2 + np.absolute(j-y)**2) #just Pythagorean theorem
        if np.logical_not(np.isnan(k)): #only consider [P] with known values
            dists.append(dist)
            concs.append(k)

    mindex = min(enumerate(dists), key=itemgetter(1))[0]

    return concs[mindex] #return [P] from location that's nearest to the given x, y

def get_global_phosphate(csv, check_corr, outprefix):
    """
    Parse NOAA global phosphate data to get concentrations
    :param csv:
    :param check_corr:
    :param outprefix:
    :return:
    """

    ## parse NOAA data
    df = pd.read_csv(csv, sep=",", header=1)
    lat = df['#COMMA SEPARATED LATITUDE']
    lon = df[' LONGITUDE']
    pho = df[' AND VALUES AT DEPTHS (M):0']
    gridx = np.arange(-179.5, 180.5, 1) #1 degree grid
    gridy = np.arange(-89.5, 90.5, 1) #1 degree grid

    pairs = list(itertools.product(gridx, gridy))

    pdict = {}

    for p in pairs:
        pdict[p] = np.nan

    for a, b, c in zip(lon, lat, pho):
        s = tuple([a, b])
        if s in pdict:
            pdict[s] = c

    sorted_d = sorted(pdict.items(), key=operator.itemgetter(0))
    lons = []
    lats = []
    phos = []
    for i in sorted_d:
        lons.append(i[0][0])
        lats.append(i[0][1])
        phos.append(i[1])

    return (lons, lats, phos)

def func(x, a, b, c):
    """
    Exponential decay function
    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    """
    return a * np.exp(-b * x) + c

def check_arsc_phos_corr(ars_data, global_phos, tara_phos, d, outprefix):
    """
    Perform LMWP_ArsC % abundance vs. [Phosphate] correlations
    :param ars_data:
    :param global_phos:
    :param tara_phos:
    :param d:
    :param outprefix:
    :return:
    """
    ## parse global phos data
    glons = global_phos[0]
    glats = global_phos[1]
    gphos = global_phos[2]

    ## parse tara_phos data
    df = pd.read_csv(tara_phos, sep="\t", header=0)
    tlats = []
    tlons = []
    tphos = []
    for i, j, k in zip(df['Mean_Lat*'], df['Mean_Long*'], df['PO4 [umol/L]**']):
        if k <= 0.02: #below detection limit (they should be considered practically zero)
            tlats.append(i)
            tlons.append(j)
            #tphos.append(k)
            tphos.append(0.02) #remove zeros and assign them to detection limit
            #tphos.append(0.02*1000) #in nanomols
        else:
            tlats.append(i)
            tlons.append(j)
            tphos.append(k)
            #tphos.append(k*1000) #in nanomols

        ## assign the concentrations as they are found (those at detection limit are not assigned zero)
        #tlats.append(i)
        #tlons.append(j)
        #tphos.append(k)

    ## parse LMWP_ArsC abundance data
    los = []
    las = []
    pcts = []

    for k, v in ars_data.iteritems():
        lo = float(k.split(",")[0])
        la = float(k.split(",")[1])
        if v[0] != 0:
            pcts.append((float(v[1]) / v[0]) * 100)
        else:
            pcts.append(0)
        los.append(lo)
        las.append(la)

    print "length of pcts:", len(pcts)
    print "Checking Pearsonr for % abundance of LMWP_ArsC vs. [P]..."

    pconcs = []
    if d == "tara":
        print "Looking for [P] from nearest coordinates in TARA dataset..."
        for a, b, c in tqdm(zip(los, las, pcts)):
            pconc = get_nearest_phosphate_conc(a, b, tlons, tlats, tphos)
            pconcs.append((c, pconc))  # list of tuples containing % abundance of ArsC and [P]
    elif d == "global":
        print "Looking for [P] from nearest coordinates in global NOAA dataset..."
        for a, b, c in tqdm(zip(los, las, pcts)):
            pconc = get_nearest_phosphate_conc(a, b, glons, glats, gphos)
            pconcs.append((c, pconc))  # list of tuples containing % abundance of ArsC and [P]
    print "Done checking distances"

    arsc = []
    phos = []
    print "[P]" + "\t" + "ArsC(%)"
    for i in pconcs:
        if np.isfinite(i[1]):
            print str(i[1]) + "\t" + str(i[0])
            if i[0] == 0:
                arsc.append(i[0])
            else:
                arsc.append(i[0])
                #transformed = (2/np.pi) * np.arcsin(np.sqrt(i[0]/100.0))
                #arsc.append(transformed)
            if i[1] == 0:
                phos.append(i[1])
            else:
                phos.append(i[1])
                #phos.append(np.log10(i[1]))


    pr = pearsonr(arsc, phos)
    sr = spearmanr(arsc, phos)
    print "Pearson correlation between ArsC rel. abundance and [P]  =", pr
    print "Spearman correlation between ArsC rel. abundance and [P] =", sr

    abd = pd.Series(arsc)
    pc = pd.Series(phos)
    newdf = pd.DataFrame({'arsc': abd, 'phos': pc})

    sns.set(style="ticks", color_codes=True)

    #fig, axes = plt.subplots(ncols=2, figsize=(10,5), sharey=True)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharey=False)

    sns.regplot(x="phos", y="arsc",
                line_kws={'color': '#000000', 'lw': 2,
                          'label': r'Spearman R = ' + '{0:.2f}'.format(sr[0]) + '\n' +
                                   'P = ' + '{0:.2e}'.format(sr[1])},
                data=newdf, ax=axes[0], scatter_kws={"s": 10, "color": "#228B22"})

    #sns.regplot(x="phos", y="arsc",
    #            line_kws={'color': '#000000', 'lw': 2,
    #                      'label': r'Pearson R = ' + '{0:.2f}'.format(pr[0]) + '\n' +
    #                               'P = ' + '{0:.2e}'.format(pr[1])},
    #            data=newdf, ax=axes[0], scatter_kws={"s": 10, "color": "#228B22"})


    axes[0].legend()
    axes[0].set_title('Linear regression')
    #axes[0].set_title('Pearson Correlation')
    #axes[0].set_xlabel(r'log10([P] $\mu mol/L$)')
    axes[0].set_xlabel(r'[P] ($\mu mol/L$)')
    #axes[0].set_ylabel('normalized LMWP_ArsC abundances (%)')
    axes[0].set_ylabel('LMWP_ArsC abundances (% of total SCG)')

    #model1 = lambda x, A, l, c: A * np.exp(-l * x) + c
    model1 = lambda x, A, l, c: A * np.exp(-l * x)

    popt, pcov = curve_fit(model1, newdf["phos"].values, newdf["arsc"].values)
    residuals = newdf["arsc"].values - model1(newdf["phos"].values, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((newdf["arsc"].values - np.mean(newdf["arsc"].values))**2)
    rsq = 1 - (ss_res / ss_tot)
    print "Model fit R2 =", rsq

    x = np.linspace(newdf["phos"].values.min(), newdf["phos"].values.max(), 250)

    axes[1].set_ylabel('LMWP_ArsC abundances (% of total SCG)')
    axes[1].scatter(newdf["phos"].values, newdf["arsc"].values, alpha=0.6, c='#7171C6', s=10)
    #fiteq = r'$fit: y = A e^{-\lambda x} + c$'
    fiteq = r'$fit: y = A e^{-\lambda x}$'
    axes[1].plot(x, model1(x, *popt), label=fiteq, c='#A52A2A')
    axes[1].set_title('exponential decay model fit')
    #axes[1].set_xlabel(r'log10([P] $\mu mol/L$)')
    axes[1].set_xlabel(r'[P] ($\mu mol/L$)')

    plt.subplots_adjust(top=0.90)
    #fig.suptitle('% abundance of SAR11 LMWP_ArsC vs. [P]', size=10)
    fig.suptitle('SAR11 LMWP_ArsC abundance vs. [P]', size=10)
    plt.legend()
    plt.savefig(outprefix + "_corr.pdf", format='pdf', dpi=1000)  # save the correlation plot but don't show
    #plt.show()

def draw_map(proj, coords, enz, out, res, pdata):
    """
    Plots the map with distribution of enzyme abundances (shown as relative percentages of total genes found)
    :param proj:
    :param bg:
    :param coords:
    :return:
    """
    projections = {'kav7': 'kav7', 'gall': 'gall', 'eck4':'eck4', 'robin': 'robin', 'mbtfpq': 'mbtfpq',
                   'cea': 'cea', 'moll': 'moll', 'ortho': 'ortho', 'merc': 'merc'}
    if proj in projections:
        m = Basemap(projection=proj, lon_0=0, lat_0=0, resolution=res) #choose c, l, i, h, or f (only feasible up to i)

    lons = []
    lats = []
    pcts = []
    biom = []

    for k, v in coords.iteritems():
        lon = float(k.split(",")[0])
        lat = float(k.split(",")[1])
        if v[0] != 0:
            pcts.append((float(v[1]) / v[0]) * 100)
        else:
            pcts.append(0)
        lons.append(lon)
        lats.append(lat)
        biom.append(v[2])

    fig = plt.figure(figsize=(12, 6))

    factor = 100

    if max(pcts) < 0.1:
        factor = 1000
    if max(pcts) > 5:
        factor = 10

    max_idx, max_val = max(enumerate(pcts), key=operator.itemgetter(1))

    ## plot phosphate data

    ## reshape arrays into grids
    array_lons = np.array(pdata[0])
    array_lats = np.array(pdata[1])
    array_phos = np.array(pdata[2])
    rs_arr_lons = array_lons.reshape(360, 180)
    rs_arr_lats = array_lats.reshape(360, 180)
    rs_arr_phos = array_phos.reshape(360, 180)

    x, y = m(rs_arr_lons, rs_arr_lats)
    p = rs_arr_phos

    ## draw coast lines
    m.drawcoastlines(linewidth=0.5, color='#9E9E9E', antialiased=1)

    ## draw land mask
    m.drawlsmask(land_color='#CCCCCC', ocean_color='#E0E0E0')

    ## draw pcolormesh
    m.pcolormesh(x, y, p, cmap=cmocean.cm.ice_r, rasterized=False, edgecolor='0.6', linewidth=0, zorder=1)

    ## cbar
    cbar = m.colorbar()
    cbar.set_label(r'$[phosphate] (\mu mol/L)$')

    ## plot scatter points of ArsC rel. abundances
    for a, b, c, d in zip(lons, lats, pcts, biom):
        xpt, ypt = m(a, b)
        if c == max_val:
            #multiplt by a factor to make bubbles visible
            m.scatter(xpt, ypt, marker='o', c='#FF6103', s=c*factor, alpha=0.8, edgecolor='#9B30FF', zorder=3)
            print "Max rel. percent is:", c, "in the", d, "at lon:", a, "and lat:", b
        else:
            m.scatter(xpt, ypt, marker='o', c='#FF6103', s=c*factor, alpha=0.8, zorder=3)

    ## scatter point legend
    slabels = []
    spoints = []
    points = np.linspace(0, max_val, 6)
    for i in points[1:]:
        x = plt.scatter([], [], s=i*factor, c="#FF6103", alpha=0.8) #multiple by size factor to get similar size as actual data points
        spoints.append(x)
        slabels.append('{0:.2f}'.format(i))
    leg1 = plt.legend(spoints, slabels, ncol=5, frameon=False, fontsize=8, handlelength=2, loc=8,
                          bbox_to_anchor=(0.5, -0.07))

    ax1 = plt.gca().add_artist(leg1)

    plt.title("SAR11 " + enz + " abundances in TARA Oceans metagenomes")
    plt.savefig(out + "_map.pdf", bbox_extra_artists=(leg1,), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script plots locations of SAR11 arsenate reductase abundances "
                                                 "in TARA Oceans data. Also checks correlation between [P] vs. "
                                                 "normalized enzyme abundances in TARA measurements.")
    parser.add_argument("-p", "--projection", required=True, help="Choose projection types (examples: kav7, gall, eck4,"
                                                                  "robin, mbtfpq, cea)")
    parser.add_argument("-c", "--companion_table", required=True, help="Tara ocean companion table")
    parser.add_argument("-x", "--counts", required=True, help="Counts")
    parser.add_argument("-e", "--enz", required=True, help="Enzyme to check")
    parser.add_argument("-r", "--res", required=True, help="Resolution for map boundaries: c, l, i, h, f")
    parser.add_argument("-i", "--phos", required=True, help="csv file of phosphate data from NOAA")
    parser.add_argument("-n", "---nutrients", required=True, help="TARA nutrient measurements")
    parser.add_argument("-z", "--check_corr", required=True, help="yes or no (to check correlation)")
    parser.add_argument("-d", "--dta", required=True, help="Which dataset to check for [P]; choose: tara or global")
    parser.add_argument("-o", "--outprefix", required=True, help="Out prefix")
    args = parser.parse_args()
    coordinates = get_coords(args.companion_table, args.counts, args.enz)
    global_phos = get_global_phosphate(args.phos, args.check_corr, args.outprefix)
    if args.check_corr == "yes":
        check_arsc_phos_corr(coordinates, global_phos, args.nutrients, args.dta, args.outprefix)
    draw_map(args.projection, coordinates, args.enz, args.outprefix, args.res, global_phos)
