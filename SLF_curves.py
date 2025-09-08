# Author: Stefan Hofer
# Modified by: Astrid Bragstad Gjelsvik

import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size':18})


# CALIOP DATA
folder = '/projects/NS9600K/shofer/'
data_folder = 'caliop_olimpia_new/netcdf_format/'

files_ann = ['bulk_slfs_annual.nc', 'ct_slfs_annual.nc']
files_seasonal = ['bulk_slfs_seasonal.nc', 'ct_slfs_seasonal.nc']

ann_bulk = xr.open_dataset(folder + data_folder + files_ann[0])
ann_ct = xr.open_dataset(folder + data_folder + files_ann[1])

# NORESM Data
case1 = 'N1850_f19_tn14_noresm2_3_slf_output_20250529'; label1 = 'piClim'
case2 = 'N1850_f19_tn14_noresm2_3_slf_output_0_5xCO2_20250529'; label2 = '0.5xCO2'
case3 = 'N1850_f19_tn14_noresm2_3_slf_output_2xCO2_20250529'; label3 = '2xCO2'
case4 = 'N1850_f19_tn14_noresm2_3_slf_output_4xCO2_20250529'; label4 = '4xCO2'


folder_n = ['/projects/NS9600K/astridbg/data/model/noresm_rawdata/cases/'+case1+'/atm/hist/',
	    '/projects/NS9600K/astridbg/data/model/noresm_rawdata/cases/'+case2+'/atm/hist/',
        '/projects/NS9600K/astridbg/data/model/noresm_rawdata/cases/'+case3+'/atm/hist/',
        '/projects/NS9600K/astridbg/data/model/noresm_rawdata/cases/'+case4+'/atm/hist/']

data_n = [case1+'.cam.h0.20*.nc',
	      case2+'.cam.h0.20*.nc',
          case3+'.cam.h0.20*.nc',
          case4+'.cam.h0.20*.nc']

def preprocess(ds):
    ds_new = ds[['SLFXCLD_ISOTM', 'CT_SLFXCLD_ISOTM',
                 'CT_CLD_ISOTM', 'CLD_ISOTM']]
    return ds_new

list_cases = []
for i in range(4):
    print(folder_n[i])
    case_one = xr.open_mfdataset(
        folder_n[i] + data_n[i], preprocess=preprocess)
    list_cases.append(case_one)
    print(i,": ",folder_n[i])

# =======================================================
# FUNCTIONS
# =======================================================


def arctic_slf_weighted(ds, s_bnd=66.6, n_bnd=90, ss_bnd=None, nn_bnd=None):
    '''Computes the mean of the CALIOP SLF in the Arctic,
    weighted by the size of the grid cell.
    '''
    ds_arctic = ds.sel(lat=slice(s_bnd, n_bnd))

    weighted_arctic = ds_arctic.weighted(ds_arctic.cell_weight)
    mean_arctic = weighted_arctic.mean(dim=['lat', 'lon']).SLF
    std_arctic = weighted_arctic.std(dim=['lat','lon']).SLF

    return mean_arctic, std_arctic


def arctic_slf_noresm(ds, s_bnd=66.6, n_bnd=90):
    '''Computes the mean SLF of NorESM2 input in the Arctic. Boundaries given by
    "s_bnd" and "n_bnd" variables.

    Defaults: s_bnd=66.6N
              n_bnd=90.0N
    '''
    lats = ds.lat.sel(lat=slice(s_bnd, n_bnd))
    # Select between boundaries given and delete first three months
    ds_arctic = ds.sel(lat=slice(s_bnd, n_bnd),
                       time=slice('2040-01-01', '2069-12-31'))
    weights_n = np.cos(np.deg2rad(lats))
    weighted = ds_arctic.weighted(weights_n)
    mean_arctic = weighted.mean(dim=['lat', 'lon', 'time'])

    mean_arctic['slf_ct'] = (mean_arctic.CT_SLFXCLD_ISOTM /
                             mean_arctic.CT_CLD_ISOTM)
    mean_arctic['slf_bulk'] = (mean_arctic.SLFXCLD_ISOTM /
                               mean_arctic.CLD_ISOTM)
    mean_arctic['isotherms_mpc'] = mean_arctic['isotherms_mpc'] - 273.15
    mean_arctic = mean_arctic.rename({'isotherms_mpc': 'isotherm'})

    return mean_arctic


def plot_slf_iso(ds, fig=False, axs=False):
    if axs == False:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 6))
    else:
        fig = fig
        axs = axs
    xr.plot.line(ds * 100, y='isotherm', yincrease=False, ax=axs)
    sns.despine()
    fig.tight_layout()

    return fig, axs



# This is the CALIOP data for the Arctic (no values above 82 deg N)
et_n_bulk, std_n_bulk = arctic_slf_weighted(ann_bulk, s_bnd=66.6, n_bnd=82)
et_n_ct, std_n_ct  = arctic_slf_weighted(ann_ct, s_bnd=66.6, n_bnd=82)


# ==============================================================================
# PLOT THE TWO CASES
# ==============================================================================

plt.close('all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6),dpi=300)

# Cases CAM5 and Andenes
case_one_slf = arctic_slf_noresm(list_cases[0], s_bnd=66.6, n_bnd=82)
case_two_slf = arctic_slf_noresm(list_cases[1], s_bnd=66.6, n_bnd=82)
case_three_slf = arctic_slf_noresm(list_cases[2], s_bnd=66.6, n_bnd=82)
case_four_slf = arctic_slf_noresm(list_cases[3], s_bnd=66.6, n_bnd=82)
# =============================================================
# =============== PLOTTING ROUTINE ============================
# =============================================================
"""
xr.plot.line(et_n_bulk * 100, y='isotherm',
                 ax=ax, color='black', label='Bulk Obs', lw=2.5, ls='dotted')
ax.errorbar(x=et_n_bulk * 100, y=et_n_bulk.isotherm, xerr=std_n_bulk * 100,
                 marker='.',color='black',ls='dotted',lw=2.5)
xr.plot.line(et_n_ct * 100, y='isotherm',
                 ax=ax, color='black', label='Cloud Top Obs', lw=2.5, ls='dashdot')
ax.errorbar(x=et_n_ct * 100, y=et_n_ct.isotherm, xerr=std_n_ct * 100,
                 marker='.',color='black',ls='dashdot',lw=2.5)
"""

xr.plot.line(case_one_slf.slf_bulk * 100, y='isotherm',
                label='Bulk '+label1, ax=ax, lw=2.5, ls='dashed', c='tab:green')
xr.plot.line(case_one_slf.slf_ct * 100, y='isotherm',
                label='Cloud Top '+label1, ax=ax, lw=2.5, c='tab:green')


xr.plot.line(case_two_slf.slf_bulk * 100, y='isotherm',
                label='Bulk '+label2, ax=ax, ls='dashed',lw=2.5, c='tab:blue')
xr.plot.line(case_two_slf.slf_ct * 100, y='isotherm',
                label='Cloud Top '+label2, ax=ax, lw=2.5, c='tab:blue')

xr.plot.line(case_three_slf.slf_bulk * 100, y='isotherm',
                label='Bulk '+label3, ax=ax, ls='dashed',lw=2.5, c='tab:orange')
xr.plot.line(case_three_slf.slf_ct * 100, y='isotherm',
                label='Cloud Top '+label3, ax=ax, lw=2.5, c='tab:orange')

xr.plot.line(case_four_slf.slf_bulk * 100, y='isotherm',
                label='Bulk '+label4, ax=ax, ls='dashed',lw=2.5, c='tab:red')
xr.plot.line(case_four_slf.slf_ct * 100, y='isotherm',
                label='Cloud Top '+label4, ax=ax, lw=2.5, c='tab:red')


ax.invert_yaxis()

sns.despine()
fig.tight_layout()

ax.set_xlabel('Supercooled Liquid Fraction (%)', fontsize=18)
ax.set_ylabel('Isotherm (C)', fontsize=18)
fig.subplots_adjust(right=0.6)
ax.legend(frameon=False, loc="upper left",bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(
    '/projects/NS9600K/astridbg/arctic-cld-feedbacks/SLF.png')
