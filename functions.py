

def fix_cam_time(ds):
    # Author: Marte Sofie Buraas / Ada Gjermundsen    

    """ NorESM raw CAM h0 files has incorrect time variable output,
    thus it is necessary to use time boundaries to get the correct time
    If the time variable is not corrected, none of the functions involving time
    e.g. yearly_avg, seasonal_avg etc. will provide correct information

    Parameters
    ----------
    ds : xarray.DaraSet

    Returns
    -------
    ds_weighted : xarray.DaraSet with corrected time
    """
    from cftime import DatetimeNoLeap

    months = ds.time_bnds.isel(nbnd=0).dt.month.values
    years = ds.time_bnds.isel(nbnd=0).dt.year.values
    dates = [DatetimeNoLeap(year, month, 15) for year, month in zip(years, months)]
    ds = ds.assign_coords(time=dates)
    return ds

def computeWeightedMean(ds):

    # Author: Anne Fouilloux
    import numpy as np

    # Compute weights based on the xarray you pass
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    # Compute weighted mean
    air_weighted = ds.weighted(weights)
    weighted_mean = air_weighted.mean(("lon", "lat"))
    return weighted_mean


def polarCentral_set_latlim(lat_lims, ax):
    
    # Author: Anne Fouilloux
    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib.path as mpath

    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

def computeSeasonalMean(ds, season):

    # Author: Astrid Bragstad Gjelsvik
    # Works for xarray datasets with monthly averages as time axis

    import xarray as xr
    if season == 'DJF':
        ds_season = xr.concat([ds.sel(time=slice(1, 2)), ds.sel(time=12)], dim='time')
        ds_season = ds_season.mean('time')
        return ds_season
    elif season == 'MAM':
        ds_season = ds.sel(time=slice(3, 5))
        ds_season = ds_season.mean('time')
        return ds_season
    elif season == 'JJA':
        ds_season = ds.sel(time=slice(6, 8))
        ds_season = ds_season.mean('time')
        return ds_season
    elif season == 'SON':
        ds_season = ds.sel(time=slice(9, 11))
        ds_season = ds_season.mean('time')
        return ds_season
    else:
        print('Please provide a season on the form "JJA", "DJF", "MAM" or "SON"')

def regrid_to_pressure(ds, var):
    """
    This function regrids from sigma to pressure levels
     Parameters
     ----------
     ds :           xarray.Dataset pressure varibales [p0, ps, a (->hyam), b (->hybm)] and with atmospheric data to be interpolated to pressure levels.
                    The order of the dimensions is specific.
                    The three rightmost dimensions must be level x lat x lon [e.g. TS(time,lev,lat,lon)].
                    The order of the level dimension must be top-to-bottom.
     var :          str, name of the atmospheric variable to be interpolated to pressure levels

     Returns
     -------
     da :            xarray.DataArray with the interpolated atmospheric variable with vertical coordinate
                     pressure in Pa

    PyNGL:
    conda create --name pyn_env --channel conda-forge pynio pyngl
    source activate pyn_env

    Ngl interpolation : https://www.pyngl.ucar.edu/Functions/Ngl.vinth2p.shtml
    array = Ngl.vinth2p(datai, hbcofa, hbcofb, plevo,
                     psfc, intyp, p0, ii, kxtrp)
    Ngl vinth2p arguments/parameters:
    datai: A NumPy array of 3 or 4 dimensions. This array needs to contain a level dimension in hybrid coordinates. The order of the dimensions is specific.
    The three rightmost dimensions must be level x lat x lon [e.g. TS(time,lev,lat,lon)]. The order of the level dimension must be top-to-bottom.
    hbcofa: A one-dimensional NumPy array or Python list containing the hybrid A coefficients. Must have the same dimension as the level dimension of datai. The order must be top-to-bottom.
    hbcofb: A one-dimensional NumPy array or Python list containing the hybrid B coefficients. Must have the same dimension as the level dimension of datai. The order must be top-to-bottom.
    plevo: A one-dimensional NumPy array of output pressure levels in mb.
    psfc: A multi-dimensional NumPy array of surface pressures in Pa. Must have the same dimension sizes as the corresponding dimensions of datai.
    intyp: A scalar integer value equal to the interpolation type: 1 = LINEAR, 2 = LOG, 3 = LOG LOG
    p0: A scalar value equal to surface reference pressure in mb.
    ii: Not used at this time. Set to 1.
    kxtrp: A logical value. If False, then no extrapolation is done when the pressure level is outside of the range of psfc.
    """
    import Ngl
    import xarray as xr
    import numpy as np

    print("In regrid_to_pressure atf.function. Regridding %s to pressure levels" % var)
    #  Extract the desired variables (need numpy arrays for vertical interpolation)
    hyam = ds["hyam"]
    if "time" in hyam.dims:
        hyam = hyam.isel(time=0).drop_vars("time")
    hybm = ds["hybm"]
    if "time" in hybm.dims:
        hybm = hybm.isel(time=0).drop_vars("time")
    psrf = ds["PS"]

    # Note that the units for psfc are Pascals (Pa) whereas the units for plevo and p0 are millibars (mb).
    P0mb = 0.01 * ds["P0"]
    if "time" in P0mb.dims:
        P0mb = P0mb.isel(time=0).drop_vars("time")

    # pnew = ds.lev.values
    # pnew = np.arange(5.0, 1000., 20.)
    pnew = np.sort(np.append(ds.lev.values, (ds.lev + 0.5 * ds.lev.diff("lev")).values))
    intyp = 1  # 1=linear, 2=log, 3=log-log
    kxtrp = False  # True=extrapolate

    # The order of the level dimension must be top-to-bottom.
    lev = ds.lev.values
    assert all(
        earlier <= later for earlier, later in zip(lev, lev[1:])
    ), f"The vertical coordinate must be top-to-bottom: {lev}"
    daP = Ngl.vinth2p(ds[var], hyam, hybm, pnew, psrf, intyp, P0mb, 1, kxtrp)
    daP[daP == 1e30] = np.NaN

    if "year" in ds[var].dims:
        da = xr.DataArray(
            daP,
            dims=("year", "plev", "lat", "lon"),
            coords={
                "year": ds.year,
                "plev": np.asarray(pnew),
                "lat": ds.lat,
                "lon": ds.lon,
            },
        )
    else:
        da = xr.DataArray(
            daP,
            dims=("time", "plev", "lat", "lon"),
            coords={
                "time": ds.time,
                "plev": np.asarray(pnew),
                "lat": ds.lat,
                "lon": ds.lon,
            },
        )
    da = da.where(da.plev <= ds.PS)
    da.attrs["units"] = ds[var].units
    da.attrs["long_name"] = ds[var].long_name
    da.attrs["standard_name"] = ds[var].long_name.replace(" ", "_")
    return da.to_dataset(name=var)