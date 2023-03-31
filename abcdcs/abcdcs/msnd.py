# msnd.py
"""Calculate MSND from PIV displacement fields for a single movie.
"""

from typing import Tuple
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt


# def calculate_msnd(ds, components=['u', 'v'], pixsize=None
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """MSND analysis from PIV displacement fields.

#     Compute MSND for each lag time with all possible pairs, but outliers
#     (absolute value of zscore >= 3) excluded.

#     Parameters
#     ----------
#     ds : xarray.Dataset
#         Displacement fields at different lag times from a single 
#         movie
#     components : list, optional
#         Names of displacement field components, by default ['u','v']
#     pixsize : float, optional
#         Micron per pixel (px). Default to None will use 1 for calculation.
#         If "meta" is provided, will use the unit information from `ds`
#         (`ds.attrs["pixelsize"]`).

#     Returns
#     -------
#     stats : pd.DataFrame
#         Dataframe with lag time and corresponding mean and SD of MSND across
#         all pairs sampled.
#     raw : pd.DataFrame
#         Dataframe with MSND for individual PIV displacement fields.
#     """
#     # Don't alter the Dataset passed in as argument
#     piv = ds.copy()

#     # x and y components for the displacement field
#     u, v = components
#     # convert pixels to microns?
#     if pixsize is None:
#         pixsize = 1
#     elif pixsize == "meta":
#         pixsize = ds.attrs["pixelsize"]

#     # MSND for each displacement field
#     piv['snd'] = pixsize**2 * (piv[u]**2 + piv[v]**2)
#     piv['msnd'] = piv['snd'].mean(dim=('X','Y'), skipna=True)
#     df = (piv[["msnd"]]
#           .to_dataframe()
#           .dropna(axis=0, how='any').reset_index()
#     )
#     df = df[["lag", "lag_s", "T", "T_s", "msnd"]]

#     # Remove outliers before aggregating all MSND for the same lag time
#     zscore = lambda x: (x - x.mean()) / x.std()
#     z_thresh = 3
#     df['zscore'] = df.groupby('lag')['msnd'].transform(zscore)
#     stats = (df[df['zscore'].abs() < z_thresh]
#         .groupby('lag_s')['msnd']
#         .aggregate(msnd_mean='mean', msnd_std='std')
#         .reset_index()
#     )
    
#     return stats, df

def calculate_msnd(
        ds, 
        components=['u', 'v'], 
        weight=None, 
        pixsize=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """MSND analysis from PIV displacement fields.

    Compute MSND for each lag time with all possible pairs, but outliers
    (absolute value of zscore >= 3) excluded.

    Parameters
    ----------
    ds : xarray.Dataset
        Displacement fields at different lag times from a single 
        movie. 
    components : list, optional
        Names of displacement field components, by default ['u','v']
    weight : str, optional default to None
        Variable name in `ds`, corresponds to the intensity image as 
        weight when averaging MSND for a single PIV displacement field.
        If None, use uniform weight.
    pixsize : float, optional
        Micron per pixel (px). Default to None will use 1 for calculation.
        If "meta" is provided, will use the unit information from `ds`
        (`ds.attrs["pixelsize"]`).

    Returns
    -------
    stats : pd.DataFrame
        Dataframe with lag time and corresponding mean and SD of MSND across
        all pairs sampled.
    raw : pd.DataFrame
        Dataframe with MSND for individual PIV displacement fields.
    """
    # TODO: is hard copy necessary within the function?
    # Don't alter the Dataset passed in as argument
    piv = ds.copy()

    # x and y components for the displacement field
    u, v = components
    # convert pixels to microns?
    if pixsize is None:
        pixsize = 1
    elif pixsize == "meta":
        pixsize = ds.attrs["pixelsize"]

    
    # MSND for each displacement field
    
    # Weight provided as one of the variables in the Dataset; otherwise
    # use uniform weight
    if weight is not None:
        W = piv[weight]
    else:
        W = xr.ones_like(piv[u])
    
    # Length of each displacement in sq. micron
    piv['snd'] = pixsize**2 * (piv[u]**2 + piv[v]**2)
    
    # Total weights: (1) set weight to NaN where displacement is NaN
    #                (2) sum over with NaNs ignored
    W_sum = W.where(piv['snd'].notnull()).sum(dim=('X','Y'), skipna=True)
    
    # Weighted sum of displacement length normalized by total weight
    piv['msnd'] = (piv['snd'] * W).sum(dim=('X','Y'), skipna=True) / W_sum
    
    
    df = (piv[["msnd"]]
          .to_dataframe()
          .dropna(axis=0, how='any').reset_index()
    )
    df = df[["lag", "lag_s", "T", "T_s", "msnd"]]

    # Remove outliers before aggregating all MSND for the same lag time
    zscore = lambda x: (x - x.mean()) / x.std()
    z_thresh = 3
    df['zscore'] = df.groupby('lag')['msnd'].transform(zscore)
    stats = (df[df['zscore'].abs() < z_thresh]
        .groupby('lag_s')['msnd']
        .aggregate(msnd_mean='mean', msnd_std='std')
        .reset_index()
    )
    
    return stats, df


def plot_msnd_stats(df, ax=None, errorbar=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
    if errorbar:
        _plot_msnd_stats(ax, df['lag_s'], df['msnd_mean'], df['msnd_std'], **kwargs)
    else:
        _plot_msnd_stats(ax, df['lag_s'], df['msnd_mean'], **kwargs)
    try:
        fig.set_tight_layout(True)
    except UnboundLocalError:
        pass

    return ax

def _plot_msnd_stats(ax, x, y, err=None, **kwargs):
    if err is not None:
        ax.errorbar(x=x, y=y, yerr=err, fmt='o', alpha=0.5, **kwargs)
    else:
        ax.scatter(x=x, y=y, alpha=0.5, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Lag time (s)')
    ax.set_ylabel('MSND ($\mu m^2$/s)')

    return ax


def plot_msnd_raw(df):
    g = sns.relplot(data=df[df["zscore"].abs()<3], 
                    x="T_s", y="msnd", hue="lag_s", 
                    kind="line", palette="Set2")
    return g
    