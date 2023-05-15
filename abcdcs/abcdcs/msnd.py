# msnd.py
"""Calculate MSND from PIV displacement fields for a single movie.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss


class MSND:
    """Calculate MSND from PIV displacement fields for a single movie.
    
    Parameters
    ----------
    data : xr.Dataset | tuple[xr.DataArray, xr.DataArray]
        PIV displacement fields. If a Dataset, must have the
        components specified in `components`. The underlying DataArray
        should have its dimensions including ('lag', 'T', 'Y', 'X') and 
        dimension coordinates including lag, lag_s, T, T_s, Y, X.
    components : tuple[str, str] | None, default = None
        Components of the displacement fields. If `data` is a
        Dataset, this must be specified. Otherwise, this is ignored
    mpp : float | None, default = None
        Physical pixel size in microns. If not specified, will try to 
        get from `data` if it is a Dataset. If not found, will use pixel
        as unit (same as specifying `mpp=1`).
    fps : float | None, default = None
        Frame rate in frames per second. If not specified, will
        use coordinate variables found in `data`.
    """

    def __init__(
        self, 
        data: xr.Dataset | tuple[xr.DataArray, xr.DataArray],
        components: tuple[str, str] | None = None,
        mpp: float | None = None, 
        fps: float | None = None,
    ) -> None:
        # prepare displacement fields
        if isinstance(data, xr.Dataset):
            if components is None:
                raise ValueError(
                    "components must be specified if data is a Dataset."
                )
            self._u, self._v = data[components[0]], data[components[1]]
        else:
            self._u, self._v = data
        self._D2 = (self._u**2 + self._v**2).rename('D2')

        
        # use mpp specified. if not specified, try to get from piv 
        # dataset. if not found, use pixel as unit.
        if mpp is not None:
            self._mpp = mpp
        elif isinstance(data, xr.Dataset):
            try:
                self._mpp = data.attrs['pixelsize']
            except KeyError:
                self._use_pixel_as_unit()
        else:
            self._use_pixel_as_unit()

        
        # use fps specified. if not specified, leave it as None so 
        # will use coords found in data.
        self._fps = fps


    def _use_pixel_as_unit(self):
        self._mpp = 1.0
        print("Cannot find physical pixel size. "
              "Unit for MSND is now in pixel squared.")

    def calculate(
        self, 
        method: str = 'normal', 
        **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate MSND from PIV displacement fields.

        Parameters
        ----------
        method : str, default = 'normal'
            Method used to calculate MSND. Must be one of 'normal',
            'weighted', 'eachlevel', or 'eachlevel2d'.
        **kwargs
            Additional keyword arguments passed to the method specified
            in `method`.

            
        Other parameters
        ----------------
        ### normal
        no additional parameters are needed.

        ### weighted
        weights : xr.DataArray
            Weights for each displacement field. Must have the same
            dimensions as `data`. NaNs will be replaced with 0. So
            don't rely on NaNs for magical masking side effects.
            Perform masking to `data` explicitly before passing it
            to this class.
        
        ### eachlevel
        byimage : xr.DataArray
            Array used to group the displacement fields. Must only 
            dimensions that can be found from `data`.
        bins : int | nd.ndarray
            Bins to be passed to `np.digitize`.

        ### eachlevel2d
        byimage : xr.DataArray (coords has 'C' with two values)
        bins : nd.ndarray
            Bin edges.

            
        Returns
        -------
        stats : pd.DataFrame
            MSND statistics. The columns are 'lag', 'lag_s', 'mean',
            'std', 'sem', 'sizeT'. MSND quantities have unit in micron 
            squared. Additional columns 'level' is added if 
            `method = 'eachlevel'`, and two image channel names if 
            `method = 'eachlevel2d'`.
        raw : pd.DataFrame
            MSND calculated from individual PIV displacement fields. The
            columns are 'lag', 'lag_s', 'T', 'T_s', 'msnd'. Additional 
            column 'level' is added if `method = 'eachlevel'`. The extra 
            columns 'T' and 'T_s' are for each time-sample. MSND's unit 
            is in micron squared.
        """
        # dispatch to the method specified
        self._dispatch(method=method)

        # calculate msnd and convert to dataframe with non-NaN rows
        da = self._calculate(**kwargs)
        df = (da
            .to_dataframe()
            .dropna(axis=0, how='any')
            .reset_index()
            [self._colnames]
        )

        # apply unit conversion
        df['msnd'] = df['msnd'] * self._mpp**2
        if self._fps is not None:
            df['lag_s'] = df['lag'] / self._fps
            df['T_s'] = df['T'] / self._fps
            print(f"Overwriting lag_s (and T_s) using fps = {self._fps} "
                  f"({1000 / self._fps} ms between adjacent frames.)")
            
        # calculate statistics
        stats = self._calculate_stats(df)

        # TODO: use image arg info for making colnames at dispatch time
        #
        # update colnames dynamically after stats calculation is done
        if method == 'eachlevel2d':
            colname_mapper = {
                'level1': kwargs['byimage'].coords['C'].values[0],
                'level2': kwargs['byimage'].coords['C'].values[1],
            }
            stats = stats.rename(columns = colname_mapper)
            df = df.rename(columns = colname_mapper)
            
        return stats, df
    

    def _dispatch(self, method: str) -> None:
        if method == 'normal':
            self._colnames = ['lag', 'lag_s', 'T', 'T_s', 'msnd']
            self._calculate = self._normal_msnd
        elif method == 'weighted':
            self._colnames = ['lag', 'lag_s', 'T', 'T_s', 'msnd']
            self._calculate = self._weighted_msnd
        elif method == 'eachlevel':
            self._colnames = ['level', 'lag', 'lag_s', 'T', 'T_s', 'msnd']
            self._calculate = self._eachlevel_msnd
        elif method == 'eachlevel2d':
            self._colnames = ['level1', 'level2', 'lag', 'lag_s', 'T', 'T_s', 'msnd']
            self._calculate = self._eachlevel2d_msnd
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Must be one of 'normal', 'weighted', 'eachlevel', 'eachlevel2d'."
            )

    @staticmethod
    def _msnd(D2: xr.DataArray) -> xr.DataArray:
        return D2.mean(dim=('X', 'Y'), skipna=True).rename('msnd')

    def _normal_msnd(self) -> xr.DataArray:
        """Dispatched msnd calculation function for `method = 'normal'`
        """
        return self._msnd(self._D2)
    
    def _weighted_msnd(self, 
        weights: xr.DataArray
    ) -> xr.DataArray:
        """Dispatched msnd calculation function for `method = 'weighted'`
        """
        weights = weights.fillna(0)
        D2_weighted = (self._D2).weighted(weights)
        return self._msnd(D2_weighted)
    
    def _eachlevel_msnd(self, 
        byimage: xr.DataArray, 
        bins: int | np.ndarray, 
    ) -> xr.DataArray:
        """Dispatched msnd calculation function for `method = 'eachlevel'`
        """
        # aligned image to displacement field
        D2, image = xr.align(self._D2.rename('D2'), 
                             byimage.rename('image'), 
                             join='left')
        # digitize image into levels
        digitized = np.digitize(image, bins=bins, right=True)
        # use right of each bin as level
        # so for x in level e: bins[i-1] < x <= bins[i]=e
        binlabels = bins[1:]

        # calculate msnd for each level
        msnd = []
        for i,b in enumerate(binlabels):
            msnd.append(
                self._msnd(D2.where(digitized == i))
                .assign_coords(level=b)
            )

        return xr.concat(msnd, dim='level')
    
    def _eachlevel2d_msnd(self,
        byimage: xr.Dataset,
        bins: np.ndarray,
    ) -> xr.DataArray:
        
        if isinstance(bins, int):
            raise ValueError(f"2D-level MSND only supports `bins` "
                             f"specified as edges, but {type(bins)} "
                             f"is provided.")
        
        def binned_statistic_2d(x1, x2, val, statistic, bins):
            ret = ss.binned_statistic_2d(
                x1.flatten(),
                x2.flatten(),
                val.flatten(),
                statistic=statistic,
                bins=bins,
            )
            return ret[0]

        D2, image = xr.align(self._D2, byimage, join='left')
        centers = (bins[0:-1] + bins[1:])/2

        binned_D2 = (
            xr.apply_ufunc(
                binned_statistic_2d,
                    image.isel(C=0),
                    image.isel(C=1),
                    D2,
                input_core_dims=[['Y','X'], ['Y','X'], ['Y','X']],
                output_core_dims=[['level1','level2']],
                vectorize=True,
                kwargs={
                    'statistic': lambda x: np.nanmean(x),
                    'bins': bins,
                }
            ).assign_coords({'level1': centers, 'level2': centers})
        )

        return binned_D2.rename('msnd')



    def _calculate_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        groupby_cols = [c for c in self._colnames if c not in ['T', 'T_s', 'msnd']]
        stats = (df
            .groupby(groupby_cols)['msnd']
            .aggregate(mean='mean', std='std', size='count')
            .reset_index()
        )
        stats['sem'] = stats['std'] / np.sqrt(stats['size'])
        return stats
        




def plot_msnd_stats(df, ax=None, errorbar=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,3))
    if errorbar:
        _plot_msnd_stats(ax, df['lag_s'], df['mean'], df['std']/np.sqrt(df['size']), **kwargs)
    else:
        _plot_msnd_stats(ax, df['lag_s'], df['mean'], **kwargs)
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
    g = sns.relplot(data=df, 
                    x="T_s", y="msnd", hue="lag_s", 
                    kind="line", palette="Set2")
    return g
    

def calculate_msnd(
        ds, 
        components=['u', 'v'], 
        weight=None, 
        pixsize=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(TO DEPRECATE) MSND analysis from PIV displacement fields.

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