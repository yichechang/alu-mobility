# quiver.py
"""
Generate quiver plots for a single PIV displacement field
"""

import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker
import numpy as np

# ----------------------------------------------------------------------
#    Global default settings for plotting
# ----------------------------------------------------------------------
DEFAULT_QUIVER_KWARGS = {
    'angles': 'xy', 
    'scale_units': 'xy',
    'width': 0.005, 
    'headwidth': 3, 
    'headlength': 5, 
    'headaxislength': 5,
    'scale': 0.1, 
    'minlength': 1, 
    'minshaft': 1,
}
DEFAULT_IMSHOW_KWARGS = {
    'origin': 'upper',
}
DEFAULT_QUIVER_CB_LOC = "left"
DEFAULT_QUIVER_CMAP_ANGLE = 'hsv'
DEFAULT_QUIVER_CMAP_AMP = 'viridis'
DEFAULT_IMSHOW_CB_LOC = "right"


# ----------------------------------------------------------------------
#    Main plotting functions to use from this module
# ----------------------------------------------------------------------
def quiverplot_from_pivset(pivset, movie=None,
                           lag=None, T=None, ilag=None, iT=None, 
                           imgmode=None, 
                           **kwargs):
    """
    Show specified displacement field from a full PIV dataset and movie.

    This function provides an easy interface that integrates the 
    folloing operations: selecting a PIV field from a full PIV dataset 
    by specifying lag time (`lag`) and timepoint (`T`) and displaying
    the displacement field with configurations passed as keyword 
    arguments to `quiverplot_from_piv` and ultimately `quiverplot`.
    Optionally, quiver plot can be drawn on top of an image of the first,
    the second, or both frames based on `imgmode` setting.

    Parameters
    ----------
    pivset : xr.Dataset
        Object returned by function `loadpiv.read_pivresult`. It is a 
        full PIV dataset that contains PIV fields of various lag times
        and timepoints.
    movie : xr.DataArray
        Assumed to be prepared by `AICSImage(...).xarray_data`. Or as
        long as it contains a dimention `T`.
    lag, T, ilag, iT : 
        Number to specify the lag (frames) and T (frame), or their 
        simple positional index.
    imgmode : str, optional
        Default to None, showing no image. 'pair', 'first', 'second' are
        possible options.
    kwargs : 
        See `quiverplot_from_piv` and `quiverplot` for additional kwargs 
        to provide.

    Returns
    -------
    plt.Axes
    """

    # Extract a single PIV
    piv = _get_piv_from_pivset(pivset, lag, T, ilag, iT)

    # Extract the pair of images used for the single PIV extracted
    if movie is not None:
        im1, im2 = _get_pair_from_movie(movie, piv)
    
    # Choose to show either the 1st, 2nd, or both images to show
    if imgmode == 'pair':
        image = _diff_image_pair(im1, im2)
    elif imgmode == 'first':
        image = im1
    elif imgmode == 'second':
        image = im2
    else:
        image = None

    return quiverplot_from_piv(piv, image=image, **kwargs)



def quiverplot_from_piv(piv, components=('fu', 'fv'), **kwargs):
    """Directly plot displacement field for a single PIV as xr.Dataset

    Parameters
    ----------
    piv : xr.Dataset
        Needs to be of ndim=2
    components : tuple, optional
        variable names for the U and V components of displacement vector,
        by default ('fu', 'fv')
    kwargs : 
        See `quiverplot` for possible kwargs to provide

    Returns
    -------
    plt.Axes
    """
    return quiverplot(piv['X'], piv['Y'],
                      piv[components[0]], piv[components[1]], **kwargs)


def quiverplot(
        x, y, u, v, 
        ax=None, 
        colorby=None, cb_quiver=False,
        arrowkey=False, pxpergrid=None, 
        unitlength=False, fillna=False, quiver_kwargs=dict(),
        image=None, cb_image=False, imshow_kwargs=dict(), 
    ) -> plt.Axes:
    """
    A wrapper around plt.quiver to handle common PIV plotting needs.

    Parameters
    ----------
    x, y, u, v: np.ndarray or xr.DataArray
        ndim = 2
    ax : plt.Axes, optional
        Axes to draw quiver on, by default None will create a new one
    colorby : str or None, optional
        If 'angle', arrows will be colored by angle, by default None
    cb_quiver : bool, optional
        switch to show arrow color code, by default False
    arrowkey : bool, optional
        switch to show arrow key, by default False
    gridsize : float, optional
        grid spacing used in PIV analysis, by default None
    unitlength : bool, optional
        switch to show all vectors as the same length, by default False
    fillna : bool, option
        switch to replace nan (not shown) with 0 (drawn as a dot)
    image : np.ndarray or xr.DataArray, option
        image to overlay with quiver. by default None
    cb_image : bool, optional
        switch to show intensity colorbar for the image
    imshow_kwargs : dict, optional
        any kwargs to pass to `plt.imshow`, by default dict()
    quiver_kwargs : dict, optional
        any kwargs to pass to `plt.quiver`, by default dict()

    Returns
    -------
    A plt.Axes with quiver drawn on.

    Notes
    ------
    * `arrowkey` and `unitlength` cannot both be True.
    * `colorbar` cannot be True unless `colorby` is set to `angle`
    """

    # TODO:
    # - [ ] add try except to deal with numpy array input (.nan_to_num)
    if fillna:
        u = u.fillna(0)
        v = v.fillna(0)


    #
    # Ensure we have an axis to work with
    #
    ax = _prepare_axis(ax)
    
    #
    # Quiver plot itself is generated here
    #
    add_quiver = _gen_quiver_func(colorby=colorby)
    qobj = add_quiver(x, y, u, v, ax, unitlength=unitlength, **quiver_kwargs)

    #
    # Adding other elements to the axis or outside of the axis
    #

    # image to overlay
    if image is not None:
        iobj = _add_image(ax, image=image, **imshow_kwargs)
    
    # all colorbar drawings are dealt together as we want to be using
    # just one same divider
    if any([cb_quiver, cb_image]):
        divider = None
        if cb_quiver:
            add_quiver_cb = _gen_quiver_colorbar_func(colorby=colorby)
            _, divider = add_quiver_cb(ax, qobj, divider=divider)
        if cb_image:
            _, divider = _add_image_colorbar(ax, iobj, divider=divider)

    if arrowkey:
        if unitlength:
            raise ValueError("Can't set both unitlength and arrowkey "
                             "to True at the same time.")
        if pxpergrid is None:
            raise ValueError("gridsize needs to be provided when "
                             "arrowkey is set to True.")
        _add_arrowkey(ax, qobj, pxpergrid)

    return ax


# ----------------------------------------------------------------------
#    Quiver plot itself
# ----------------------------------------------------------------------
def _gen_quiver_func(colorby=None):
    """Choosing the correct quiver generating function"""
    if colorby == 'angle':
        return _quiver_color_by_angle
    elif colorby == 'amp':
        return _quiver_color_by_amp
    else:
        return _quiver


def _adjust_uv(u, v, unitlength=False):
    if unitlength:
        u, v = _unit_vector(u, v)
    return u, v

def _quiver(x, y, u, v, ax, unitlength=False, **qkwargs):
    new_qkwargs = {**DEFAULT_QUIVER_KWARGS, **qkwargs}
    u, v = _adjust_uv(u, v, unitlength)
    qobj = ax.quiver(x, y, u, v, **new_qkwargs)
    return qobj


def _quiver_color_by_angle(x, y, u, v, ax, unitlength=False, **qkwargs):
    theta = _calc_angle(u, v)
    new_qkwargs = {**DEFAULT_QUIVER_KWARGS, **qkwargs}
    u, v = _adjust_uv(u, v, unitlength)
    qobj = ax.quiver(x, y, u, v,
                     theta, cmap=DEFAULT_QUIVER_CMAP_ANGLE, clim=(-180, 180),
                     **new_qkwargs)
    return qobj


def _quiver_color_by_amp(x, y, u, v, ax, unitlength=False, **qkwargs):
    new_qkwargs = {**DEFAULT_QUIVER_KWARGS, **qkwargs}
    amp = _calc_amp(u, v)
    u, v = _adjust_uv(u, v, unitlength)
    qobj = ax.quiver(x, y, u, v,
                     amp, cmap=DEFAULT_QUIVER_CMAP_AMP,
                     **new_qkwargs)
    return qobj

# ----------------------------------------------------------------------
#    Helper functions for plotting
# ----------------------------------------------------------------------

#
# transforming underlying data
#
def _unit_vector(u, v):
    U = u / np.sqrt(u**2 + v**2)
    V = v / np.sqrt(u**2 + v**2)
    return U, V


def _calc_angle(u, v):
    """Calculate angle for vector (u,v) in degrees, -180 to 180."""
    return np.arctan2(v, u) / np.pi * 180


def _calc_amp(u, v):
    return np.sqrt(u**2 + v**2)


#
# plotting data 
#
def _prepare_axis(ax=None) -> plt.Axes:
    """Prepare a formatted axis for a quiver plot. New axis will be 
    created if none is given.
    """
    # create a new axis if not gien
    if ax is None:
        _, ax = plt.subplots()
    # format axis for quiver plot
    ax.set_aspect(1)
    ax.set_axis_off()
    return ax


def _add_image(ax, image, **imshow_kwargs):
    
    # image can be greyscaled single frame (ndim=2) or RGB fused image 
    # pair (ndim=3). If the former, use grey colormap specifically. If 
    # latter, use plt.imshow default.
    if image.ndim == 2:
        extra_imshow_kwargs = {'cmap': 'Greys_r'}
    if image.ndim == 3:
        extra_imshow_kwargs = dict()    
    new_imshow_kwargs = {**DEFAULT_IMSHOW_KWARGS, 
                         **extra_imshow_kwargs, 
                         **imshow_kwargs}

    iobj = ax.imshow(image, **new_imshow_kwargs)

    return iobj


#
# adding legend
#
def _add_arrowkey(qax, qobj, gridsize):
    qax.quiverkey(qobj, 0.05, -0.05, gridsize/2,
                  coordinates='axes', label='0.5 grid', labelpos='S')


#
# adding legend -- colorbars
#
def _gen_quiver_colorbar_func(colorby=None):
    """Choosing the correct colorbar drawing function"""
    if colorby == 'angle':
        return _add_angle_colorbar
    if colorby == 'amp':
        return _add_amp_colorbar


def _add_some_colorbar(ax, obj, divider=None, 
                       location=None, size="3%", pad=0.05):
    """
    Draw a colorbar of the same height as the main axis.

    We can specify the divider or create a new one if none is provided. 
    The divider used then is returned so can be used in future colorbar
    construction.
    """
    # make a divider if none provided
    if divider is None:
        divider = make_axes_locatable(ax)

    cax = divider.append_axes(location, size=size, pad=pad)
    plt.colorbar(obj, cax=cax)
    cax.yaxis.set_ticks_position(location)

    return cax, divider


def _add_angle_colorbar(ax, obj, divider, 
                        location=DEFAULT_QUIVER_CB_LOC, **kwargs):
    """Quiver plot colorbar according to vector angle"""
    cax, _divider = _add_some_colorbar(ax=ax, obj=obj, divider=divider,
                                       location=location, **kwargs)
    # extra settings
    cax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(45))
    cax.set_title(r'$\theta$ [$\degree$]')

    return cax, _divider


def _add_amp_colorbar(ax, obj, divider, 
                      location=DEFAULT_QUIVER_CB_LOC, **kwargs):
    cax, _divider = _add_some_colorbar(ax=ax, obj=obj, divider=divider,
                                       location=location, **kwargs)
    # extra settings
    cax.set_title('$||\overrightarrow{d}||$ [px]')

    return cax, _divider


def _add_image_colorbar(ax, obj, divider, 
                        location=DEFAULT_IMSHOW_CB_LOC, **kwargs):
    """Image colorbar for intensity"""
    cax, _divider = _add_some_colorbar(ax=ax, obj=obj, divider=divider, 
                                       location=location, **kwargs)
    # extra settings
    cax.set_title('$I$ [a.u.]')

    return cax, _divider


# ----------------------------------------------------------------------
#   Helper functions for handling background images
# ----------------------------------------------------------------------

def _diff_image_pair(im1, im2):
    """
    Make RGB image by overlaying 1st (magenta) and 2ns (green) images.

    Image 1 will be in magenta and image 2 in green. So the movement 
    should be read from magenta to green. Normalized image has range
    from 0 to 1.
    """

    def normalize_minmax(a):
        return (a - a.min()) / (a.max() - a.min())
    
    im1 = normalize_minmax(im1)
    im2 = normalize_minmax(im2)
    fused = np.dstack((im1, im2, im1))

    return normalize_minmax(fused)


def _get_piv_from_pivset(pivset, lag=None, T=None, ilag=None, iT=None):
    """
    Return a PIV field specified by a single (lag, T) pair.

    `lag` or `T` doesn't need to be exact, and the 'nearest' value 
    will be selected automatically. 
      
    Note that this function is meant to extract only a single PIV
    field specified by a single lag and a single T. Do not pass `slice`
    objects in, as it doesn't work with `.sel(..., method='nearest')`.
    """
    if lag is not None:
        piv = pivset.sel(lag=lag, method='nearest')
    elif ilag is not None:
        piv = pivset.isel(lag=ilag)
    else:
        raise ValueError("Need to provide `lag` or `ilag`.")
    
    if T is not None:
        piv = piv.sel(T=T, method='nearest')
    elif iT is not None:
        piv = piv.isel(T=iT)
    else: 
        raise ValueError("Need to provide `T` or `iT`.")
    
    return piv


def _get_pair_from_movie(movie, piv):
    lag = piv.coords['lag'].item()
    T = piv.coords['T'].item()
    im1 = movie.sel(T=T, method='nearest')
    im2 = movie.sel(T=T+lag, method='nearest')

    return im1, im2