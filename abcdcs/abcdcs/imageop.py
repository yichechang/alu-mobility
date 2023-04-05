'''
image operations
'''
from typing import List, Union, Callable
import numpy as np
import xarray as xr
from skimage.morphology import disk, binary_dilation, binary_erosion
from aicsimageio import AICSImage


class Image:

    @staticmethod
    def read(fpath: str, fmt: str, 
             channel_names: List[str] = None, **kwargs,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Read image from file.

        Parameters
        ----------
        fpath : str
            Path to image file.
        fmt : str
            Image format. 'DataArray' or 'Dataset'.
        channel_names : List[str]
            Names for channels. If None, use default names.
        **kwargs : dict, optional

        Other Parameters
        ----------------
        bitdepth : int, optional. Default is None.
        rescale : bool, optional. Default is False. If True, bitdepth 
            must be provided.
        background : float, optional. Default is 0. Provide as raw, 
            value before rescaling to [0, 1].
        dtype : np.dtype, optional. Default is np.float64.
        squeeze : bool, optional. Default is False.

        Returns
        -------
        xr.DataArray or xr.Dataset
        """
        reader = ImageReader(channel_names, **kwargs)
        return reader.read(fpath, fmt)
    
    @staticmethod
    def normalize(data: Union[xr.Dataset, xr.DataArray], 
        method: str, **kwargs,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Rescale for 2D [+ T] [+ C] image, plane by plane.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset
        method : str
            Normalizing method. Currently only 'zscore' and 'pclip' are 
            supported.
        """

        rescaled = xr.apply_ufunc(
            Normalize(method),
            data,
            input_core_dims=[['Y','X']],
            output_core_dims=[['Y','X']],
            kwargs={**kwargs},
            vectorize=True
        )
        return rescaled

    @staticmethod
    def mask_to_keep(data: Union[xr.Dataset, xr.DataArray], 
                     mask: xr.DataArray,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Mask data by setting values outside the mask to NaN.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset
            Data to be masked.
        mask : xr.DataArray
            Mask to be applied. mask's dimensions must all be found in 
            data, and must not contain 'C' (channel) dimension.
        """
        return data.where(mask.sel({d: data.coords[d] for d in mask.dims}))


class ImageReader:
    """
    Read image by wrapping aicsimageio.AICSImage with xarray interface.

    Parameters
    ----------
    channel_names : List[str], optional
        Names of channels to read. If None, all channels will be read.
    bitdepth : int, optional
        Bitdepth of the image.
    rescale : bool, optional
        Whether to rescale the image to [0, 1] (raw value minus 
        background, then devided by 2**bitdepth - 1). If False, the 
        image will be returned as is (background-subtracted).
    background : float, optional
        Background value to subtract from the image. Default is 0.
    dtype : np.dtype, optional
        Data type of the image. Default is np.float64.
    squeeze : bool, optional
        Whether to squeeze the image to remove singleton dimensions.
    """
    def __init__(
        self, 
        channel_names: List[str] = None,
        *, 
        bitdepth: int = None,
        rescale: bool = False,
        background: float = 0,
        dtype: np.dtype = np.float64,
        squeeze: bool = False,
    ) -> None :
        self._channel_names = channel_names
        self._bitdepth = bitdepth
        self._rescale = rescale
        self._background = background
        self._dtype = dtype
        self._squeeze = squeeze


    def read(self, fpath: str, fmt: str) -> Union[xr.DataArray, xr.Dataset]:
        """Read image and return as xarray.DataArray or xarray.Dataset.

        Parameters
        ----------
        fpath : str
            Path to the image file.
        fmt : str
            'DataArray' or 'Dataset'. If Dataset, dimension 'C' will be 
            removed, and each channel will be a variable.

        Returns
        -------
        xr.DataArray or xr.Dataset representing the image.
        """
        if fmt == 'DataArray':
            im = self._read(fpath)
            if im.sizes['C'] == 1:
                im = im.rename(im.coords['C'].item())
        elif fmt == 'Dataset':
            im = self._read(fpath).to_dataset(dim='C')
        else:
            raise ValueError(f"Output format (fmt) `{fmt}` not supported. "
                             f"Supported formats: DataArray, Dataset.")
        
        # squeeze just before returning
        if self._squeeze:
            im = im.squeeze()

        return im


    def _read(self, fpath: str) -> xr.DataArray:

        # read image and subtract background
        im = AICSImage(fpath).xarray_data.astype(self._dtype)
        im -= self._background

        # add channel names if specified
        if self._channel_names is not None:
            im = im.assign_coords({'C': self._channel_names})
        
        # rescale to [0, 1] if specified
        if self._rescale:
            if self._bitdepth is None:
                raise ValueError(
                    "bitdepth must be specified if rescale is True.")
            im /= (2**self._bitdepth - 1)

        # Use integer index as coords for non-singlton dim without 
        # coords
        for d in im.squeeze().dims:
            if d not in im.coords:
                im = im.assign_coords({d: np.arange(im.sizes[d])})

        return im


class Normalize:
    def __init__(self, method: str) -> None:
        if method not in ['zscore', 'pclip']:
            raise ValueError(
                f"Normalizing method `{method}` not supported.")
        elif method == 'pclip':
            self._rescale = self._pclip
        elif method == 'zscore':
            self._rescale = self._zscore

    def __call__(self, data: Union[xr.Dataset, xr.DataArray], **kwargs
    ) -> Union[xr.Dataset, xr.DataArray]:
        return self._rescale(data, **kwargs)
    
    @staticmethod
    def _zscore(data):
        """
        Rescale by z-score standardization.
        """
        return ((data - np.nanmean(data)) / np.nanstd(data))
    
    @staticmethod
    def _pclip(data, p: float = 2.0):
        """
        Rescale to [0, 1] by percentile clipping at p% and (100-p)%.

        Default corresponds to p2 and p98.
        """
        p = np.nanpercentile(data, p)
        q = np.nanpercentile(data, 100-p)
        return (data - p) / (q - p)


class Mask:

    @staticmethod
    def read(fpath: str, fmt: str, 
             channel_names: List[str] = ["mask"], 
             drop_single_C: bool = False, 
             **kwargs,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Read mask and return as xarray.DataArray or xarray.Dataset.

        Parameters
        ----------
        fpath : str
            Path to the mask file.
        fmt : str
            'DataArray' or 'Dataset'. If Dataset, dimension 'C' will be
            removed, and each channel will be a variable.
        channel_names : List[str], optional
            Names of masks corresponding to 'C' dimension of the image.
            Default is ['mask'].
        **kwargs : dict, optional

        Other Parameters
        ----------------
        dtype : np.dtype, optional
            Data type of the image. Default is np.int_.
        squeeze : bool, optional
            Whether to squeeze the image to remove singleton dimensions.
        drop_single_C : bool, optional
            Whether to drop the 'C' dimension if it has only one element.
            Default is False. (This has no effect if fmt is 'Dataset' 
            where 'C' is always dropped as it will be turned into 
            data_vars.)
        """
        reader = MaskReader(channel_names, **kwargs)
        return reader.read(fpath, fmt, drop_single_C=drop_single_C)


    @staticmethod
    def erode_by_disk(mask: Union[xr.DataArray, xr.Dataset], 
                      radius: int
    ) -> Union[xr.DataArray, xr.Dataset]:
        return _apply_ufunc_simple_by_plane(binary_erosion, mask, 
                                            footprint=disk(radius))
    
    @staticmethod
    def dilate_by_disk(mask: Union[xr.DataArray, xr.Dataset], 
                       radius: int
    ) -> Union[xr.DataArray, xr.Dataset]:
        return _apply_ufunc_simple_by_plane(binary_dilation, mask, 
                                            footprint=disk(radius))


class MaskReader(ImageReader):
    def __init__(
        self, 
        channel_names: List[str] = ["mask"], 
        *, 
        dtype: np.dtype = np.int_, 
        squeeze: bool = False
    ) -> None:
        super().__init__(
            channel_names, 
            bitdepth=None, rescale=False, background=0, 
            dtype=dtype, squeeze=squeeze
            )

    def read(
        self, 
        fpath: str, 
        fmt: str, 
        *, 
        drop_single_C: bool = False
    ) -> Union[xr.DataArray, xr.Dataset]:
        im = super().read(fpath, fmt)

        # If there's only one channel (mask object), and the output 
        # format is DataArray, drop the dimension 'C'.
        if drop_single_C:
            if fmt == 'DataArray' and im.coords['C'].size == 1:
                im = im.drop_vars('C')

        return im

def _apply_ufunc_simple_by_plane(
    f: Callable[[Union[xr.DataArray, xr.Dataset]], 
                Union[xr.DataArray, xr.Dataset]],
    data: Union[xr.DataArray, xr.Dataset], 
    **kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    """Wrapper for applying a function to each XY-plane.

    This is a wrapper for xarray.apply_ufunc with some assumptions and
    defaults that are useful for applying a function to every XY-plane.
    The function f takes a 2D array as its first argument, and does not
    require another DataArray or Dataset as its second argument. It can
    take other arguments to be passed in as kwargs. 

    Returns
    -------
    xr.DataArray
    """
    return xr.apply_ufunc(
        f,
        data,
        input_core_dims=[['Y', 'X']],
        output_core_dims=[['Y', 'X']],
        vectorize=True,
        kwargs=kwargs,
    )