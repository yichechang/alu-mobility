from typing import List, Tuple, Optional, Union, Dict, Callable, Any

from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray, Dataset
from skimage.filters import gaussian

from abcdcs import imageop, loadpiv, pivop, msnd

def _blur_image(image: DataArray, sigma: float = 0.5
                ) -> DataArray:
    return xr.apply_ufunc(
        gaussian,
        image,
        input_core_dims=[['Y', 'X']],
        output_core_dims=[['Y', 'X']],
        vectorize=True,
        kwargs={'sigma': sigma}
    )

class DataReader:
    """
    Parameters
    ----------
    data_name_and_paths: dict
        snakemake.input
    data_configs: dict
        the "data" block in rule msnd's config.
        key = name. value = dict(path_pattern=..., type=...)
    chnames: list[str]
    """
    def __init__(self, data_name_and_paths, data_configs, chnames):
        self._name_and_paths = data_name_and_paths
        self._configs = data_configs
        self._chnames = chnames

    def _data_type_to_func(self, data_type):
        mapping = {
            "image":
                partial(imageop.Image.read,
                        fmt="DataArray", channel_names=self._chnames, squeeze=True),
            "mask":
                partial(imageop.Mask.read,
                        fmt="DataArray", squeeze=True, drop_single_C=True),
            "piv":
                loadpiv.read_matpiv,
        }
        return mapping[data_type]

    def read(self):
        data = {}
        for name,path in self._name_and_paths.items():
            data_type = self._configs[name]['type']
            data_reader = self._data_type_to_func(data_type)
            data[name] = data_reader(path)

        return data


class PreprocessStepHandler:
    """
    Prepare proper step to be consumed by _apply_steps

    - map function name to actual function
    - change param/value if for reference

    Example YAML block:
    # - func: datatype_funcname
    #   params:
    #     param1: value1       # regular param/value
    #     _param2: value2     # underscore for `param2=data[value2]`

    """

    def _func_name_to_func(self):
        mapping = {
            "piv": {
                'mask_filter': pivop.mask_filter,
                'global_filter': pivop.global_filter,
            },
            "mask": {
                'erode_by_disk': imageop.Mask.erode_by_disk,
            },
            "image": {
                'blur': _blur_image,
            }
        }
        return mapping[self._func_kind][self._func_name]

    def __init__(self, *, step: Any, data: Dict):
        self._step = step
        self._data = data
        self._func_kind_and_name = self._step['func']
        self._params = self._step['params']

    def __call__(self):
        return self.func, self.params

    @property
    def func(self):
        self._parse_func_kind_and_name()
        return self._func_name_to_func()

    @property
    def params(self):
        return self._replace_params_using_data()

    def _parse_func_kind_and_name(self):
        # Extract data type and actual function name by
        # splitting the string at the first '_'
        parts = self._func_kind_and_name.split('_', 1)
        self._func_kind = parts[0]
        self._func_name = parts[1]

    def _replace_params_using_data(self):
        processed_params = {}
        for param, val in self._params.items():
            if param.startswith('_'):
                processed_params[param[1:]] = self._data[val]
            else:
                processed_params[param] = val
        return processed_params


class PreprocessPipe:
    # no steps will use newly calculated results as non-first param

    # usage: updated_data_dict = PreprocessPipe(pipe_def, data_dict)()
    # usage: data_dict[output_name] = PreprocessPipe(pipe_def, data_dict).output

    def __init__(self, pipe_dict, data):
        self._data = data

        # extract from string-based pipe definition
        self._name = pipe_dict["name"]
        self._input = pipe_dict["input"]
        self._output = pipe_dict["output"]
        self._steps = pipe_dict["steps"]

        self._result = None

    def __call__(self):
        self._data[self._output] = self.output

    @property
    def output(self):
        if self._result is None:
            self.preprocess()
        return self._result

    @property
    def input(self):
        return self._data[self._input]

    @property
    def steps(self):
        return [PreprocessStepHandler(step=step, data=self._data)()
                for step in self._steps]

    def preprocess(self):
        for step_func, step_params in self.steps:
            if self._result is None:
                self._result = step_func(self.input, **step_params)
            else:
                self._result = step_func(self.output, **step_params)


class MSNDStepFuncHandler:
    @staticmethod
    def _msnd_normal(
            MSNDobj: msnd.MSND
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return MSNDobj.calculate('normal')

    @staticmethod
    def _msnd_eachlevel(
            MSNDobj: msnd.MSND,
            *,
            byimage: DataArray,
            bins: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return MSNDobj.calculate('eachlevel',
                                 byimage=byimage, bins=bins)

    @staticmethod
    def _msnd_eachlevel2d(
            MSNDobj: msnd.MSND,
            *,
            byimage: DataArray,
            bins: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return MSNDobj.calculate('eachlevel2d',
                                 byimage=byimage, bins=bins)

    @staticmethod
    def __call__(step_func) -> Callable[..., Tuple[pd.DataFrame, pd.DataFrame]]:
        _mapping = {
            '0d': MSNDStepFuncHandler._msnd_normal,
            '1d': MSNDStepFuncHandler._msnd_eachlevel,
            '2d': MSNDStepFuncHandler._msnd_eachlevel2d,
        }
        return _mapping[step_func]


class MSNDStepParamsHandler:
    @staticmethod
    def bins_args(bins_args):
        start = bins_args[0]
        end = bins_args[1]
        steps = int((end - start) / bins_args[2] + 1)
        bins = np.linspace(start, end, steps)
        return {'bins': bins}

    @staticmethod
    def __call__(
            step_func: str,
            step_params: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        klass = MSNDStepParamsHandler
        mapping = {
            '0d': {

            },
            '1d': {
                'bins_args': klass.bins_args,
            },
            '2d': {
                'bins_args': klass.bins_args,
            }
        }

        processed_params = {}
        for param, val in step_params.items():
            if param in mapping[step_func]:
                proc = mapping[step_func][param]
                processed_params.update(proc(val))
            else:
                processed_params.update({param: val})

        return processed_params


class MSNDStepHandler:

    def __init__(self, step: Dict[str, Any]):
        self._func: str = step['name']
        self._params: Dict[str, Any] = step['params']

    @property
    def func(self):
        return MSNDStepFuncHandler()(self._func)

    @property
    def params(self):
        return MSNDStepParamsHandler()(self._func, self._params)

    def __call__(self):
        return self.func, self.params


class MSNDPipeline:
    """
    Consume data and config to calculate MSND

    1. preprocess data
    2. calculate MSND

    Usage
    -----
    MSNDPipeline(data, preprocess_pipes, msnd_step)()

    Attributes
    ----------
    data : Dict[str, Union[xr.DataArray, xr.Dataset]]
        Input data to be processed
    preprocess_pipes : Optional[List[Dict[str, Any]]]
        List of pipe definitions to preprocess data
    msnd_step : Dict[str, Any]
        MSND step definition

    Properties
    ----------
    result
        Return MSND results
    preprocessed_data


    Methods
    -------
    preprocess()
        Run preprocessing steps
    process()
        Run MSND calculation
    calculate() -> Tuple[DataFrame, DataFrame]
        Return MSND results
    """

    def __init__(
            self,
            data: Dict[str, Union[DataArray, Dataset]],
            preprocess_pipes: Optional[List[Dict[str, Any]]],
            msnd_step: Dict[str, Any]
    ) -> None:
        self._data = self._preprocessed_data = data
        self._preprocessed = False
        self._preprocess_pipes = preprocess_pipes
        self._msnd_step = msnd_step

        self._result: Tuple[pd.DataFrame, pd.DataFrame] = None
        self._processed = False

    def calculate(self):
        return self.result

    @property
    def result(self):
        if not self._processed:
            self.process()
        return self._result

    @property
    def preprocessed_data(self):
        if not self._preprocessed:
            self.preprocess()
        return self._preprocessed_data

    def preprocess(self):
        # make sure in config there's piv_final and image_final as
        # output names
        if self._preprocess_pipes is not None:
            for pipe in self._preprocess_pipes:
                PreprocessPipe(pipe, self._preprocessed_data)()

        # piv_final and image_final need to exist for msnd processing
        for prefix in ["piv", "image"]:
            try:
                self._preprocessed_data[f"{prefix}_final"]
            except KeyError:
                self._preprocessed_data[f"{prefix}_final"] = self._data[prefix]

        self._preprocessed = True

    def process(self):
        # make sure preprocessing is done
        if not self._preprocessed:
            self.preprocess()

        msnd_func, msnd_params = MSNDStepHandler(self._msnd_step)()
        data = self._data
        MSND = msnd.MSND(data["piv_final"], components=('u', 'v'))
        self._result = msnd_func(MSND, byimage=data["image_final"], **msnd_params)
        self._processed = True