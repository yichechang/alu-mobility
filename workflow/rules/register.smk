"""
"""

rule register_nucleus:
    input:
        multi="results/image/multi_ch/{RoiUID}.ome.tif",
    output:
        multi="results/image_registered/multi_ch/{RoiUID}.ome.tif",
    params:
        fluoros=[ch['fluoro'] for ch in config['input']['channels']],
        bitdepth=config['input']['bitdepth'],
        registers=config['register_nucleus']['registers'],
        transformation=config['register_nucleus']['transformation'],
        mode=config['register_nucleus']['mode'],
    run:
        import xarray as xr
        from pystackreg import StackReg
        from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
        from abcdcs import imageop

        transformations = {
            'TRANSLATION': StackReg.TRANSLATION,
            'RIGID_BODY': StackReg.RIGID_BODY,
            'SCALED_ROTATION': StackReg.SCALED_ROTATION,
            'AFFINE': StackReg.AFFINE,
            'BILINEAR': StackReg.BILINEAR
        }

        images = (
            imageop.Image.read(
                input.multi,
                fmt='DataArray',
                channel_names=params.fluoros,
                bitdepth=params.bitdepth,
            )
            .squeeze()
        )
        
        def _dims_T_first(da):
            all_dims = list(da.dims)
            remaining_dims = [dim for dim in all_dims if dim != 'T']
            return ['T'] + remaining_dims


        def register(image, target, by):
            # prepare data to feed pystackreg: numpy with T being first dim
            image_noC = image.isel(C=0).drop_vars('C')
            dims_noC_firstT = _dims_T_first(image_noC)
            target_data = image.sel(C=target).transpose(*dims_noC_firstT).data
            by_data = image.sel(C=by).transpose(*dims_noC_firstT).data

            # actual part for registering
            sr = StackReg(transformations[params.transformation])
            tmats = sr.register_stack(by_data, reference=params.mode)
            reg = sr.transform_stack(target_data)
            
            # assemble a dataarray for registered result
            image_noC_firstT = image_noC.transpose(*dims_noC_firstT)
            dims = image_noC_firstT.dims
            coords = image_noC_firstT.coords
            # assemble dataarray to have no C and T being first dim
            da = xr.DataArray(reg, coords=coords, dims=dims)
            # return to the original dim order
            da = da.transpose(*image_noC.dims)
            # assign C coords value
            da = da.assign_coords(C=target)
            return da

        registered = []
        for r in params.registers:
            registered.append(register(images, target=r['target'], by=r['by']))
        registered_da = xr.concat(registered, dim='C').transpose(*images.dims)
        
        OmeTiffWriter.save(registered_da.data, output.multi, dim_order='TCYX')


all_register_input = [
    lambda w: expand(
        "results/image_registered/multi_ch/{RoiUID}.ome.tif",
        RoiUID=get_checkpoint_RoiUID(w)
    ),
]




