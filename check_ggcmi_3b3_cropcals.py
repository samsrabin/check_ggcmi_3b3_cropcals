# %%
# pylint: disable=invalid-name

import os
import glob
import logging
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr

os.chdir("/glade/work/samrabin/ggcmi_3b3_cropcals/files")
outdir = os.path.join(os.getcwd(), os.pardir, "outputs")
txtdir = os.path.join(outdir, "txt")
if not os.path.exists(txtdir):
    os.makedirs(txtdir)


# %% Test with one file

file = "ssp126soc-adapt/ggcmi-crop-calendar_ukesm1-0-ll_ssp126_swh-firr_annual_2015_2100.nc"
# file = "ssp126soc-adapt/ggcmi-crop-calendar_ukesm1-0-ll_ssp126_sun-firr_annual_2015_2100.nc"

ds = xr.open_dataset(file, decode_times=False)

time_units_in = ds["time"].attrs["units"]
if time_units_in == "years since 1601-1-1 00:00:00":
    new_times = ds["time"].values + 1601
    new_time_da = xr.DataArray(
        data=new_times.astype(int),
        dims=["time"],
        attrs={
            "units": "year"
        }
    )
    ds["time"] = new_time_da
else:
    raise RuntimeError("Unknown time units: " + time_units_in)

# %%

years = ds["time"].values
decade_starts = np.array([
    int(x) for x in np.unique(np.floor(years / 10)) * 10 + 1 if x < max(years)+1
])
decade_ends = decade_starts + 9

indent = "   "
logfile = os.path.join(txtdir, file).replace(".nc", ".txt")
if os.path.exists(logfile):
    os.remove(logfile)

var_list = list(ds.keys())
var_list.sort()

# Check each variable for constancy
file_problem_found = False
for this_var in var_list:
    var_indent = 1 * indent
    da = ds[this_var]
    var_problem_found = False
    for d, decade_end in enumerate(decade_ends):
        decade_start = max(decade_starts[d], min(years))
        decade_indent = 2 * indent
        decade_str = f"{decade_start}-{decade_end}"

        # Get subset of Dataset for this decade
        where_this_decade = np.where((years >= decade_start) & (years <= decade_end))[0]
        if len(where_this_decade) == 0:
            logging.error("No years found in %s!!!", decade_str)
        da_decade = da.isel(time=where_this_decade)

        # NaN masks
        mask_min = np.isnan(da_decade).min(dim="time")
        mask_max = np.isnan(da_decade).max(dim="time")
        mask_constant = mask_min == mask_max
        always_nan = mask_constant & (mask_max == 1)

        da_min = da_decade.min(dim="time", skipna=True)
        da_max = da_decade.max(dim="time", skipna=True)
        values_constant = (da_min == da_max) | always_nan
        # values_constant = np.isclose(da_min, da_max) | always_nan

        if np.any(~(mask_constant & values_constant)):
            if not file_problem_found:
                if not os.path.exists(os.path.dirname(logfile)):
                    os.makedirs(os.path.dirname(logfile))
                logging.basicConfig(
                    filename=logfile,
                    level=logging.INFO,
                    format='%(message)s',
                    filemode='w')
                file_problem_found = True
                logging.warning(file)
            if not var_problem_found:
                var_problem_found = True
                logging.warning("%s%s", var_indent, this_var)
            masks_vary = np.any(~mask_constant)
            nonnan_values_vary = np.any(~values_constant)
            if nonnan_values_vary:
                max_diff = np.nanmax((da_max - da_min).values)
                if masks_vary:
                    msg = "Masks and non-NaN values vary"
                else:
                    msg = "Only non-NaN values vary"
                msg += f"; max diff {max_diff}"
            else:
                if not masks_vary:
                    raise RuntimeError("???")
                msg = "Only masks vary"
            logging.warning("%s%s: %s", decade_indent, decade_str, msg)

            # (~(mask_constant & values_constant)).plot()
            # plt.show()
            # stopping
