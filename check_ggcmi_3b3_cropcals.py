# %%
# pylint: disable=invalid-name
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods

import os
import glob
import logging

import numpy as np
import xarray as xr

os.chdir("/glade/work/samrabin/ggcmi_3b3_cropcals/files")
outdir = os.path.join(os.getcwd(), os.pardir, "outputs")
txtdir = os.path.join(outdir, "txt")
if not os.path.exists(txtdir):
    os.makedirs(txtdir)

indent = "   "
decade_indent = 2 * indent


# %% Classes


class Decade:
    def __init__(self, timeinfo, d):
        self.end = timeinfo.dec_ends[d]
        self.start = timeinfo.dec_starts[d]
        self.str = f"{self.start}-{self.end}"


class ProblemsFound:
    def __init__(self):
        self.file = False
        self.var = False


class TimeInfo:
    def __init__(self, ds):
        self.years = ds["time"].values
        self.dec_starts = np.array(
            [
                int(x)
                for x in np.unique(np.floor(self.years / 10)) * 10 + 1
                if x < max(self.years) + 1
            ]
        )
        self.dec_ends = self.dec_starts + 9
        # Restrict values to years actually in the file
        self.dec_starts[0] = max(self.dec_starts[0], min(self.years))
        self.dec_ends[-1] = min(self.dec_ends[-1], max(self.years))


# %% Functions


def open_dataset(file):
    """
    Open the dataset without decoding times, because cftime has issues with ISIMIP's format
    """
    ds = xr.open_dataset(file, decode_times=False)

    # Process time units
    time_units_in = ds["time"].attrs["units"]
    if time_units_in == "years since 1601-1-1 00:00:00":
        new_times = ds["time"].values + 1601
        new_time_da = xr.DataArray(
            data=new_times.astype(int), dims=["time"], attrs={"units": "year"}
        )
        ds["time"] = new_time_da
    else:
        raise RuntimeError("Unknown time units: " + time_units_in)
    return ds


def log_decade_problem(decade, mask_constant, da_min, da_max, values_constant):
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
    logging.warning("%s%s: %s", decade_indent, decade.str, msg)


def process_decade(
    file,
    years,
    logger,
    logfile,
    pf,
    this_var,
    var_indent,
    da,
    decade,
):
    where_this_decade = np.where((years >= decade.start) & (years <= decade.end))[0]
    if len(where_this_decade) == 0:
        logging.error("No years found in %s!!!", decade.str)
    da_decade = da.isel(time=where_this_decade)

    # NaN masks
    mask_min = np.isnan(da_decade).min(dim="time")
    mask_max = np.isnan(da_decade).max(dim="time")
    mask_constant = mask_min == mask_max
    always_nan = mask_constant & (mask_max == 1)

    da_min = da_decade.min(dim="time", skipna=True)
    da_max = da_decade.max(dim="time", skipna=True)
    values_constant = (da_min == da_max) | always_nan

    if np.any(~(mask_constant & values_constant)):
        if not pf.file:
            logger = set_up_logger(logfile)
            pf.file = True
            logging.warning(file)
        if not pf.var:
            pf.var = True
            logging.warning("%s%s", var_indent, this_var)
        log_decade_problem(decade, mask_constant, da_min, da_max, values_constant)
    return logger, pf


def set_up_logger(logfile):
    if not os.path.exists(os.path.dirname(logfile)):
        os.makedirs(os.path.dirname(logfile))
    logger = logging.FileHandler(
        filename=logfile,
        mode="w",
    )
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(message)s")
    logger.setFormatter(formatter)
    logging.getLogger().addHandler(logger)
    return logger


# %% Loop through all files


def main():
    # pylint: disable=too-many-nested-blocks
    file_list = glob.glob("*/*.nc")
    file_list.sort()
    n_files = len(file_list)

    pf = ProblemsFound()

    for f, file in enumerate(file_list):
        print(f"{f+1}/{n_files}: {file}")

        # Open the file
        ds = open_dataset(file)

        # Get all decades in this file
        timeinfo = TimeInfo(ds)

        # Set up log file
        logfile = os.path.join(txtdir, file).replace(".nc", ".txt")
        if os.path.exists(logfile):
            os.remove(logfile)

        # Loop through all variables
        var_list = list(ds.keys())
        var_list.sort()
        pf.file = False
        logger = None
        for this_var in var_list:
            var_indent = 1 * indent
            da = ds[this_var]
            pf.var = False

            # Check variable for constancy within each decade
            for d in np.arange(len(timeinfo.dec_ends)):
                decade = Decade(timeinfo, d)

                # Get subset of Dataset for this decade
                logger, pf = process_decade(
                    file,
                    timeinfo.years,
                    logger,
                    logfile,
                    pf,
                    this_var,
                    var_indent,
                    da,
                    decade,
                )
        if logger:
            logging.getLogger().removeHandler(logger)
            logger.close()


main()
