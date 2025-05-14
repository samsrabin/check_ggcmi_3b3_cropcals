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
import argparse

import numpy as np
import xarray as xr

# Indentation for logfile messages
INDENT = "   "
VAR_INDENT = 1 * INDENT
PROBLEM_INDENT = 2 * INDENT

# List of crops that we expect to have had the growing season algorithm applied
GS_ALGO_CROP_LIST = [
    "mai",
    "ri1",
    "sor",
    "soy",
    "swh",
    "wwh",
]

# List of all crops
CROP_LIST = GS_ALGO_CROP_LIST + [
    "cas",
    "mil",
    "nut",
    "pea",
    "rap",
    "ri2",
    "sgb",
    "sgc",
    "sun",
]


# %% Classes


class Decade:
    """
    A simple class that helps us pass decade info more simply
    """

    def __init__(self, timeinfo, d):
        self.end = timeinfo.dec_ends[d]
        self.start = timeinfo.dec_starts[d]
        self.str = f"{self.start}-{self.end}"


class ProblemsFound:
    """
    A simple class to keep track of whether the current file or variable have had problems found
    """

    def __init__(self):
        self.file = False
        self.var = False


class TimeInfo:
    """
    A simple class that helps us pass time info more simply
    """

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


def open_dataset(file, logger, pf, txtdir):
    """
    Open the dataset without decoding times, because cftime has issues with ISIMIP's format
    """
    ds = xr.open_dataset(file, decode_times=False)

    # Set up log file
    logfile = os.path.join(txtdir, file).replace(".nc", ".txt")
    if os.path.exists(logfile):
        os.remove(logfile)

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

    # Get crop info
    if "Crop" in ds.attrs:
        crop = ds.attrs["Crop"][0:3]
    else:
        logger, pf = problem_setup(file, logger, logfile, pf)
        logging.warning("%sAttribute 'Crop' missing; getting from filename", VAR_INDENT)
        crop = os.path.basename(file).split("_")[3][0:3]
    if crop not in CROP_LIST:
        raise ValueError(f"Unexpected crop: {crop}")
    ds.attrs["crop"] = crop
    ds.attrs["gs_algo"] = crop in GS_ALGO_CROP_LIST

    return ds, pf, logfile, logger


def get_ok_ranges(attrs):
    """Get acceptable ranges for variables, depending on the crop

    Args:
        attrs (dict): Dataset attributes

    Returns:
        dict: Acceptable ranges for each variable. None: Variable can take any value.
    """

    # ri2 can have date 0, which means it's not actually present
    if attrs["crop"] == "ri2":
        days_in_year = [0, 365]
    else:
        days_in_year = [1, 365]

    # Harvest reason is 7 if growing season algorithm isn't applied
    if attrs["gs_algo"]:
        harvest_reason_range = [1, 6]
    else:
        harvest_reason_range = [7, 7]

    ok_ranges = {
        "growing_period": None,
        "harvest_reason": harvest_reason_range,
        "maturity_day": days_in_year,
        "planting_day": days_in_year,
        "planting_day-mavg": days_in_year,
        "planting_day-mavg-window": None,
        "planting_season": [1, 2],
        "seasonality": None,
    }
    return ok_ranges


def check_constancy(
    file, logger, logfile, pf, this_var, da, decade=None, expect_const=True
):
    """Check whether a DataArray varies over time

    Args:
        file (str): Path to Dataset file
        logger (logging.FileHandler): For writing to this Dataset's output .txt file
        logfile (str): Path to this Dataset's output .txt file
        pf (ProblemsFound)
        this_var (str): Name of the variable to check
        da (xarray.DataArray): DataArray to check
        decade (Decade, optional): Decade to check. Defaults to None.
        expect_const (bool, optional): Whether we expect the variable to be constant over time.
                                       Defaults to True.

    Raises:
        NotImplementedError: Haven't yet implemented error message
    """
    mask_min = np.isnan(da).min(dim="time")
    mask_max = np.isnan(da).max(dim="time")
    mask_constant = mask_min == mask_max
    always_nan = mask_constant & (mask_max == 1)

    da_min = da.min(dim="time", skipna=True)
    da_max = da.max(dim="time", skipna=True)
    values_constant = (da_min == da_max) | always_nan

    is_not_constant = np.any(~(mask_constant & values_constant))
    if is_not_constant and expect_const:
        logger, pf = problem_setup(file, logger, logfile, pf, this_var)
        if decade is not None:
            log_decade_problem(decade, mask_constant, da_min, da_max, values_constant)
        else:
            raise NotImplementedError(
                "Expected entire file to be constant but it's not; log this"
            )
    elif (not is_not_constant) and (not expect_const):
        logger, pf = problem_setup(file, logger, logfile, pf, this_var)
        logging.warning(
            "%sExpected variation across time but found none", PROBLEM_INDENT
        )
    return logger, pf


def check_var_range(ok_range, pf, file, logfile, logger, this_var, da):
    """
    Check whether variable's data is within acceptable bounds
    """
    ok_min = ok_range[0]
    ok_max = ok_range[1]
    this_min = np.nanmin(da)
    this_max = np.nanmax(da)
    too_low = this_min < ok_min
    too_high = this_max > ok_max
    if too_low or too_high:
        # pylint: disable=logging-fstring-interpolation
        logger, pf = problem_setup(file, logger, logfile, pf, this_var)
        logging.warning(
            f"{PROBLEM_INDENT}Variable range {this_min}-{this_max} outside limits ({ok_min}-{ok_max})"
        )
    return logger, pf


def log_decade_problem(decade, mask_constant, da_min, da_max, values_constant):
    """
    Log a warning message when data unexpectedly varies within a decade
    """
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
    logging.warning("%s%s: %s", PROBLEM_INDENT, decade.str, msg)


def check_constancy_within_decades(file, logfile, timeinfo, this_var, da, logger, pf):
    """
    Check that data does not vary over time within each decade
    """
    for d in np.arange(len(timeinfo.dec_ends)):
        decade = Decade(timeinfo, d)

        # Get subset of Dataset for this decade
        where_this_decade = np.where(
            (timeinfo.years >= decade.start) & (timeinfo.years <= decade.end)
        )[0]
        if len(where_this_decade) == 0:
            logging.error("No years found in %s!!!", decade.str)
        da_decade = da.isel(time=where_this_decade)

        # Check constancy in this decade
        logger, pf = check_constancy(
            file, logger, logfile, pf, this_var, da_decade, decade
        )

    return logger, pf


def problem_setup(file, logger, logfile, pf, this_var=None):
    """
    Set up the log file and print initial messages, if needed
    """
    if not pf.file:
        logger = set_up_logger(logfile)
        pf.file = True
        logging.warning(file)
    if this_var is not None and not pf.var:
        pf.var = True
        logging.warning("%s%s", VAR_INDENT, this_var)
    return logger, pf


def set_up_logger(logfile):
    """
    Set up the log file
    """
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


def main(args):
    # pylint: disable=too-many-nested-blocks

    # Set up directories
    os.chdir(args.in_dir)
    txtdir = os.path.join(args.out_dir, "txt")
    if not os.path.exists(txtdir):
        os.makedirs(txtdir)

    # Get list of input files
    file_list = glob.glob("*/*.nc")
    file_list.sort()
    n_files = len(file_list)

    pf = ProblemsFound()

    for f, file in enumerate(file_list):
        print(f"{f+1}/{n_files}: {file}")

        # Open the file
        pf.file = False
        logger = None
        ds, pf, logfile, logger = open_dataset(file, logger, pf, txtdir)
        ok_ranges = get_ok_ranges(ds.attrs)

        # Get all decades in this file
        timeinfo = TimeInfo(ds)

        # Loop through all variables
        var_list = list(ds.keys())
        var_list.sort()
        for this_var in var_list:
            da = ds[this_var]
            pf.var = False

            # Check variable for acceptable bounds
            if ok_ranges[this_var] is not None:
                logger, pf = check_var_range(
                    ok_ranges[this_var],
                    pf,
                    file,
                    logfile,
                    logger,
                    this_var,
                    da,
                )

            if ds.attrs["crop"] in GS_ALGO_CROP_LIST:
                # Check variable for constancy within each decade
                logger, pf = check_constancy_within_decades(
                    file, logfile, timeinfo, this_var, da, logger, pf
                )
                # Check that variable does change over time
                if this_var not in ["planting_season"]:
                    logger, pf = check_constancy(
                        file, logger, logfile, pf, this_var, da, expect_const=False
                    )
            else:
                # Check constancy across entire file
                logger, pf = check_constancy(file, logger, logfile, pf, this_var, da)

        # Close the logfile, if it was ever opened
        if logger:
            logging.getLogger().removeHandler(logger)
            logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in-dir",
        help="Directory containing the input files",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        help="Where to save the output files",
        required=True,
    )

    my_args = parser.parse_args()
    my_args.in_dir = os.path.abspath(my_args.in_dir)
    my_args.out_dir = os.path.abspath(my_args.out_dir)

    main(my_args)
