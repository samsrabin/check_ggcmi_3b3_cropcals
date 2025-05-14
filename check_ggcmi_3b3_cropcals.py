# %%
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
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


class CropcalFile:
    def __init__(self, txtdir, file):
        self.file = file
        self.logfile = os.path.join(txtdir, self.file).replace(".nc", ".txt")
        self.pf = ProblemsFound()
        self.logger = None
        self.ds = None
        self.ok_ranges = None
        self.timeinfo = None

    def __enter__(self):

        # Open dataset
        self.open_dataset()
        self.ok_ranges = get_ok_ranges(self.ds.attrs)

        # Get all decades in this file
        self.timeinfo = TimeInfo(self.ds)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the logfile, if it was ever opened
        """
        if self.logger:
            logging.getLogger().removeHandler(self.logger)
            self.logger.close()

    def process(self):
        """
        Loop through and test all variables
        """

        # Get variable info
        var_list = list(self.ds.keys())
        var_list.sort()

        for this_var in var_list:
            da = self.ds[this_var]
            self.pf.var = False

            # Check variable for acceptable bounds
            self.check_var_range(da)

            if self.ds.attrs["crop"] in GS_ALGO_CROP_LIST:
                # Check variable for constancy within each decade
                self.check_constancy_within_decades(da)
                # Check that variable does change over time
                if this_var not in ["planting_season"]:
                    self.check_constancy(da, expect_const=False)
            else:
                # Check constancy across entire file
                self.check_constancy(da)

    def warn(self, *args, this_var=None):
        if not self.pf.file:
            self.logger = set_up_logger(self.logfile)
            self.pf.file = True
            logging.warning(self.file)
        if this_var is not None and not self.pf.var:
            self.pf.var = True
            logging.warning("%s%s", VAR_INDENT, this_var)
        logging.warning(*args)

    def open_dataset(self):
        """
        Open the dataset without decoding times, because cftime has issues with ISIMIP's format
        """
        self.ds = xr.open_dataset(self.file, decode_times=False)

        # Set up log file
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

        # Process time units
        time_units_in = self.ds["time"].attrs["units"]
        if time_units_in == "years since 1601-1-1 00:00:00":
            new_times = self.ds["time"].values + 1601
            new_time_da = xr.DataArray(
                data=new_times.astype(int), dims=["time"], attrs={"units": "year"}
            )
            self.ds["time"] = new_time_da
        else:
            raise RuntimeError("Unknown time units: " + time_units_in)

        # Get crop info
        if "Crop" in self.ds.attrs:
            crop = self.ds.attrs["Crop"][0:3]
        else:
            self.warn("%sAttribute 'Crop' missing; getting from filename", VAR_INDENT)
            crop = os.path.basename(self.file).split("_")[3][0:3]
        if crop not in CROP_LIST:
            raise ValueError(f"Unexpected crop: {crop}")
        self.ds.attrs["crop"] = crop
        self.ds.attrs["gs_algo"] = crop in GS_ALGO_CROP_LIST

    def check_var_range(self, da):
        """
        Check whether variable's data is within acceptable bounds
        """
        this_var = da.name
        ok_range = self.ok_ranges[this_var]
        if ok_range is None:
            return

        ok_min = ok_range[0]
        ok_max = ok_range[1]
        this_min = np.nanmin(da)
        this_max = np.nanmax(da)
        too_low = this_min < ok_min
        too_high = this_max > ok_max
        if too_low or too_high:
            ok_range_str = f"{ok_min}-{ok_max}"
            this_range_str = f"{this_min}-{this_max}"
            self.warn(
                f"{PROBLEM_INDENT}Variable range {this_range_str} outside limits ({ok_range_str})",
                this_var=this_var,
            )

    def check_constancy_within_decades(self, da):
        """
        Check that data does not vary over time within each decade
        """
        for d in np.arange(len(self.timeinfo.dec_ends)):
            decade = Decade(self.timeinfo, d)

            # Get subset of Dataset for this decade
            where_this_decade = np.where(
                (self.timeinfo.years >= decade.start)
                & (self.timeinfo.years <= decade.end)
            )[0]
            if len(where_this_decade) == 0:
                logging.error("No years found in %s!!!", decade.str)
            da_decade = da.isel(time=where_this_decade)

            # Check constancy in this decade
            self.check_constancy(da_decade, decade=decade)

    def check_constancy(self, da, decade=None, expect_const=True):
        """Check whether a DataArray varies over time

        Args:
            da (xarray.DataArray): DataArray to check
            decade (Decade, optional): Decade to check. Defaults to None.
            expect_const (bool, optional): Whether we expect the variable to be constant over time.
                                           Defaults to True.

        Raises:
            NotImplementedError: Haven't yet implemented error message
        """
        this_var = da.name
        mask_min = np.isnan(da).min(dim="time")
        mask_max = np.isnan(da).max(dim="time")
        mask_constant = mask_min == mask_max
        always_nan = mask_constant & (mask_max == 1)

        da_min = da.min(dim="time", skipna=True)
        da_max = da.max(dim="time", skipna=True)
        values_constant = (da_min == da_max) | always_nan

        is_not_constant = np.any(~(mask_constant & values_constant))
        if is_not_constant and expect_const:
            if decade is not None:
                msg = get_decade_warning(
                    decade, mask_constant, da_min, da_max, values_constant
                )
                self.warn(msg, this_var=this_var)
            else:
                raise NotImplementedError(
                    "Expected entire file to be constant but it's not; log this"
                )
        elif (not is_not_constant) and (not expect_const):
            self.warn(
                "%sExpected variation across time but found none",
                PROBLEM_INDENT,
                this_var=this_var,
            )


# %% Functions


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


def get_decade_warning(decade, mask_constant, da_min, da_max, values_constant):
    """
    Get a warning message for when data unexpectedly varies within a decade
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
    return f"{PROBLEM_INDENT}{decade.str}: {msg}"


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
    file_list = glob.glob("**/*.nc", recursive=True)
    file_list.sort()
    n_files = len(file_list)

    for f, file in enumerate(file_list):
        print(f"{f+1}/{n_files}: {file}")

        # Process the file
        with CropcalFile(txtdir, file) as f:
            f.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in-dir",
        help="Directory containing the input files",
        required=True,
        type=os.path.abspath,
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        help="Where to save the output files",
        required=True,
        type=os.path.abspath,
    )

    main(parser.parse_args())
