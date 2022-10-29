import xarray as xr
from data.echogram import DataReaderZarr


# Load predictions

# predictions = xr.open_zarr('2019/S2019847/ACOUSTIC/GRIDDED/S2019847_sv.zarr')

# Load data

reader = DataReaderZarr('/data/2019/S2019847/ACOUSTIC/GRIDDED/S2019847_sv.zarr')

# Print raw files included in the survey

print(reader.raw_file_included)

# Visualize a raw file

reader.visualize(raw_file="2019847-D20190512-T194702.raw", draw_seabed=True)