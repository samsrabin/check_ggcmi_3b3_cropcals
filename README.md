# Check GGCMI's ISIMIP Phase 3b group III crop calendar input files

Things checked for all files:
- File has "Crop" attribute
- Values are within acceptable bounds: all variables except `growing_period`, `planting_day-mavg-window`, and `seasonality`.

Things checked for files of crops that have the growing season algorithm applied (`mai`, `ri1`, `sor`, `soy`, `swh`, `wwh`):
- Every variable is constant within each decade (including both NaN masks and actual values)
- Every variable does vary over time (except for `seasonality`)

Things checked for files of other crops:
- Every variable is constant over time
