# aind-ophys-oasis-event-detection

This capsule uses OASIS to extract neural activity from fluorescence imaging traces through nonnegative deconvolution.

## Input

All logic lives in the `aind-ophys-oasis-event-detection-library` package; this capsule is a
thin wrapper. Parameters are defined as a `pydantic-settings` model
(`OasisSettings` in the library's `settings.py`) and are passed as
`python run_capsule.py --name=value`, which is the only form a Code Ocean app
panel emits. The most important one is `--input_dir`, which should point to a
directory containing `<experiment_id>/dff/<experiment_id>_dff.h5` with the
dataset `data` (a 2D array of $\Delta F/F$ traces), plus the acquisition
metadata: `acquisition.json` (aind-data-schema v2) or `session.json` (v1),
from which the frame rate is read.

## Output

The main output is the `events_oasis.h5` file. 
It contains datasets: 

`events`:  The deconvolved neural activity ("events" / "spike rates").  
`denoised`: The inferred denoised fluorescence signal.   

If the parameters are automatically estimated, it will also contain the following parameter estimates:

`b_hat`: The estimated fluorescence baseline value.   
`lam_hat`:  The sparsity penalty parameter. Estimated as the optimal Lagrange multiplier for the dual noise constraint problem.   
`tau_hat`:  The estimated exponential decay time based on the data's autocovariance.   
`tau_rise_hat`: Optionally, the exponential rise time. Set to zero by default (i.e., negligible), and thus omitted.
