# aind-ophys-oasis-event-detection

This capsule uses OASIS to extract neural activity from fluorescence imaging traces through nonnegative deconvolution.

## Input

All parameters are passed to run_capsule.py using `python run_capsule.py [parameters]`.
All parameters are defined in __main__ using argparse. The most important one is 'input-dir' 
which should point to a directory containing file `dff.h5` with the dataset 'data', a 2D array 
of $\Delta F/F$ traces, and file `processing.json` to obtain the frame rate. 

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
