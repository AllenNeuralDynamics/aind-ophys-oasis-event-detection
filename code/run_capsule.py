""" pulled from matt davis' mjd_oasis_simple capsule """
import numpy as np
import h5py
from oasis.functions import deconvolve
from pathlib import Path
import logging
import seaborn as sns
import json
import matplotlib.pyplot as plt
import argparse
import os 
from aind_data_schema.core.processing import Processing, DataProcess, ProcessName, PipelineProcess
from typing import Union
from datetime import datetime as dt
from datetime import timezone as tz



def write_output_metadata(
    metadata: dict,
    process_name: str,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    start_date_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    input_fp: str
        path to data input
    output_fp: str
        path to data output
    """
    processing = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="Multplane Ophys Processing Pipeline",
            pipeline_url="https://codeocean.allenneuraldynamics.org/capsule/5472403/tree",
            pipeline_version="0.1.0",
            data_processes=[
                DataProcess(
                    name=process_name,
                    software_version=os.getenv("COMMIT_SHA"),
                    start_date_time=start_date_time,  # TODO: Add actual dt
                    end_date_time=dt.now(tz.utc),  # TODO: Add actual dt
                    input_location=str(input_fp),
                    output_location=str(output_fp),
                    code_url=(os.getenv("REPO_URL")),
                    parameters=metadata,
                )
            ],
        )
    )
    print(f"Output filepath: {output_fp}")
    with open(Path(output_fp).parent.parent / "processing.json", "r") as f:
        proc_data = json.load(f)
    processing.write_standard_file(output_directory=Path(output_fp).parent.parent)
    with open(Path(output_fp).parent.parent / "processing.json", "r") as f:
        dct_data = json.load(f)
    proc_data["processing_pipeline"]["data_processes"].append(
        dct_data["processing_pipeline"]["data_processes"][0]
    )
    with open(Path(output_fp).parent.parent / "processing.json", "w") as f:
        json.dump(proc_data, f, indent=4)

def generate_oasis_events_for_h5_path(
    h5_path: Path,
    expt_id : str,
    out_path: Path,
    trace_key: list = "data",
    estimate_parameters: bool = True,
    qc_plot: bool = True,
    **kwargs,
) -> None:
    """Generate oasis events for all traces in pipe_dev dff folder

    Parameters
    ----------
    h5_path : Path
        Path to h5 file (dff traces h5 from LIMS pipeline)
    expt_id: str
        experiment id
    out_path : Path
        Path to save oasis events
    trace_key : list, optional
        usually "data"
    estimate_parameters : bool, optional
        Whether to estimate parameters (CONSTRAINED AR1)
        or use provided parameters (UNCONSTRAINED AR1)
    qc_plot: bool,
        create qc plot of events
    **kwargs : dict
        UNCONSTRAINED AR1 kwargs
        + tau
        + rate
        + s_min
        CONSTRAINED AR1 kwargs
        + optimize_g
        + penalty
        + g (optimized)
        + sn (optimized)
        + b (optimized)

    Returns
    -------
    oasis_h5 : Path
        Path to oasis events h5
    params: dict
        Dictionary of parameters used for oasis
    """

    # DEFAULT PARAMS
    # TODO: make these as inputs
    params = {}
    params["g"] = (None,)
    params["b"] = None
    params["sn"] = None
    params["optimize_g"] = 0
    params["penalty"] = 1
    params["b_nonneg"] = True
    params["estimate_parameters"] = estimate_parameters
    params["method"] = "constrained_oasisAR1" if estimate_parameters else "unconstrained_oasisAR1"

    try:
        out_path.mkdir(exist_ok=True, parents=True)
        oasis_h5 = out_path / f"{expt_id}_events_oasis.h5"
        print(f"Processing {expt_id}")

        # cehck if oasis h5 exists
        if oasis_h5.exists():
            print(f"{oasis_h5} already exists")
            return

        with h5py.File(h5_path, "r") as f:
            traces = f["data"][:]
            roi_ids = f["roi_names"][:]

        # remove all nans
        nan_inds = ~np.isnan(traces).any(axis=1)
        traces = traces[nan_inds, :]
        roi_ids = roi_ids[nan_inds]

        # rate, timestamp_df = get_correct_frame_rate(expt_id)
        # timestamps = timestamp_df.ophys_frames.values[0]

        ophys_frame_rate = 10.7  # hardcoded in CO since we dont have LIMS or new pipline
        timestamps = (
            np.arange(traces.shape[1]) * 1 / ophys_frame_rate
        )  # dummy CO since we dont have LIMS or new pipline

        params["rate"] = ophys_frame_rate

        # traces, n_nan_list = quick_fix_nans(traces)
        # # warn that some traces have nans
        # if len(n_nan_list) > 0:
        #     print(f"WARNING: {len(n_nan_list)} traces have nans")

        nans = np.where(np.isnan(traces))[0]
        if len(nans) > 0:
            raise ValueError(f"Traces have nans: {len(nans)} in {expt_id}")

        spikes, params = oasis_deconvolve(traces, params, estimate_parameters)

        # save to h5
        with h5py.File(oasis_h5, "w") as file:
            file.create_dataset("cell_roi_id", data=roi_ids)
            file.create_dataset("events", data=spikes)

        if qc_plot:
            plots_path = out_path / "plots"
            plots_path.mkdir(exist_ok=True, parents=True)
            plot_trace_and_events_png(traces, spikes, timestamps, roi_ids, params, plots_path)

        params["events_path"] = str(oasis_h5)
        params["trace_key"] = trace_key
        # params['n_nans'] = n_nan_list

        # dump params to json
        with open(out_path / f"{expt_id}_params.json", "w") as file:
            json.dump(params, file)

        logging.info(f"SUCCESS: {expt_id}")
    except Exception as e:
        logging.error(f"FAILED: {expt_id}")
        raise e
    return oasis_h5, params

def plot_trace_and_events_png(
    traces, spikes, timestamps, roi_ids, params, plots_path, show_fig=False
):
    sns.set_context("talk")
    for i, (spike, trace, cell) in enumerate(zip(spikes, traces, roi_ids)):
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        ax.plot(timestamps, trace, color="g", label="new_dff")

        if params["estimate_parameters"]:
            g = params["g_hat"][i]
        else:
            g = params["g"]
        ax2 = ax.twinx()
        ax2.plot(timestamps, spike * 1, color="orange", label=f"events, g={g}")

        # xlim
        ax.set_xlim(400, 580)
        ax2.set_xlim(400, 580)  # arbitrary time period to check
        ax.legend()
        ax2.legend()
        cell = str(cell)
        ax.set_title(f"cell_roi_id: {cell}")
        fig.savefig(plots_path / f"{cell}_oasis.png")
        if not show_fig:
            plt.close(fig)


def oasis_deconvolve(traces, params: dict, estimate_parameters: bool = True, **kwargs) -> np.array:
    """Deconvolve traces for for all cells in a trace array

    Parameters
    ----------
    traces : np.array
        Array of traces, shape (n_cells, n_frames)
    params : dict
        Dictionary of parameters for oasis
    estimate_parameters : bool, optional
        Whether to estimate parameters, by default True
    **kwargs : dict
        Additional arguments to pass to oasisAR1

    Returns
    -------
    spikes : np.array
        Array of spikes, shape (n_cells, n_frames)

    Notes
    -----
    + See OASIS repo for more parameters

    estimate_parameters=True
    ------------------------
    + penalty: (sparsity penalty) 1: min |s|_1  0: min |s|_0
    + g
    + sn
    + b
    + b_nonneg
    + optimize_g: number of large, isolated events to FURTHER optimize g
    + kwargs

    estimate_parameters=False
    -------------------------
    + tau
    + rate
    + s_min

    """
    if estimate_parameters:
        # check params for required keys
        required_keys = ["g", "sn", "b", "b_nonneg", "optimize_g", "penalty"]
        for key in required_keys:
            if key not in params:
                raise UserWarning(f"params must contain {key}")

    elif not estimate_parameters:
        # check params for required keys
        required_keys = ["tau", "rate", "s_min"]
        for key in required_keys:
            if key not in params:
                raise UserWarning(f"params must contain {key}")

        g = np.exp(-1 / (params["tau"] * params["rate"]))
        lam = 0

        params["g"] = g
        params["lam"] = lam

    # run oasis on each trace
    spikes = []
    calcium = []
    baseline = []
    g_hat = []
    lam_hat = []
    for t in traces:
        # check if trace has any nans, if so return all nans
        if np.isnan(t).any():
            spikes.append(np.full_like(t, np.nan))
        else:
            if estimate_parameters:
                c, s, b, g, lam = deconvolve(
                    t,
                    g=params["g"],
                    sn=params["sn"],
                    b=params["b"],
                    b_nonneg=params["b_nonneg"],
                    optimize_g=params["optimize_g"],
                    penalty=params["penalty"],
                    **kwargs,
                )
                g_hat.append(g)  # Note using AR1, so only need the first param
                lam_hat.append(lam)

                spikes.append(s)
                calcium.append(c)
                baseline.append(b)
            else:
                # c: inferred calcium, s: inferred spikes
                c, s = oasisAR1(t, params["g"], s_min=params["s_min"], lam=params["lam"])
                spikes.append(s)
                calcium.append(c)
    if estimate_parameters:
        # AR1, so just list of g, if need Ar2, need to account for tuple in json dump
        params["g_hat"] = g_hat
        params["lam_hat"] = lam_hat

    # smoothing
    # y = np.convolve(y, np.ones(3)/3, mode='same')

    return np.array(spikes), params

def make_output_directory(output_dir: str, experiment_id: str=None) -> str:
    """Creates the output directory if it does not exist
    
    Parameters
    ----------
    output_dir: str
        output directory
    experiment_id: str
        experiment_id number
    
    Returns
    -------
    output_dir: str
        output directory
    """
    if experiment_id:
        output_dir = os.path.join(output_dir, experiment_id)
    else:
        output_dir = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return Path(output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, help="Input directory", default="../data/")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory", default="../results/")
    args = parser.parse_args()
    start_time = dt.now(tz.utc)
    output_dir = Path(args.output_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    dff_file = [i for i in list(input_dir.glob('*/*')) if 'dff.h5' in str(i)][0]
    motion_corrected_fn = [i for i in list(input_dir.glob("*/*")) if "decrosstalk.h5" in str(i)][0]
    experiment_id = motion_corrected_fn.name.split("_")[0]
    output_dir = make_output_directory(output_dir, experiment_id)

    oasis_h5, params = generate_oasis_events_for_h5_path(
        dff_file,
        experiment_id,
        output_dir, 
        trace_type="data", 
        estimate_parameters=True, 
        qc_plot=True
    )

    write_output_metadata(
        {},
        ProcessName.FLUORESCENCE_EVENT_DETECTION,
        str(dff_file),
        str(oasis_h5),
        start_time,
    )


if __name__ == "__main__":
    main()
