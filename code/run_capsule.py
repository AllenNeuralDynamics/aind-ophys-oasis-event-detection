import argparse
import json
import logging
import os
from datetime import datetime as dt
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_data_schema.core.quality_control import (QCEvaluation, QCMetric,
                                                   QCStatus, Stage, Status)
from aind_data_schema_models.modalities import Modality
from aind_log_utils.log import setup_logging
from aind_qcportal_schema.metric_value import CurationMetric, DropdownMetric
from oasis.functions import deconvolve
from oasis.oasis_methods import oasisAR1, oasisAR1_f32, oasisAR2


def write_data_process(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    raw_movie: str
        path to raw movies
    output_fp: str
        path to motion corrected movies
    unique_id: str
        unique identifier for the processing
    start_time: dt
        start time of processing
    end_time: dt
        end time of processing
    """
    data_proc = DataProcess(
        name=ProcessName.FLUORESCENCE_EVENT_DETECTION,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=str(input_fp),
        output_location=str(output_fp),
        code_url=(os.getenv("REPO_URL", "")),
        parameters=metadata,
    )
    if isinstance(output_fp, str):
        output_dir = Path(output_fp).parent
    else:
        output_dir = output_fp.parent
    with open(output_dir / f"{unique_id}_oasis_events_data_process.json", "w") as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def make_output_directory(output_dir: Path, experiment_id: str) -> Path:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    experiment_id: str
        experiment_id number

    Returns
    -------
    output_dir: Path
        output directory
    """
    output_dir = output_dir / experiment_id / "events"
    output_dir.mkdir(exist_ok=True, parents=True)

    return output_dir


def plot_trace_and_events_png(
    trace, ca, spike, timestamps, roi_id, tau, plots_path, experiment_id, show_fig=False
) -> None:
    sns.set_context("talk")
    fig, ax = plt.subplots(2, 1, figsize=(20, 5), sharex=True)
    ax[0].plot(timestamps, 100 * trace, color="C0", label=r"raw $\Delta$F/F")
    ax[0].plot(timestamps, 100 * ca, color="C1", label="denoised")
    end = min(580, timestamps[-1])  # arbitrary time period to check
    ax[0].set_xlim(max(0, end - 180), end)
    ax[0].legend()
    ax[0].set_ylabel(r"$\Delta$F/F [%]")
    ax[0].set_title(f"cell_roi_id: {roi_id}")
    ax[1].plot(timestamps, spike, color="C2", label=f"events, tau={tau:.4f}s")
    ax[1].legend()
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Spike rate [a.u.]")
    plt.tight_layout(pad=0.2)
    fig.savefig(plots_path / f"{experiment_id}_{roi_id}_oasis.png")
    if not show_fig:
        plt.close(fig)


def write_qc_evalutation(output_dir: Path, experiment_id: str, N: int) -> None:
    """Writes QC metrics to json files. Creates one json file per ROI.

    Parameters
    ----------
    output_dir: Path
        output directory
    experiment_id: str
        unique plane id
    N: int
        number of ROIs detected
    """
    cell_plots = dict()

    for roi_id in range(N):
        cell_plots[roi_id] = f"{experiment_id}/events/plots/{experiment_id}_{roi_id}_oasis.png"
    curation = CurationMetric(curations=[json.dumps(cell_plots)])
    metric = QCMetric(
        name=f"{experiment_id} Event Detection",
        description="dF / F and roi events detected by oasis",
        reference="",
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=curation,
    )

    evaluation = QCEvaluation(
        modality=Modality.from_abbreviation("pophys"),
        stage=Stage.PROCESSING,
        name="Events",
        description="Events detected in each roi for all fovs",
        allow_failed_metrics=False,
        metrics=[metric],
        tags=["events"],
    )

    with open(output_dir / f"{experiment_id}_oasis_events_evaluation.json", "w") as f:
        json.dump(json.loads(evaluation.model_dump_json()), f, indent=4)


def get_metadata(input_dir: Path, meta_type: str) -> dict:
    """Extracts metadata from processing and subject json files

    Parameters
    ----------
    input_dir: Path
        input directory
    meta_type: str
        type of metadata to extract

    Returns
    -------
    metadata: dict
        metadata
    """
    input_fp = next(input_dir.rglob(f"{meta_type}"), "")
    if not input_fp:
        raise FileNotFoundError(f"No {meta_type} file found in {input_dir}")
    with open(input_fp, "r") as f:
        metadata = json.load(f)
    return metadata


def get_frame_rate(session: dict) -> float:
    """Attempt to pull frame rate from session.json
    Raises ValueError if frame rate not in session.json

    Parameters
    ----------
    session: dict
        session metadata

    Returns
    -------
    frame_rate: float
        frame rate in Hz
    """
    frame_rate_hz = None
    for i in session.get("data_streams", ""):
        if i.get("ophys_fovs", ""):
            frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
            break
    if frame_rate_hz is None:
        raise ValueError("No frame rate found in session.json")
    if isinstance(frame_rate_hz, str):
        frame_rate_hz = float(frame_rate_hz)
    return frame_rate_hz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=str, default="../data/", help="Input directory"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="../results/", help="Output directory"
    )
    parser.add_argument(
        "--estimate_parameters",
        type=bool,
        default=True,
        help="Whether to estimate parameters, in particular sparsity parameter lam, "
        "using the noise constraint or whether to use provided parameters.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Exponential decay time in seconds (1/e, thus equal to half-life time divided "
        "by ln(2)). Estimated from the autocovariance of the data if no value is given. "
        "Has to be provided explicitly if estimate_parameters==False",
    )
    parser.add_argument(
        "--tau_rise",
        type=float,
        default=0,
        help="Exponential rise time in seconds (1/e, thus equal to half-rise time divided by "
        " ln(2)). Estimated from the autocovariance of the data if no value is given.",
    )
    parser.add_argument(
        "--optimize_tau",
        type=int,
        default=0,
        help="Number of large, isolated events to consider for further optimizing tau. "
        "No optimization if optimize_tau=0.",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="Fluorescence baseline value. If no value is given, "
        "then b is optimized if estimate_parameters==True else 0.",
    )
    parser.add_argument(
        "--b_nonneg",
        type=bool,
        default=True,
        help="Enforce strictly non-negative baseline if True",
    )
    parser.add_argument(
        "--sn",
        type=float,
        default=None,
        help="Standard deviation of the noise distribution. If no value is given, "
        "then sn is estimated from the data based on power spectral density.",
    )
    parser.add_argument(
        "--penalty",
        type=int,
        default=1,
        help="Sparsity penalty. 1: min |s|_1  0: min |s|_0",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Sparsity penalty parameter. If no value is given, then lam is "
        "estimated as the optimal Lagrange multiplier for noise constraint "
        "under L1 penalty if estimate_parameters==True else 0",
    )
    parser.add_argument(
        "--s_min",
        type=float,
        default=0,
        help="Minimal non-zero activity within each bin (minimal 'spike size').",
    )
    parser.add_argument("--no_qc", action="store_true", help="Skip QC plots.")
    args = parser.parse_args()
    params = vars(args)
    start_time = dt.now()
    output_dir = Path(args.output_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    dff_dir = next(input_dir.rglob("*/dff/"))
    experiment_id = dff_dir.parent.name
    dff_fp = next(dff_dir.glob("*dff.h5"))
    output_dir = make_output_directory(output_dir, experiment_id)
    session_data = get_metadata(input_dir, "session.json")
    frame_rate = get_frame_rate(session_data)
    subject_data = get_metadata(input_dir, "subject.json")
    subject_id = subject_data.get("subject_id", "")
    data_description_data = get_metadata(input_dir, "data_description.json")
    name = data_description_data.get("name", "")
    setup_logging(
        "aind-ophys-oasis-event-detection", mouse_id=subject_id, session_name=name
    )
    # convert time constants to parameters of the auto-regressive (AR) process
    if args.tau is None or args.tau_rise is None:  # automatically estimate tau
        if args.tau_rise == 0:  # negligible rise time -> AR1
            params["g"] = (None,)
        else:  # automatically estimate rise time too -> AR2
            params["g"] = (None, None)
    else:
        if args.tau_rise == 0:  # negligible rise time -> AR1
            params["g"] = (np.exp(-1 / (args.tau * frame_rate)),)
        else:  # AR2
            d, r = (
                np.exp(-1 / (args.tau * frame_rate)),
                np.exp(-1 / (args.tau_rise * frame_rate)),
            )
            params["g"] = (d + r, -d * r)

    if not args.estimate_parameters:
        if args.tau is None:
            raise UserWarning(
                "'estimate_parameters' is False, but no value for decay time "
                "constant 'tau' has been provided."
            )
        if args.lam is None:
            params["lam"] = 0
            logging.info(
                "'estimate_parameters' is False, but no value for sparsity penalty "
                " 'lam' has been provided,thus automatically setting 'lam' to 0."
            )
        if args.b is None:
            params["b"] = 0
            logging.info(
                "'estimate_parameters' is False, but no value for baseline 'b' has been provided, "
                "thus automatically setting 'b' to 0."
            )

    def _deconv(t):
        if np.isnan(t).any():  # check if trace has any nans, if so return all nans
            c, s = np.full_like(t, np.nan), np.full_like(t, np.nan)
            return (
                (c, s, np.nan, np.nan, np.nan) if args.estimate_parameters else (c, s)
            )
        else:
            if args.estimate_parameters:
                relevant_params = {
                    k: params[k] for k in ["g", "sn", "b", "b_nonneg", "penalty"]
                }
                relevant_params["optimize_g"] = params["optimize_tau"]
                return deconvolve(t, **relevant_params)
            elif args.tau_rise == 0:
                return (oasisAR1_f32 if t.dtype == np.float32 else oasisAR1)(
                    t - params["b"], args.g[0], s_min=args.s_min, lam=args.lam
                )
            else:
                return oasisAR2(
                    t.astype(float) - params["b"],
                    args.g[0],
                    args.g[1],
                    s_min=args.s_min,
                    lam=args.lam,
                )

    try:
        print(f"Performing Event Detection for {experiment_id}")

        with h5py.File(dff_fp, "r") as f:
            traces = f["data"][:]
        N, T = traces.shape
        nans = np.where(np.isnan(traces))[0]
        if len(nans) > 0:
            logging.info(f"Traces have nans: {len(nans)} in {experiment_id}")

        # run oasis on each trace in parallel
        if N:
            pool = Pool(int(tmp) if (tmp := os.environ.get("CO_CPUS")) else tmp)
            res = pool.map(_deconv, traces)
            calcium, spikes = [
                np.array([r[i] for r in res], dtype="f4") for i in (0, 1)
            ]
            if args.estimate_parameters:
                b_hat, g_hat, lam_hat = [
                    np.array([r[i] for r in res], dtype="f4") for i in (2, 3, 4)
                ]
                # convert parameters of the auto-regressive (AR) process to time constants
                if g_hat.ndim == 1:  # AR1
                    tau_hat = -1 / np.log(g_hat) / frame_rate
                else:  # AR2
                    tmp = np.sqrt(g_hat[:, 0] ** 2 + 4 * g_hat[:, 1]) / 2
                    tau_hat = -1 / np.log(g_hat[:, 0] / 2 + tmp) / frame_rate
                    tau_rise_hat = -1 / np.log(g_hat[:, 0] / 2 - tmp) / frame_rate
        else:  # no ROIs detected
            calcium, spikes = [np.empty((0, T), dtype="f4")] * 2
            if args.estimate_parameters:
                b_hat, tau_hat, lam_hat = [], [], []
                if args.tau_rise != 0:
                    tau_rise_hat = []

        # save to h5
        oasis_h5 = output_dir / f"{experiment_id}_events_oasis.h5"
        with h5py.File(oasis_h5, "w") as f:
            f.create_dataset("events", data=spikes, compression="gzip")
            f.create_dataset("denoised", data=calcium, compression="gzip")
            if args.estimate_parameters:
                f.create_dataset("b_hat", data=b_hat)
                f.create_dataset("tau_hat", data=tau_hat)
                if args.tau_rise != 0:
                    f.create_dataset("tau_rise_hat", data=tau_rise_hat)
                f.create_dataset("lam_hat", data=lam_hat)

        # QC plots
        if N:
            if not args.no_qc:
                plots_path = output_dir / "plots"
                plots_path.mkdir(exist_ok=True, parents=True)
                timestamps = np.arange(T) / frame_rate
                pool.starmap(
                    plot_trace_and_events_png,
                    zip(
                        traces,
                        calcium
                        + (b_hat[:, None] if args.estimate_parameters else params["b"]),
                        spikes,
                        [timestamps] * N,
                        range(N),
                        tau_hat if args.estimate_parameters else [args.tau] * N,
                        [plots_path] * N,
                        [experiment_id] * N,
                    ),
                )
            pool.close()

        logging.info(f"SUCCESS: {experiment_id}")
    except Exception as e:
        logging.error(f"FAILED: {experiment_id}")
        raise e

    write_data_process(
        params,
        dff_fp,
        oasis_h5,
        experiment_id,
        start_time,
        end_time=dt.now(),
    )

    write_qc_evalutation(output_dir, experiment_id, N)
