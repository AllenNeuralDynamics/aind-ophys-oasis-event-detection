import argparse
import json
import logging
import os
from datetime import datetime as dt
from datetime import timezone
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.components.wrappers import AssetPath
from aind_data_schema.core.processing import DataProcess, ProcessName, ProcessStage
from aind_data_schema.core.quality_control import (
    CurationMetric,
    QCStatus,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from logging_util import setup_logging
from aind_metadta_manager.utils import (
    SchemaVersion,
    get_acquisition_metadata,
    get_major_schema_version,
    get_metadata,
)
from oasis.functions import deconvolve
from oasis.oasis_methods import oasisAR1, oasisAR1_f32, oasisAR2


def write_data_process(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
    experimenters: list[str],
) -> None:
    """Writes output metadata to plane processing.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    input_fp: str
        path to raw movies
    output_fp: str
        path to motion corrected movies
    unique_id: str
        unique identifier for the processing
    start_time: dt
        start time of processing
    end_time: dt
        end time of processing
    experimenters: list[str]
        names of people responsible for processing, pulled from data_description.json
    """
    data_proc = DataProcess(
        process_type=ProcessName.FLUORESCENCE_EVENT_DETECTION,
        stage=ProcessStage.PROCESSING,
        experimenters=experimenters,
        code=Code(
            url=os.getenv("REPO_URL", ""),
            version=os.getenv("VERSION", ""),
            parameters=metadata,
            input_data=[DataAsset(url=str(input_fp))],
        ),
        start_date_time=start_time,
        end_date_time=end_time,
        output_path=AssetPath(Path(output_fp).as_posix()),
    )
    output_dir = Path(output_fp).parent
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


def write_qc_metric(output_dir: Path, experiment_id: str, N: int) -> None:
    """Writes a curation metric json file with per-ROI event detection plot references.

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
        cell_plots[str(roi_id)] = {
            "reference": f"{experiment_id}/events/plots/{experiment_id}_{roi_id}_oasis.png"
        }
    metric = CurationMetric(
        name=f"{experiment_id} Event Detection",
        modality=Modality.from_abbreviation("pophys"),
        stage=Stage.PROCESSING,
        description="dF / F and roi events detected by oasis",
        status_history=[
            QCStatus(
                evaluator="Automated",
                timestamp=dt.now(timezone.utc),
                status=Status.PASS,
            )
        ],
        value=[json.dumps(cell_plots)],
        type="events",
    )

    with open(output_dir / f"{experiment_id}_oasis_events_metric.json", "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


def get_frame_rate(metadata: dict, version: SchemaVersion) -> float:
    """Attempt to pull frame rate from session.json (v1) or acquisition.json (v2).

    v1 path: data_streams[i].ophys_fovs[0].frame_rate
    v2 path: data_streams[i].configurations[j].sampling_strategy.frame_rate

    Raises ValueError if frame rate is not found.

    Parameters
    ----------
    metadata: dict
        session (v1) or acquisition (v2) metadata
    version: SchemaVersion
        SchemaVersion.V1 or SchemaVersion.V2

    Returns
    -------
    frame_rate: float
        frame rate in Hz
    """
    frame_rate_hz = None
    if version == SchemaVersion.V2:
        for stream in metadata.get("data_streams", []):
            for config in stream.get("configurations", []):
                sampling = config.get("sampling_strategy")
                if sampling and sampling.get("frame_rate") is not None:
                    frame_rate_hz = sampling["frame_rate"]
                    break
            if frame_rate_hz is not None:
                break
    else:
        for stream in metadata.get("data_streams", []):
            if stream.get("ophys_fovs"):
                frame_rate_hz = stream["ophys_fovs"][0]["frame_rate"]
                break
    if frame_rate_hz is None:
        raise ValueError(f"No frame rate found in {version} acquisition metadata")
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
    data_description_data = get_metadata(input_dir, "data_description.json")
    schema_version = get_major_schema_version(data_description_data)
    acquisition_data = get_acquisition_metadata(input_dir, schema_version)
    frame_rate = get_frame_rate(acquisition_data, schema_version)
    name = data_description_data.get("name", "")
    experimenters = [
        inv["name"] for inv in data_description_data.get("investigators", [])
    ]
    process_name = os.getenv("PROCESS_NAME")
    setup_logging(
        process_name,
        acquisition_name=name,
        process_name=process_name,
        pipeline_name=os.getenv("PIPELINE_NAME", ""),
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
        experimenters=experimenters,
    )

    write_qc_metric(output_dir, experiment_id, N)
