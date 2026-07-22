

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pyedflib
import scipy.io as sio

_FS = 512.0






def load_mat_bytes(data: bytes) -> np.ndarray:
    """Load EEG array from raw .mat bytes. Returns (channels, samples) float64."""
    mat = sio.loadmat(io.BytesIO(data))
    eeg = mat["EEG"].astype(np.float64)



    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T
    return eeg


def load_mat_file(path: Path) -> np.ndarray:
    """Load EEG array from a .mat file on disk. Returns (channels, samples)."""
    with open(path, "rb") as f:
        return load_mat_bytes(f.read())


def load_swec_info(info_path: Path):
    """Load _info.mat → (fs, seizure_begin_sec, seizure_end_sec)."""
    mat = sio.loadmat(str(info_path))
    fs = float(
        mat.get("fs", mat.get("Fs", mat.get("sampling_rate", [[_FS]]))).flat[0]
    )
    sb = mat.get("seizure_begin", mat.get("Ession_begin", np.array([])))
    se = mat.get("seizure_end",   mat.get("Ession_end",   np.array([])))
    sb = sb.flatten() if sb.size > 0 else np.array([])
    se = se.flatten() if se.size > 0 else np.array([])
    return fs, sb, se


def write_edf(output_path: Path, data: np.ndarray, fs: float) -> None:
    """Write (channels, samples) float64 array to EDF+."""
    n_ch, n_samples = data.shape
    writer = pyedflib.EdfWriter(str(output_path), n_ch, file_type=pyedflib.FILETYPE_EDFPLUS)
    channel_info = []
    for i in range(n_ch):
        abs_max = float(np.max(np.abs(data[i]))) + 1.0
        channel_info.append({
            "label": f"iEEG{i + 1:03d}",
            "dimension": "uV",
            "sample_frequency": fs,
            "physical_max": abs_max,
            "physical_min": -abs_max,
            "digital_max": 32767,
            "digital_min": -32768,
            "transducer": "",
            "prefilter": "HP:0.5Hz LP:150Hz",
        })
    writer.setSignalHeaders(channel_info)
    writer.writeSamples(list(data))
    writer.close()


def write_edf_sidecar(edf_path: Path, fs: float, n_samples: int) -> None:
    """Write the JSON sidecar required by ReBrain's dataloader."""
    duration_s = n_samples / fs
    sidecar = {
        "SamplingFrequency": fs,
        "RecordingDuration": duration_s,
        "PowerLineFrequency": 50,
        "EEGReference": "median",
        "Manufacturer": "SWEC-ETHZ",
        "TaskName": "szMonitoring",
    }
    json_path = edf_path.with_suffix("").with_suffix("").parent / (
        edf_path.stem.replace("_eeg", "") + "_eeg.json"
    )
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)






def create_bids_root(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    desc_path = output_dir / "dataset_description.json"
    if not desc_path.exists():
        with open(desc_path, "w") as f:
            json.dump({
                "Name": "SWEC-ETHZ iEEG Dataset (BIDS converted)",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
                "License": "CDLA-Permissive-2.0",
            }, f, indent=2)


def get_ieeg_dir(output_dir: Path, subject: str, session: str = "01") -> Path:
    d = output_dir / f"sub-{subject}" / f"ses-{session}" / "ieeg"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_channels_tsv(ieeg_dir: Path, subject: str, session: str,
                        n_channels: int, fs: float) -> None:
    tsv_path = ieeg_dir / f"sub-{subject}_ses-{session}_task-szMonitoring_channels.tsv"
    if tsv_path.exists():
        return
    with open(tsv_path, "w") as f:
        f.write("name\ttype\tunits\tsampling_frequency\n")
        for i in range(n_channels):
            f.write(f"iEEG{i + 1:03d}\tiEEG\tuV\t{fs}\n")


def edf_stem(subject: str, session: str, run: int) -> str:
    return f"sub-{subject}_ses-{session}_task-szMonitoring_run-{run:02d}_eeg"








_SHORT_TERM_PATIENTS = [
    "ID1", "ID2", "ID3",
    "ID4a", "ID4b",
    "ID5", "ID6", "ID7", "ID8", "ID9", "ID10", "ID11", "ID12",
    "ID13a", "ID13b",
    "ID14a", "ID14b",
    "ID15", "ID16",
]


def _parse_patient_id(zip_stem: str):
    """
    'ID1'   → subject='01', session='01'
    'ID4a'  → subject='04', session='a'
    'ID13b' → subject='13', session='b'
    """
    m = re.fullmatch(r"ID(\d+)([ab]?)", zip_stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse patient ID from '{zip_stem}'")
    num = m.group(1).zfill(2)
    suffix = m.group(2).lower() if m.group(2) else "01"
    return num, suffix


def convert_short_term(input_dir: Path, output_dir: Path,
                        patient_filter=None, max_seizures=None) -> None:
    """
    Convert all Short-term zip files in input_dir to BIDS EDF.

    Each zip → one BIDS subject/session.
    Each Sz{n}.mat → one run (run-{n:02d}).
    """
    create_bids_root(output_dir)

    zip_files = sorted(input_dir.glob("ID*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No ID*.zip files found in {input_dir}")

    if patient_filter:
        pf_upper = [p.upper() for p in patient_filter]
        zip_files = [z for z in zip_files if z.stem.upper() in pf_upper]

    print(f"Found {len(zip_files)} zip files to convert")

    for zip_path in zip_files:
        subject, session = _parse_patient_id(zip_path.stem)
        print(f"\n[{zip_path.stem}] sub-{subject} ses-{session}")

        with zipfile.ZipFile(zip_path) as zf:
            mat_names = sorted(
                [n for n in zf.namelist() if n.endswith(".mat")],
                key=lambda x: int(re.search(r"Sz(\d+)", x).group(1))
                if re.search(r"Sz(\d+)", x) else 0,
            )

            if max_seizures:
                mat_names = mat_names[:max_seizures]

            if not mat_names:
                print("  No Sz*.mat files found, skipping")
                continue


            with zf.open(mat_names[0]) as f:
                first_data = load_mat_bytes(f.read())
            n_channels = first_data.shape[0]
            print(f"  {len(mat_names)} seizures, {n_channels} channels")

            ieeg_dir = get_ieeg_dir(output_dir, subject, session)
            write_channels_tsv(ieeg_dir, subject, session, n_channels, _FS)

            for mat_name in mat_names:
                m = re.search(r"Sz(\d+)", mat_name)
                run = int(m.group(1)) if m else mat_names.index(mat_name) + 1

                stem = edf_stem(subject, session, run)
                edf_path = ieeg_dir / f"{stem}.edf"

                if edf_path.exists():
                    print(f"  Skipping {stem}.edf (exists)")
                    continue

                print(f"  {mat_name} → {stem}.edf", end="", flush=True)
                try:
                    with zf.open(mat_name) as f:
                        data = load_mat_bytes(f.read())

                    if data.shape[0] != n_channels:
                        print(f"  WARNING: channel mismatch ({data.shape[0]} vs {n_channels}), skipping")
                        continue

                    write_edf(edf_path, data, _FS)
                    write_edf_sidecar(edf_path, _FS, data.shape[1])
                    print(f"  [{data.shape[1]} samples, {data.shape[1]/_FS:.1f}s]")
                except Exception as e:
                    print(f"  ERROR: {e}")






def convert_long_term(input_dir: Path, output_dir: Path, subject: str,
                       max_files=None) -> None:
    """
    Convert Long-term hourly .mat files to BIDS EDF.
    Each IDxx_Nh.mat → run-{N:02d}.
    """
    create_bids_root(output_dir)

    info_files = list(input_dir.glob("*_info.mat"))
    if info_files:
        fs, _, _ = load_swec_info(info_files[0])
        print(f"Sampling rate from info file: {fs} Hz")
    else:
        fs = _FS
        print(f"No info file found, using default fs={fs} Hz")

    mat_files = sorted(
        [f for f in input_dir.glob("*.mat") if "info" not in f.name.lower()],
        key=lambda x: x.name,
    )
    if max_files:
        mat_files = mat_files[:max_files]

    if not mat_files:
        raise FileNotFoundError(f"No hourly .mat files found in {input_dir}")

    print(f"Found {len(mat_files)} hourly .mat files")

    first_data = load_mat_file(mat_files[0])
    n_channels = first_data.shape[0]
    print(f"Channels: {n_channels}")

    session = "01"
    ieeg_dir = get_ieeg_dir(output_dir, subject, session)
    write_channels_tsv(ieeg_dir, subject, session, n_channels, fs)

    for mat_file in mat_files:
        m = re.search(r"_(\d+)h\.mat$", mat_file.name, re.IGNORECASE)
        run = int(m.group(1)) if m else mat_files.index(mat_file) + 1

        stem = edf_stem(subject, session, run)
        edf_path = ieeg_dir / f"{stem}.edf"

        if edf_path.exists():
            print(f"  Skipping {stem}.edf (exists)")
            continue

        print(f"  {mat_file.name} → {stem}.edf", end="", flush=True)
        try:
            data = load_mat_file(mat_file)
            if data.shape[0] != n_channels:
                print(f"  WARNING: channel mismatch ({data.shape[0]} vs {n_channels}), skipping")
                continue
            write_edf(edf_path, data, fs)
            write_edf_sidecar(edf_path, fs, data.shape[1])
            print(f"  [{data.shape[1]} samples, {data.shape[1]/fs:.1f}s]")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! BIDS data written to {output_dir}")






def main():
    parser = argparse.ArgumentParser(
        description="Convert SWEC iEEG dataset (.mat) to BIDS EDF format"
    )
    parser.add_argument(
        "--mode", choices=["short", "long"], required=True,
        help="'short' = BioCAS2018 short-term (zip files), 'long' = DATE2019 long-term (hourly .mat)"
    )
    parser.add_argument("--input", required=True,
                        help="Input directory. Short: folder with ID*.zip. Long: patient folder (e.g. .../ID01)")
    parser.add_argument("--output", required=True,
                        help="Output BIDS root directory")
    parser.add_argument("--subject", default=None,
                        help="[Long-term only] BIDS subject ID (e.g. '01')")
    parser.add_argument("--patients", nargs="+", default=None,
                        help="[Short-term only] Subset of patients to convert (e.g. ID1 ID4a ID13b)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files/seizures per patient (for quick testing)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.mode == "short":
        convert_short_term(
            input_dir, output_dir,
            patient_filter=args.patients,
            max_seizures=args.max_files,
        )
    else:
        if args.subject is None:

            m = re.search(r"ID(\d+)", input_dir.name, re.IGNORECASE)
            subject = m.group(1).zfill(2) if m else "01"
        else:
            subject = args.subject.zfill(2)
        convert_long_term(input_dir, output_dir, subject, max_files=args.max_files)


if __name__ == "__main__":
    main()
