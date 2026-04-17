"""Chargement et préparation des données du concours Copernicus."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import warnings
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
from numpy.lib import format as npy_format

from config import (
    FORECAST_HORIZONS_MINUTES,
    INPUT_VARIABLES,
    PROCESSING_CHUNK_SIZE,
    PROCESSED_DATA_DIR,
    PROCESSED_DTYPE,
    PROCESSED_PROFILES,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    ROI_SLICE,
    TARGET_ARRAY_SHAPE,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_SUBMISSION_PATH,
    Y_TRAIN_PATH,
)
from src.utils import ensure_directory, ensure_exists, normalize_indices


def npz_array_metadata(npz_path: str | Path) -> pd.DataFrame:
    """Lit les noms, formes et types des tableaux d'un fichier .npz sans charger les données."""
    npz_path = ensure_exists(npz_path)
    records = []

    with ZipFile(npz_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as array_file:
                version = npy_format.read_magic(array_file)
                shape, fortran_order, dtype = _read_npy_header(array_file, version)

            records.append(
                {
                    "file": npz_path.name,
                    "array": Path(member).stem,
                    "shape": shape,
                    "dtype": str(dtype),
                    "fortran_order": fortran_order,
                    "compression": "stored" if archive.getinfo(member).compress_type == ZIP_STORED else "compressed",
                    "size_mb": round(archive.getinfo(member).file_size / 1024**2, 2),
                }
            )

    return pd.DataFrame(records)


def _read_npy_header(array_file, version: tuple[int, int]):
    """Lit l'en-tête d'un tableau .npy selon la version du format."""
    if version == (1, 0):
        return npy_format.read_array_header_1_0(array_file)
    if version == (2, 0):
        return npy_format.read_array_header_2_0(array_file)
    return npy_format._read_array_header(array_file, version)  # noqa: SLF001


def load_targets(csv_path: str | Path = Y_TRAIN_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Charge le fichier CSV cible, avec une limite optionnelle sur le nombre de lignes."""
    csv_path = ensure_exists(csv_path)
    return pd.read_csv(csv_path, nrows=nrows)


def load_targets_for_indices(
    indices: Iterable[int] | int,
    csv_path: str | Path = Y_TRAIN_PATH,
) -> pd.DataFrame:
    """Charge des lignes ciblées du fichier CSV de sortie."""
    csv_path = ensure_exists(csv_path)
    indices = np.asarray([indices] if isinstance(indices, int) else list(indices), dtype=int)
    if len(indices) == 0:
        raise ValueError("indices cannot be empty.")
    if indices.min() < 0:
        raise IndexError("indices must be non-negative.")

    y = pd.read_csv(csv_path, nrows=int(indices.max()) + 1)
    return y.iloc[indices].reset_index(drop=True)


def targets_to_array(y: pd.DataFrame, dtype: str | np.dtype | None = "float32") -> np.ndarray:
    """Convertit le format CSV cible en tableau de forme (n_samples, 4, 51, 51)."""
    if "id_sequence" not in y.columns:
        raise ValueError("Expected an 'id_sequence' column in the target DataFrame.")

    values = y.drop(columns="id_sequence").to_numpy(dtype=dtype)
    expected_columns = int(np.prod(TARGET_ARRAY_SHAPE))
    if values.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} target columns, got {values.shape[1]}.")

    return values.reshape(-1, *TARGET_ARRAY_SHAPE).transpose(0, 1, 3, 2)


def array_to_submission(y: np.ndarray, ids: Iterable[int] | None = None) -> pd.DataFrame:
    """Convertit un tableau de forme (n_samples, 4, 51, 51) au format de soumission."""
    y = np.asarray(y)
    if y.ndim != 4 or y.shape[1:] != TARGET_ARRAY_SHAPE:
        raise ValueError(f"Expected y with shape (n_samples, {TARGET_ARRAY_SHAPE}), got {y.shape}.")

    values = y.transpose(0, 1, 3, 2).reshape(y.shape[0], -1)
    frame = pd.DataFrame(values)
    frame.insert(0, "id_sequence", list(ids) if ids is not None else np.arange(y.shape[0]))
    return frame


def get_npz_n_samples(npz_path: str | Path) -> int:
    """Renvoie le nombre d'échantillons stockés dans un fichier .npz du concours."""
    metadata = npz_array_metadata(npz_path)
    sample_shapes = [shape for shape in metadata["shape"] if isinstance(shape, tuple) and len(shape) > 0]
    if not sample_shapes:
        raise ValueError(f"No arrays found in {npz_path}.")
    return int(sample_shapes[0][0])


def load_input_batch(
    npz_path: str | Path = X_TRAIN_PATH,
    sample_indices: Iterable[int] | int | None = (0,),
    variables: Iterable[str] = INPUT_VARIABLES,
    prefer_mmap: bool = True,
) -> dict[str, object]:
    """Charge un ou plusieurs échantillons d'entrée avec une dimension échantillon."""
    variables = tuple(variables)
    npz_path = ensure_exists(npz_path)
    indices = normalize_indices(sample_indices, get_npz_n_samples(npz_path))

    if prefer_mmap:
        arrays = open_npz_arrays_mmap(npz_path, variables=variables)
        x = {variable: np.asarray(arrays[variable][indices]) for variable in variables}
    else:
        with np.load(npz_path, allow_pickle=False) as archive:
            x = {variable: np.asarray(archive[variable][indices]) for variable in variables}

    return {
        "X": x,
        "indices": indices,
        "datetime": load_datetime_samples(npz_path, indices),
    }


def load_input_sample(
    npz_path: str | Path = X_TRAIN_PATH,
    sample_index: int = 0,
    variables: Iterable[str] = INPUT_VARIABLES,
    prefer_mmap: bool = True,
) -> dict[str, np.ndarray]:
    """Charge un échantillon pour chaque variable demandée depuis un fichier .npz."""
    batch = load_input_batch(npz_path, [sample_index], variables=variables, prefer_mmap=prefer_mmap)
    sample = {variable: values[0] for variable, values in batch["X"].items()}
    if len(batch["datetime"]):
        sample["datetime"] = batch["datetime"][0]
    return sample


def load_input_samples(
    npz_path: str | Path = X_TRAIN_PATH,
    sample_indices: Iterable[int] = (0,),
    variables: Iterable[str] = INPUT_VARIABLES,
    prefer_mmap: bool = True,
) -> list[dict[str, np.ndarray]]:
    """Charge plusieurs échantillons pour chaque variable demandée depuis un fichier .npz."""
    batch = load_input_batch(npz_path, sample_indices, variables=variables, prefer_mmap=prefer_mmap)
    samples = []
    for position in range(len(batch["indices"])):
        sample = {variable: values[position] for variable, values in batch["X"].items()}
        if len(batch["datetime"]):
            sample["datetime"] = batch["datetime"][position]
        samples.append(sample)
    return samples


def get_sample_from_open_arrays(
    arrays: dict[str, np.ndarray],
    sample_index: int,
    variables: Iterable[str] = INPUT_VARIABLES,
) -> dict[str, np.ndarray]:
    """Extrait un échantillon depuis des tableaux déjà ouverts en mémoire."""
    variables = tuple(variables)
    return {var: np.asarray(arrays[var][sample_index]) for var in variables}


def load_datetime_sample(npz_path: str | Path, sample_index: int = 0):
    """Charge une date depuis le fichier .npz du concours."""
    values = load_datetime_samples(npz_path, [sample_index])
    return values[0] if len(values) else None


def load_datetime_samples(npz_path: str | Path, sample_indices: Iterable[int] | int | None = None) -> np.ndarray:
    """Charge une ou plusieurs dates depuis le fichier .npz du concours."""
    npz_path = ensure_exists(npz_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*align should be passed.*")
        with np.load(npz_path, allow_pickle=True) as archive:
            if "datetime" not in archive.files:
                return np.array([])
            datetimes = archive["datetime"]
            if sample_indices is None:
                return np.asarray(datetimes)
            indices = normalize_indices(sample_indices, len(datetimes))
            return np.asarray(datetimes[indices])


def load_dataset_overview(npz_path: str | Path) -> dict[str, object]:
    """Résume le contenu d'un fichier .npz du concours."""
    npz_path = ensure_exists(npz_path)
    metadata = npz_array_metadata(npz_path)
    return {
        "file": Path(npz_path).name,
        "arrays": metadata["array"].tolist(),
        "n_arrays": len(metadata),
        "n_samples": get_npz_n_samples(npz_path),
    }


def open_npz_arrays_mmap(
    npz_path: str | Path,
    variables: Iterable[str] = INPUT_VARIABLES,
    mmap_mode: str = "r",
) -> dict[str, np.memmap]:
    """Ouvre en mémoire mappée les tableaux numériques d'un fichier .npz non compressé."""
    npz_path = ensure_exists(npz_path)
    arrays = {}

    with ZipFile(npz_path) as archive:
        members = {Path(name).stem: archive.getinfo(name) for name in archive.namelist() if name.endswith(".npy")}

    for variable in variables:
        if variable not in members:
            raise KeyError(f"Array '{variable}' not found in {npz_path}.")

        info = members[variable]
        if info.compress_type != ZIP_STORED:
            raise ValueError(
                f"Array '{variable}' is compressed in {npz_path}. "
                "Use extract_npz_to_npy(...) before memory mapping."
            )

        data_offset, shape, dtype, fortran_order = _npy_payload_offset_in_npz(npz_path, info)
        arrays[variable] = np.memmap(
            npz_path,
            dtype=dtype,
            mode=mmap_mode,
            offset=data_offset,
            shape=shape,
            order="F" if fortran_order else "C",
        )

    return arrays


def _npy_payload_offset_in_npz(npz_path: Path, info):
    """Calcule la position du contenu .npy dans un membre du fichier .npz."""
    npy_start = _zip_member_payload_offset(npz_path, info)
    with npz_path.open("rb") as file:
        file.seek(npy_start)
        version = npy_format.read_magic(file)
        shape, fortran_order, dtype = _read_npy_header(file, version)
        data_offset = file.tell()
    return data_offset, shape, dtype, fortran_order


def _zip_member_payload_offset(npz_path: Path, info) -> int:
    """Calcule la position du contenu binaire d'un membre ZIP local."""
    with npz_path.open("rb") as file:
        file.seek(info.header_offset)
        local_header = file.read(30)

    if local_header[:4] != b"PK\x03\x04":
        raise ValueError(f"Invalid local ZIP header for {info.filename}.")

    filename_length = int.from_bytes(local_header[26:28], byteorder="little")
    extra_length = int.from_bytes(local_header[28:30], byteorder="little")
    return info.header_offset + 30 + filename_length + extra_length


def extract_npz_to_npy(
    npz_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Extrait les membres .npz en fichiers .npy ouvrables ensuite en mémoire mappée."""
    npz_path = ensure_exists(npz_path)
    output_dir = ensure_directory(output_dir)

    paths = {}
    with ZipFile(npz_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue

            destination = output_dir / Path(member).name
            if destination.exists() and not overwrite:
                paths[Path(member).stem] = destination
                continue

            with archive.open(member) as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            paths[Path(member).stem] = destination

    return paths


def open_extracted_arrays(
    extracted_dir: str | Path,
    mmap_mode: str | None = "r",
) -> dict[str, np.ndarray]:
    """Ouvre des tableaux .npy extraits, avec mémoire mappée par défaut."""
    extracted_dir = Path(extracted_dir)
    arrays = {}
    for path in extracted_dir.glob("*.npy"):
        arrays[path.stem] = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    return arrays


def extract_roi(array: np.ndarray) -> np.ndarray:
    """Découpe la région centrale 51 par 51 dans des images brutes 81 par 81."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array[ROI_SLICE, ROI_SLICE]
    if array.ndim == 3:
        # Fonctionne pour les formats avec temps en premier ou en dernier.
        if array.shape[0] in {4, 8}:
            return array[:, ROI_SLICE, ROI_SLICE]
        return array[ROI_SLICE, ROI_SLICE, :]
    if array.ndim == 4:
        return array[:, :, ROI_SLICE, ROI_SLICE]
    raise ValueError(f"Unsupported shape: {array.shape}")


def processed_profile_dir(profile: str, split: str | None = None) -> Path:
    """Renvoie le dossier associé à un profil de données prétraitées."""
    if profile not in PROCESSED_PROFILES:
        raise KeyError(f"Unknown processed profile '{profile}'. Expected one of {tuple(PROCESSED_PROFILES)}.")
    root = PROCESSED_DATA_DIR / profile
    return root / split if split else root


def processed_profile_exists(profile: str, split: str = "train") -> bool:
    """Indique si un profil prétraité existe déjà sur disque."""
    split_dir = processed_profile_dir(profile, split)
    return (split_dir / "manifest.json").exists() and (split_dir / "X").exists()


def prepare_processed_profile(
    profile: str = "dev",
    split: str = "train",
    variables: Iterable[str] = INPUT_VARIABLES,
    indices: Iterable[int] | int | None = None,
    include_target: bool | None = None,
    crop_roi: bool = True,
    dtype: str | np.dtype = PROCESSED_DTYPE,
    chunk_size: int = PROCESSING_CHUNK_SIZE,
    overwrite: bool = False,
) -> dict[str, object]:
    """Crée un profil prétraité en fichiers .npy séparés et réutilisables."""
    if profile not in PROCESSED_PROFILES:
        raise KeyError(f"Unknown processed profile '{profile}'. Expected one of {tuple(PROCESSED_PROFILES)}.")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")

    variables = tuple(variables)
    include_target = split == "train" if include_target is None else include_target
    source_npz = X_TRAIN_PATH if split == "train" else X_TEST_PATH
    n_available = get_npz_n_samples(source_npz)
    profile_limit = PROCESSED_PROFILES[profile]["n_samples"]
    sample_indices = normalize_indices(indices, n_available, limit=profile_limit)

    split_dir = ensure_directory(processed_profile_dir(profile, split))
    x_dir = ensure_directory(split_dir / "X")
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} already exists. Pass overwrite=True to rebuild it.")

    raw_arrays = open_npz_arrays_mmap(source_npz, variables=variables)
    written_x = {}

    for variable in variables:
        source = raw_arrays[variable]
        sample_shape = source.shape[1:]
        if crop_roi:
            roi_size = ROI_SLICE.stop - ROI_SLICE.start
            sample_shape = (sample_shape[0], roi_size, roi_size)

        destination = x_dir / f"{variable}.npy"
        output = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=dtype,
            shape=(len(sample_indices), *sample_shape),
        )

        for start in range(0, len(sample_indices), chunk_size):
            chunk_indices = sample_indices[start : start + chunk_size]
            values = np.asarray(source[chunk_indices], dtype=dtype)
            if crop_roi:
                values = extract_roi(values)
            output[start : start + len(chunk_indices)] = values
        output.flush()
        written_x[variable] = destination

    np.save(split_dir / "indices.npy", sample_indices)
    np.save(split_dir / "datetime.npy", load_datetime_samples(source_npz, sample_indices), allow_pickle=True)

    target_path = None
    if include_target:
        target_rows = load_targets_for_indices(sample_indices, Y_TRAIN_PATH)
        y = targets_to_array(target_rows, dtype=dtype)
        target_path = split_dir / "y.npy"
        np.save(target_path, y)
        np.save(split_dir / "target_ids.npy", target_rows["id_sequence"].to_numpy())

    manifest = {
        "profile": profile,
        "split": split,
        "description": PROCESSED_PROFILES[profile]["description"],
        "n_samples": int(len(sample_indices)),
        "indices": [int(index) for index in sample_indices],
        "variables": list(variables),
        "dtype": str(np.dtype(dtype)),
        "crop_roi": crop_roi,
        "roi": [ROI_SLICE.start, ROI_SLICE.stop],
        "source_npz": str(source_npz),
        "x_paths": {name: str(path) for name, path in written_x.items()},
        "target_path": str(target_path) if target_path else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def open_processed_profile(
    profile: str = "dev",
    split: str = "train",
    variables: Iterable[str] = INPUT_VARIABLES,
    mmap_mode: str | None = "r",
) -> dict[str, object]:
    """Ouvre un profil prétraité depuis le disque, avec mémoire mappée par défaut."""
    split_dir = ensure_exists(processed_profile_dir(profile, split))
    x_dir = ensure_exists(split_dir / "X")
    manifest_path = ensure_exists(split_dir / "manifest.json")

    data: dict[str, object] = {
        "X": {},
        "indices": np.load(split_dir / "indices.npy", allow_pickle=False),
        "datetime": np.load(split_dir / "datetime.npy", allow_pickle=True),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
    }
    for variable in tuple(variables):
        data["X"][variable] = np.load(x_dir / f"{variable}.npy", mmap_mode=mmap_mode, allow_pickle=False)

    target_path = split_dir / "y.npy"
    if target_path.exists():
        data["y"] = np.load(target_path, mmap_mode=mmap_mode, allow_pickle=False)
    return data
