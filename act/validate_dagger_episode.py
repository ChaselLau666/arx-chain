"""Offline integrity validator for Human DAgger schema-v2 HDF5 episodes."""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:  # Works both as ``python act/...py`` and as a package import.
    from .human_dagger_recorder import (
        ControlMode,
        DAGGER_VECTOR_SPECS,
        EventType,
        HANDOFF_EVENT_TYPES,
        LEGACY_ACTION_SPECS,
        OBSERVATION_SPECS,
        REQUIRED_METADATA_DEFAULTS,
        SCHEMA_VERSION,
        SOURCE_TIMESTAMP_NAMES,
    )
except ImportError:
    from human_dagger_recorder import (  # type: ignore
        ControlMode,
        DAGGER_VECTOR_SPECS,
        EventType,
        HANDOFF_EVENT_TYPES,
        LEGACY_ACTION_SPECS,
        OBSERVATION_SPECS,
        REQUIRED_METADATA_DEFAULTS,
        SCHEMA_VERSION,
        SOURCE_TIMESTAMP_NAMES,
    )


@dataclass
class ValidationResult:
    path: Path
    num_frames: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "valid": self.valid,
            "num_frames": self.num_frames,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_fixed_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).rstrip(b"\x00").decode("utf-8")
    return str(value)


def _jpeg_decoder() -> Tuple[Optional[Callable[[bytes], bool]], Optional[str]]:
    try:
        import cv2  # type: ignore

        def decode_with_cv2(payload: bytes) -> bool:
            encoded = np.frombuffer(payload, dtype=np.uint8)
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR) is not None

        return decode_with_cv2, None
    except ImportError:
        pass

    try:
        from PIL import Image  # type: ignore

        def decode_with_pillow(payload: bytes) -> bool:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                return image.width > 0 and image.height > 0

        return decode_with_pillow, None
    except ImportError:
        return None, "JPEG validation requires OpenCV (cv2) or Pillow"


def _check_dataset_shape(
    root: h5py.File,
    path: str,
    expected_tail: Tuple[int, ...],
    frame_count: int,
    errors: List[str],
) -> Optional[h5py.Dataset]:
    if path not in root:
        errors.append(f"missing dataset {path}")
        return None
    value = root[path]
    if not isinstance(value, h5py.Dataset):
        errors.append(f"{path} is not a dataset")
        return None
    expected = (frame_count,) + expected_tail
    if value.shape != expected:
        errors.append(f"{path} has shape {value.shape}, expected {expected}")
        return None
    return value


def _check_finite(dataset: h5py.Dataset, errors: List[str], block_size: int = 128) -> None:
    if dataset.dtype.kind not in "fc":
        return
    for start in range(0, dataset.shape[0], block_size):
        block = dataset[start : start + block_size]
        if not np.all(np.isfinite(block)):
            errors.append(f"{dataset.name} contains NaN or infinity")
            return


def _parse_camera_names(root: h5py.File, errors: List[str]) -> Tuple[str, ...]:
    value = _decode_attr(root.attrs.get("camera_names", ""))
    try:
        parsed = json.loads(value) if isinstance(value, str) else list(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        errors.append("camera_names attr is not valid JSON")
        return ()
    if not isinstance(parsed, list) or not parsed or not all(isinstance(x, str) and x for x in parsed):
        errors.append("camera_names attr must be a non-empty JSON string list")
        return ()
    if len(parsed) != len(set(parsed)):
        errors.append("camera_names attr contains duplicates")
        return ()
    return tuple(parsed)


def _validate_images(
    root: h5py.File,
    camera_names: Tuple[str, ...],
    frame_count: int,
    errors: List[str],
    *,
    decode_images: bool,
) -> None:
    if "/observations/images" not in root:
        errors.append("missing group /observations/images")
        return
    image_group = root["/observations/images"]
    if not isinstance(image_group, h5py.Group):
        errors.append("/observations/images is not a group")
        return
    if set(image_group.keys()) != set(camera_names):
        errors.append(
            "/observations/images cameras do not match camera_names attr: "
            f"{sorted(image_group.keys())} != {sorted(camera_names)}"
        )
    if "/compress_len" not in root:
        errors.append("missing dataset /compress_len")
        return
    lengths = root["/compress_len"]
    expected_shape = (len(camera_names), frame_count)
    if not isinstance(lengths, h5py.Dataset) or lengths.shape != expected_shape:
        shape = getattr(lengths, "shape", None)
        errors.append(f"/compress_len has shape {shape}, expected {expected_shape}")
        return
    if lengths.dtype.kind not in "iu":
        errors.append(f"/compress_len must use an integer dtype, got {lengths.dtype}")
        return

    decoder: Optional[Callable[[bytes], bool]] = None
    if decode_images:
        decoder, dependency_error = _jpeg_decoder()
        if dependency_error:
            errors.append(dependency_error)
            return

    for camera_index, camera_name in enumerate(camera_names):
        path = f"/observations/images/{camera_name}"
        if path not in root:
            continue
        dataset = root[path]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
            errors.append(f"{path} must be a 2-D padded uint8 dataset")
            continue
        if dataset.shape[0] != frame_count:
            errors.append(f"{path} has {dataset.shape[0]} frames, expected {frame_count}")
            continue
        if dataset.dtype != np.dtype(np.uint8):
            errors.append(f"{path} must use uint8, got {dataset.dtype}")
        camera_lengths = lengths[camera_index]
        bad_lengths = np.flatnonzero((camera_lengths <= 0) | (camera_lengths > dataset.shape[1]))
        if bad_lengths.size:
            errors.append(
                f"{path} has invalid compress_len at frames {bad_lengths[:8].tolist()}"
            )
            continue
        for frame_index, length_value in enumerate(camera_lengths):
            length = int(length_value)
            row = dataset[frame_index]
            payload = row[:length].tobytes()
            if not payload.startswith(b"\xff\xd8") or b"\xff\xd9" not in payload[-16:]:
                errors.append(f"{path}[{frame_index}] lacks JPEG SOI/EOI markers")
                continue
            if np.any(row[length:] != 0):
                errors.append(f"{path}[{frame_index}] has non-zero bytes after compress_len")
            if decoder is not None:
                try:
                    decoded = decoder(payload)
                except Exception as exc:  # Decoder errors differ across backends.
                    errors.append(f"{path}[{frame_index}] cannot be decoded: {exc}")
                else:
                    if not decoded:
                        errors.append(f"{path}[{frame_index}] cannot be decoded")


def _validate_modes_and_masks(
    root: h5py.File,
    frame_count: int,
    errors: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    required = (
        "control_mode",
        "intervention_mask",
        "supervision_valid",
        "policy_action_valid",
        "expert_action_valid",
        "control_epoch",
        "action_seq",
    )
    datasets: Dict[str, h5py.Dataset] = {}
    for name in required:
        dataset = _check_dataset_shape(root, f"/dagger/{name}", (), frame_count, errors)
        if dataset is not None:
            datasets[name] = dataset
    if len(datasets) != len(required):
        return np.array([], dtype=np.uint8), np.array([], dtype=np.int64)
    if any(datasets[name].dtype.kind not in "iub" for name in required):
        errors.append("one or more scalar /dagger datasets have a non-integer dtype")
        return np.array([], dtype=np.uint8), np.array([], dtype=np.int64)

    modes = datasets["control_mode"][:].astype(np.uint8, copy=False)
    epochs = datasets["control_epoch"][:].astype(np.int64, copy=False)
    allowed_values = {int(mode) for mode in ControlMode}
    invalid = np.flatnonzero(~np.isin(modes, list(allowed_values)))
    if invalid.size:
        errors.append(f"/dagger/control_mode has unknown values at {invalid[:8].tolist()}")

    intervention = datasets["intervention_mask"][:].astype(bool, copy=False)
    expected_intervention = modes == int(ControlMode.HUMAN)
    mismatch = np.flatnonzero(intervention != expected_intervention)
    if mismatch.size:
        errors.append(f"intervention_mask disagrees with HUMAN ownership at {mismatch[:8].tolist()}")

    supervision = datasets["supervision_valid"][:].astype(bool, copy=False)
    expected_supervision = np.zeros(frame_count, dtype=bool)
    if frame_count > 1:
        expected_supervision[:-1] = modes[:-1] == int(ControlMode.HUMAN)
    mismatch = np.flatnonzero(supervision != expected_supervision)
    if mismatch.size:
        errors.append(
            "supervision_valid must label exactly HUMAN-owned t->t+1 intervals; "
            f"mismatch at {mismatch[:8].tolist()}"
        )

    policy_valid = datasets["policy_action_valid"][:].astype(bool, copy=False)
    required_policy_valid = modes == int(ControlMode.POLICY)
    forbidden_policy_valid = ~np.isin(
        modes,
        (int(ControlMode.POLICY), int(ControlMode.HANDOFF_TO_POLICY)),
    )
    mismatch = np.flatnonzero(
        (required_policy_valid & ~policy_valid) | (forbidden_policy_valid & policy_valid)
    )
    if mismatch.size:
        errors.append(
            "policy_action_valid must cover every POLICY frame and only "
            "POLICY/HANDOFF_TO_POLICY ownership; "
            f"mismatch at {mismatch[:8].tolist()}"
        )

    expert_valid = datasets["expert_action_valid"][:].astype(bool, copy=False)
    expected_expert_valid = modes == int(ControlMode.HUMAN)
    mismatch = np.flatnonzero(expert_valid != expected_expert_valid)
    if mismatch.size:
        errors.append(f"expert_action_valid disagrees with HUMAN ownership at {mismatch[:8].tolist()}")

    if np.any(epochs < 0):
        errors.append("control_epoch contains negative values")
    if frame_count > 1 and np.any(np.diff(epochs) < 0):
        errors.append("control_epoch decreases")
    action_seq = datasets["action_seq"][:]
    if np.any(action_seq[policy_valid] < 0):
        errors.append("POLICY frames contain a negative action_seq")

    allowed_transitions = {
        int(ControlMode.POLICY): {
            int(ControlMode.POLICY),
            int(ControlMode.HANDOFF_TO_HUMAN),
            int(ControlMode.FAULT_HOLD),
        },
        int(ControlMode.HANDOFF_TO_HUMAN): {
            int(ControlMode.HANDOFF_TO_HUMAN),
            int(ControlMode.HUMAN),
            int(ControlMode.FAULT_HOLD),
        },
        int(ControlMode.HUMAN): {
            int(ControlMode.HUMAN),
            int(ControlMode.HANDOFF_TO_POLICY),
            int(ControlMode.FAULT_HOLD),
        },
        int(ControlMode.HANDOFF_TO_POLICY): {
            int(ControlMode.HANDOFF_TO_POLICY),
            int(ControlMode.POLICY),
            int(ControlMode.HANDOFF_TO_HUMAN),
            int(ControlMode.FAULT_HOLD),
        },
        int(ControlMode.FAULT_HOLD): {int(ControlMode.FAULT_HOLD)},
    }
    for index in range(1, frame_count):
        previous, current = int(modes[index - 1]), int(modes[index])
        if previous in allowed_transitions and current not in allowed_transitions[previous]:
            errors.append(
                f"illegal control_mode transition {previous}->{current} at frame {index}"
            )
    return modes, epochs


def _validate_events(
    root: h5py.File,
    modes: np.ndarray,
    epochs: np.ndarray,
    errors: List[str],
) -> None:
    path = "/dagger/events"
    if path not in root:
        errors.append(f"missing dataset {path}")
        return
    dataset = root[path]
    expected_fields = {"event", "request_ns", "gate_ns", "ack_ns", "frame", "epoch", "detail"}
    actual_fields = set(dataset.dtype.names or ()) if isinstance(dataset, h5py.Dataset) else set()
    if (
        not isinstance(dataset, h5py.Dataset)
        or dataset.ndim != 1
        or not expected_fields.issubset(actual_fields)
    ):
        errors.append(f"{path} has incompatible event dtype")
        return

    frame_count = len(modes)
    rows = dataset[:]
    decoded_events: List[Tuple[str, int, int]] = []
    chronological_frames: List[int] = []
    for index, row in enumerate(rows):
        try:
            event = _decode_fixed_string(row["event"]).strip().upper()
            _decode_fixed_string(row["detail"])
        except UnicodeDecodeError:
            errors.append(f"events[{index}] contains invalid UTF-8")
            continue
        request_ns = int(row["request_ns"])
        gate_ns = int(row["gate_ns"])
        ack_ns = int(row["ack_ns"])
        frame = int(row["frame"])
        epoch = int(row["epoch"])
        if not event:
            errors.append(f"events[{index}] has an empty event name")
        if frame < -1 or frame > frame_count:
            errors.append(f"events[{index}] frame {frame} is outside [-1, {frame_count}]")
        elif frame >= 0:
            chronological_frames.append(frame)
        if epoch < -1:
            errors.append(f"events[{index}] has invalid epoch {epoch}")
        times = (request_ns, gate_ns, ack_ns)
        if any(value == 0 or value < -1 for value in times):
            errors.append(f"events[{index}] timestamps must be -1 or positive")
        present_times = [value for value in times if value > 0]
        if present_times != sorted(present_times):
            errors.append(f"events[{index}] violates request <= gate <= ack")
        if event in HANDOFF_EVENT_TYPES and any(value <= 0 for value in times):
            errors.append(f"events[{index}] handoff is missing request/gate/ack timestamp")
        if 0 <= frame < frame_count and epoch >= 0 and epoch != int(epochs[frame]):
            errors.append(
                f"events[{index}] epoch {epoch} disagrees with frame {frame} epoch {epochs[frame]}"
            )
        decoded_events.append((event, frame, epoch))

    if chronological_frames != sorted(chronological_frames):
        errors.append("event frame indexes are not chronological")

    expected_activations: List[Tuple[str, int, int]] = []
    for frame in range(1, frame_count):
        if modes[frame] == modes[frame - 1]:
            continue
        if modes[frame] == int(ControlMode.HUMAN):
            expected_activations.append(
                (EventType.HANDOFF_TO_HUMAN, frame, int(epochs[frame]))
            )
        elif modes[frame] == int(ControlMode.POLICY):
            expected_activations.append(
                (EventType.HANDOFF_TO_POLICY, frame, int(epochs[frame]))
            )

    handoff_events = [item for item in decoded_events if item[0] in HANDOFF_EVENT_TYPES]
    for expected in expected_activations:
        if expected not in handoff_events:
            errors.append(
                f"missing {expected[0]} event for activation frame={expected[1]} epoch={expected[2]}"
            )
    for actual in handoff_events:
        if actual not in expected_activations:
            errors.append(
                f"{actual[0]} event does not match an activation frame={actual[1]} epoch={actual[2]}"
            )


def validate_episode(
    path: Path | str,
    *,
    decode_images: bool = True,
    require_finalized: bool = True,
) -> ValidationResult:
    """Validate an episode without mutating it.

    The returned object contains every discovered error so operators can fix or
    quarantine a file in one pass instead of iterating through failures.
    """

    episode_path = Path(path).expanduser().resolve()
    result = ValidationResult(path=episode_path)
    if not episode_path.is_file():
        result.errors.append("file does not exist")
        return result

    try:
        root_context = h5py.File(episode_path, "r")
    except (OSError, ValueError) as exc:
        result.errors.append(f"cannot open HDF5 file: {exc}")
        return result

    with root_context as root:
        attrs = {key: _decode_attr(value) for key, value in root.attrs.items()}
        if attrs.get("schema_version") != SCHEMA_VERSION:
            result.errors.append(
                f"schema_version is {attrs.get('schema_version')!r}, expected {SCHEMA_VERSION}"
            )
        if attrs.get("collection_mode") != "human_dagger":
            result.errors.append("collection_mode must be 'human_dagger'")
        if attrs.get("action_semantics") != "current_measured_qpos":
            result.errors.append("action_semantics must be 'current_measured_qpos'")
        if attrs.get("training_action_offset_frames") != 1:
            result.errors.append("training_action_offset_frames must be 1")
        if require_finalized and not bool(attrs.get("finalized", False)):
            result.errors.append("episode is not marked finalized")
        for key in REQUIRED_METADATA_DEFAULTS:
            if key not in attrs:
                result.errors.append(f"missing metadata attr {key}")

        qpos = root.get("/observations/qpos")
        if not isinstance(qpos, h5py.Dataset) or qpos.ndim != 2:
            result.errors.append("missing or invalid /observations/qpos")
            frame_count = 0
        else:
            frame_count = int(qpos.shape[0])
        result.num_frames = frame_count
        if frame_count == 0:
            result.errors.append("episode has no frames")
        attr_frames = attrs.get("num_frames")
        try:
            attr_frame_count = int(attr_frames)
        except (TypeError, ValueError, OverflowError):
            attr_frame_count = None
        if attr_frame_count != frame_count:
            result.errors.append(f"num_frames attr {attr_frames!r} != dataset length {frame_count}")

        numeric_datasets: List[h5py.Dataset] = []
        for name, width in OBSERVATION_SPECS.items():
            dataset = _check_dataset_shape(
                root, f"/observations/{name}", (width,), frame_count, result.errors
            )
            if dataset is not None:
                numeric_datasets.append(dataset)
        for name, width in LEGACY_ACTION_SPECS.items():
            dataset = _check_dataset_shape(root, f"/{name}", (width,), frame_count, result.errors)
            if dataset is not None:
                numeric_datasets.append(dataset)
        for name, width in DAGGER_VECTOR_SPECS.items():
            dataset = _check_dataset_shape(
                root, f"/dagger/{name}", (width,), frame_count, result.errors
            )
            if dataset is not None:
                numeric_datasets.append(dataset)
        for dataset in numeric_datasets:
            _check_finite(dataset, result.errors)

        timestamp_values: Dict[str, np.ndarray] = {}
        for name in ("observation_ns", "control_ns"):
            dataset = _check_dataset_shape(
                root, f"/timestamps/{name}", (), frame_count, result.errors
            )
            if dataset is None:
                continue
            if dataset.dtype.kind not in "iu":
                result.errors.append(f"/timestamps/{name} must use an integer dtype")
                continue
            values = dataset[:].astype(np.int64, copy=False)
            timestamp_values[name] = values
            if np.any(values <= 0):
                result.errors.append(f"/timestamps/{name} contains a non-positive value")
            if len(values) > 1 and np.any(np.diff(values) <= 0):
                result.errors.append(f"/timestamps/{name} is not strictly increasing")
        for name in SOURCE_TIMESTAMP_NAMES:
            dataset = _check_dataset_shape(
                root, f"/timestamps/{name}", (), frame_count, result.errors
            )
            if dataset is None:
                continue
            if dataset.dtype.kind not in "iu":
                result.errors.append(f"/timestamps/{name} must use an integer dtype")
                continue
            values = dataset[:].astype(np.int64, copy=False)
            if np.any((values == 0) | (values < -1)):
                result.errors.append(
                    f"/timestamps/{name} contains values other than -1 or positive nanoseconds"
                )
                continue
            not_applicable = values == -1
            if np.any(not_applicable) and not np.all(not_applicable):
                result.errors.append(
                    f"/timestamps/{name} mixes not-applicable (-1) and live timestamps"
                )
                continue
            live_values = values[~not_applicable]
            if len(live_values) > 1 and np.any(np.diff(live_values) < 0):
                result.errors.append(f"/timestamps/{name} decreases")

        if {"observation_ns", "control_ns"}.issubset(timestamp_values):
            if np.any(timestamp_values["control_ns"] < timestamp_values["observation_ns"]):
                result.errors.append("control_ns precedes observation_ns")

        camera_names = _parse_camera_names(root, result.errors)
        if camera_names:
            _validate_images(
                root,
                camera_names,
                frame_count,
                result.errors,
                decode_images=decode_images,
            )
        modes, epochs = _validate_modes_and_masks(root, frame_count, result.errors)
        if len(modes) == frame_count and len(epochs) == frame_count:
            _validate_events(root, modes, epochs, result.errors)

        checksum = str(attrs.get("policy_checkpoint_sha256", ""))
        if checksum not in {"", "mock"} and (
            len(checksum) != 64
            or any(ch not in "0123456789abcdefABCDEF" for ch in checksum)
        ):
            result.errors.append(
                "policy_checkpoint_sha256 must be 'mock', empty, or 64 hexadecimal characters"
            )

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", help="schema-v2 .hdf5 episode paths")
    parser.add_argument(
        "--no-decode-images",
        action="store_true",
        help="check JPEG framing and lengths but skip full image decoding",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="do not require the finalized metadata flag",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    results = [
        validate_episode(
            path,
            decode_images=not args.no_decode_images,
            require_finalized=not args.allow_partial,
        )
        for path in args.episodes
    ]
    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = "VALID" if result.valid else "INVALID"
            print(f"{status}: {result.path} ({result.num_frames} frames)")
            for warning in result.warnings:
                print(f"  warning: {warning}")
            for error in result.errors:
                print(f"  error: {error}")
    return 0 if all(result.valid for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
