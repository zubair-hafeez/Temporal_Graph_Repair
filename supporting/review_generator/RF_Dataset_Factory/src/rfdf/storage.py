"""Atomic dataset commits, integrity checks, and portable export."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ARRAY_NAMES = ("coefficients", "delays_s", "cfr", "offsets_hz", "taps", "tap_lags")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def safe_id(value):
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_split(group_id, seed, fractions):
    """Group hashing is stable under frame count/control expansion changes."""
    fraction = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) / 2**64
    if fraction < fractions["train"]:
        return "train"
    if fraction < fractions["train"] + fractions["validation"]:
        return "validation"
    return "test"


def write_sample(root, snapshot, result, config_hash, split):
    root = Path(root)
    sample_id = safe_id(snapshot["sample_id"])
    sample_dir = root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = sample_dir / f"{sample_id}.npz"
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=sample_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **{key: result[key] for key in ARRAY_NAMES})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, arrays_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    metadata_path = sample_dir / f"{sample_id}.json"
    scene_path = sample_dir / f"{sample_id}.scene.json"
    atomic_json(scene_path, snapshot)
    record = {"sample_id": sample_id, "group_id": snapshot["group_id"], "episode": snapshot["episode"],
              "frame": snapshot["frame"], "condition": snapshot["condition"], "split": split, "config_hash": config_hash,
              "physical_state_id": snapshot.get("physical_state_id"), "link_id": snapshot.get("link_id"),
              "metrics": result["metrics"], "files": {"arrays": arrays_path.relative_to(root).as_posix(),
              "scene": scene_path.relative_to(root).as_posix()}, "checksums": {"arrays": checksum(arrays_path), "scene": checksum(scene_path)}}
    atomic_json(metadata_path, record)
    return record


def _local_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Invalid artifact path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes dataset")
    return target


def inspect_record(root, record, config_hash=None):
    root = Path(root)
    sample_id = safe_id(record["sample_id"])
    safe_id(record["group_id"])
    if config_hash is not None and record.get("config_hash") != config_hash:
        raise ValueError(f"{sample_id}: configuration hash mismatch")
    if record.get("split") not in {"train", "validation", "test"}:
        raise ValueError(f"{sample_id}: invalid split")
    for key in ("arrays", "scene"):
        path = _local_path(root, record["files"][key])
        if not path.is_file() or checksum(path) != record["checksums"][key]:
            raise ValueError(f"{sample_id}: missing or corrupt {key}")
    scene = read_json(_local_path(root, record["files"]["scene"]))
    for key in ("sample_id", "group_id", "episode", "frame", "condition"):
        if scene.get(key) != record.get(key):
            raise ValueError(f"{sample_id}: scene/record {key} mismatch")
    with np.load(_local_path(root, record["files"]["arrays"]), allow_pickle=False) as arrays:
        if set(arrays.files) != set(ARRAY_NAMES):
            raise ValueError(f"{sample_id}: unexpected array fields")
        for key in ARRAY_NAMES:
            if arrays[key].ndim != 1 or not np.all(np.isfinite(arrays[key])):
                raise ValueError(f"{sample_id}: invalid {key}")
        for key in ("coefficients", "cfr", "taps"):
            if arrays[key].dtype.kind != "c":
                raise ValueError(f"{sample_id}: {key} must be complex voltage values")
        for key in ("delays_s", "offsets_hz"):
            if arrays[key].dtype.kind != "f":
                raise ValueError(f"{sample_id}: {key} must be floating point values")
        if arrays["tap_lags"].dtype.kind not in "iu" or not np.array_equal(arrays["tap_lags"], np.arange(len(arrays["tap_lags"]))):
            raise ValueError(f"{sample_id}: invalid integer tap grid")
        if arrays["coefficients"].shape != arrays["delays_s"].shape or np.any(arrays["delays_s"] < 0):
            raise ValueError(f"{sample_id}: coefficient/delay mismatch")
        radio = scene["radio"]
        if len(arrays["cfr"]) != radio["num_subcarriers"] or arrays["offsets_hz"].shape != arrays["cfr"].shape:
            raise ValueError(f"{sample_id}: CFR shape mismatch")
        if len(arrays["taps"]) != radio["tap_count"] or arrays["tap_lags"].shape != arrays["taps"].shape:
            raise ValueError(f"{sample_id}: tap shape mismatch")
        if record["metrics"].get("path_count") != len(arrays["delays_s"]) or record["metrics"].get("valid") != bool(len(arrays["delays_s"])):
            raise ValueError(f"{sample_id}: path metrics mismatch")
        from .backends import summarize
        expected = summarize(arrays["coefficients"], arrays["delays_s"], radio)
        for key in ("cfr", "offsets_hz", "taps", "tap_lags"):
            if not np.allclose(arrays[key], expected[key], rtol=1e-7, atol=1e-12):
                raise ValueError(f"{sample_id}: inconsistent derived {key}")
        # Use the same summary contract as generation, including null no-path
        # targets and explicit unit/convention labels. Array consistency alone
        # does not protect learning targets from stale metadata.
        for key, value in expected["metrics"].items():
            if key not in record["metrics"]:
                raise ValueError(f"{sample_id}: missing derived metric {key}")
            actual = record["metrics"][key]
            if value is None or isinstance(value, (str, bool, int)):
                matches = type(actual) is type(value) and actual == value
            else:
                # Seconds need a smaller absolute tolerance than dB targets.
                absolute_tolerance = 1e-15 if key.endswith("_s") else 1e-8
                matches = (
                    not isinstance(actual, bool)
                    and isinstance(actual, (int, float))
                    and np.isfinite(actual)
                    and np.isclose(actual, value, rtol=1e-7, atol=absolute_tolerance)
                )
            if not matches:
                raise ValueError(f"{sample_id}: inconsistent derived metric {key}")
    return record


def committed_records(root, config_hash):
    records = []
    for path in sorted((Path(root) / "samples").glob("*.json")):
        if path.name.endswith(".scene.json"):
            continue
        record = read_json(path)
        if path.stem != record.get("sample_id"):
            raise ValueError("Sample filename/ID mismatch")
        records.append(inspect_record(root, record, config_hash))
    return records


def write_index(root, records):
    content = "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records)
    atomic_bytes(Path(root) / "index.jsonl", content.encode("utf-8"))


def append_index(root, record):
    # The record JSON is the atomic commit marker; an interrupted index append
    # is reconstructed from those markers on resume and replaced at completion.
    with (Path(root) / "index.jsonl").open("ab") as stream:
        stream.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def finalize(root, job, records):
    root = Path(root)
    write_index(root, records)
    split_counts = {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")}
    valid = sum(bool(record["metrics"]["valid"]) for record in records)
    manifest = {"schema_version": "1.0", "job_id": job["id"], "config_hash": job["config_hash"], "backend": job["backend"],
                "sample_count": len(records), "valid_count": valid, "expected_samples": job["progress"]["total"], "split_counts": split_counts,
                "group_count": len({r["group_id"] for r in records}), "empty_splits": [split for split, count in split_counts.items() if count == 0],
                "recipe": job["recipe"], "provenance": job.get("provenance", {}),
                "runtime": records[0]["metrics"].get("runtime", {}) if records else {}}
    atomic_json(root / "manifest.json", manifest)
    card = f"# Synthetic RF dataset\n\nJob: {job['id']}\n\nBackend: {job['backend']}\n\nSamples: {len(records)} ({valid} with paths).\n\nSplit counts: {split_counts}.\n\nAbsolute delays in seconds; complex voltage coefficients; CFR at baseband offsets; sinc taps on an absolute-delay grid. Empty-path samples are preserved.\n\nSplits isolate episode groups and matched controls. Small pilots with empty splits do not support reliable evaluation.\n\nSynthetic, uncalibrated data. One isotropic V-polarized port; no transmission, diffraction, or diffuse scattering. Analytical preview treats all boxes as opaque and has no reflections. Sionna uses approximate ITU material fits and finite path sampling.\n"
    atomic_bytes(root / "DATASET_CARD.md", card.encode("utf-8"))
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksums.json" and p.name != "job.json" and not p.name.startswith(".tmp-"))
    atomic_json(root / "checksums.json", {path.relative_to(root).as_posix(): checksum(path) for path in paths})


def _legacy_operative_config(recipe):
    """Frozen pre-program identity from release 5f21cd0; export validation only."""
    scene = copy.deepcopy(recipe["scene"])
    if scene["objects"] is not None:
        scene.pop("object_count")
    radio = copy.deepcopy(recipe["radio"])
    for endpoint in ("tx", "rx"):
        radio[endpoint + "_positions_m"] = radio[endpoint + "_positions_m"] or [
            radio[endpoint + "_position_m"]
        ]
        radio.pop(endpoint + "_position_m")
    v = recipe["variation"]
    parameters = {
        "none": (),
        "translate": ("targets", "offset_m"),
        "height": ("targets", "offset_m"),
        "count": ("count",),
        "scale": ("targets", "factor"),
        "material": ("targets", "material"),
        "aperture": ("targets", "aperture_m"),
        "remove_partition": ("targets",),
        "insert_partition": ("material",),
        "rotate_partition": ("targets", "angle_rad"),
        "carrier": ("values",),
        "bandwidth": ("values",),
        "material_scale": ("targets", "material", "factor"),
    }
    variation = {
        "strategy": v["strategy"],
        **{k: copy.deepcopy(v[k]) for k in parameters[v["strategy"]]},
    }
    if v["strategy"] in (
        "translate",
        "height",
        "scale",
        "material_scale",
        "aperture",
        "insert_partition",
        "rotate_partition",
    ):
        variation["episode_factors"] = copy.deepcopy(v["episode_factors"])
    if v["strategy"] in ("material", "material_scale"):
        variation["episode_materials"] = copy.deepcopy(v["episode_materials"])
    if v["strategy"] in ("count", "remove_partition"):
        variation["episode_counts"] = copy.deepcopy(v["episode_counts"])
    if v["strategy"] == "height":
        variation["offset_m"] = [0.0, 0.0, v["offset_m"][2]]

    def effective_motion(m):
        if not m or m["kind"] == "none":
            return {"kind": "none"}
        keys = {
            "line": ("direction", "speed_m_s"),
            "circle": ("direction", "radius_m", "turns"),
            "helix": ("direction", "radius_m", "turns", "speed_m_s"),
            "sinusoid": ("direction", "amplitude_m", "turns"),
            "waypoints": ("waypoints_m", "speed_m_s"),
            "rotation": ("rotation_axis", "angular_speed_rad_s"),
        }[m["kind"]]
        result = {
            "target": m["target"],
            "kind": m["kind"],
            **{k: copy.deepcopy(m[k]) for k in keys},
        }
        if m["kind"] in ("line", "circle", "helix", "sinusoid"):
            result["episode_directions"] = copy.deepcopy(m["episode_directions"])
        if m["kind"] == "rotation":
            result["episode_angular_speeds_rad_s"] = copy.deepcopy(
                m["episode_angular_speeds_rad_s"]
            )
        for key in ("direction", "rotation_axis"):
            if key in result:
                norm = sum(x * x for x in result[key]) ** 0.5
                result[key] = [x / norm for x in result[key]]
        return result

    motion = {
        "primary": effective_motion(recipe["motion"]),
        "secondary": effective_motion(recipe["motion"]["secondary"]),
    }
    return {
        "scene": scene,
        "radio": radio,
        "variation": variation,
        "motion": motion,
        "constraints": copy.deepcopy(recipe["constraints"]),
    } | {
        "matched_controls": recipe["generation"]["matched_controls"],
        "dt_s": recipe["generation"]["dt_s"],
    }


# Schema 1.0 originally omitted program and timing. Select compatibility only
# for the complete normalized historical shape and its recorded source pins.
# Never use this adapter for generation or job resume.
_LEGACY_SOURCE_PINS = {
    "src/rfdf/compiler.py": "6f8f17529001af4de8ea4693e3fd10cbb72e1db80ad3a1b1fb5b1db6f9284054",
    "src/rfdf/schema.py": "1039041535d399dca1e466ca4726f7fe82e9cf2ca85febf028156ab69b1b8d5b",
}
_PROGRAM_MOTION_FIELDS = (
    "duration_s", "progress_knots", "progress_curve", "phase_offset_rad",
    "waypoint_times_s", "spin_axis", "spin_rate_rad_s",
)


def _dataset_identity(manifest):
    from .compiler import compile_recipe, config_hash, operative_config

    raw = manifest["recipe"]
    plan = compile_recipe(raw)
    normalized = plan["recipe"]
    sources = manifest.get("provenance", {}).get("sources_sha256", {})
    legacy = (
        manifest.get("schema_version") == "1.0"
        and "program" not in raw
        and all(sources.get(name) == digest for name, digest in _LEGACY_SOURCE_PINS.items())
    )
    if legacy:
        old_shape = copy.deepcopy(normalized)
        del old_shape["program"]
        for motion in (old_shape["motion"], old_shape["motion"]["secondary"]):
            if motion is not None:
                for field in _PROGRAM_MOTION_FIELDS:
                    del motion[field]
        # A missing field in a modern recipe is not a legacy normalized export.
        if config_hash(old_shape) != config_hash(raw):
            raise ValueError("Historical recipe shape mismatch")
        digest = config_hash(raw)
        structure = config_hash(_legacy_operative_config(raw))
    else:
        digest = plan["config_hash"]
        structure = config_hash(operative_config(normalized))
    if digest != manifest["config_hash"]:
        raise ValueError("Manifest recipe hash mismatch")
    return normalized, digest, structure


def _split_for_identity(recipe, structure, episode):
    digest = hashlib.sha256(
        f"{structure}:{recipe['generation']['seed']}:{episode}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name in ("train", "validation", "test"):
        cumulative += recipe["splits"][name]
        if value < cumulative:
            return name
    return "test"


def validate_dataset(path):
    root = Path(path)
    errors, warnings, records = [], [], []
    try:
        manifest = read_json(root / "manifest.json")
        records = [json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        seen, groups, physical_splits = set(), {}, {}
        recipe, digest, structure = _dataset_identity(manifest)
        if read_json(root / "recipe.json") != manifest["recipe"]:
            raise ValueError("Saved recipe/manifest mismatch")
        for record in records:
            inspect_record(root, record, manifest["config_hash"])
            if record["sample_id"] in seen:
                raise ValueError("Duplicate sample ID")
            seen.add(record["sample_id"])
            if record["group_id"] in groups and groups[record["group_id"]] != record["split"]:
                raise ValueError("Episode/control group leaks across splits")
            groups[record["group_id"]] = record["split"]
            expected_group = (
                f"{structure[:16]}-{recipe['generation']['seed']}-e{record['episode']:06d}"
            )
            if record["group_id"] != expected_group:
                raise ValueError("Incorrect deterministic group assignment")
            expected_split = _split_for_identity(recipe, structure, record["episode"])
            if record["split"] != expected_split:
                raise ValueError("Incorrect deterministic split assignment")
            if record.get("physical_state_id"):
                physical_splits.setdefault(record["physical_state_id"], set()).add(record["split"])
        from .scenes import iter_snapshots
        # Motion/geometry evaluation uses current normalization; saved sample
        # identity continues to use the export's validated configuration hash.
        expected = {
            digest[:16] + snapshot["sample_id"][16:] for snapshot in iter_snapshots(recipe)
        }
        if seen != expected:
            raise ValueError("Missing or unexpected sample IDs")
        if len(records) != manifest["sample_count"] or len(records) != manifest["expected_samples"]:
            raise ValueError("Incomplete sample coverage")
        if sum(bool(r["metrics"]["valid"]) for r in records) != manifest["valid_count"]:
            raise ValueError("Invalid manifest path count")
        checksums = read_json(root / "checksums.json")
        mandatory = {"manifest.json", "index.jsonl", "recipe.json", "DATASET_CARD.md"}
        for record in records:
            mandatory.update(record["files"].values())
            mandatory.add(f"samples/{record['sample_id']}.json")
        if not mandatory.issubset(checksums):
            raise ValueError("Incomplete dataset checksum inventory")
        for relative, digest in checksums.items():
            if checksum(_local_path(root, relative)) != digest:
                raise ValueError(f"Dataset checksum mismatch: {relative}")
        sample_records = committed_records(root, manifest["config_hash"])
        if {r["sample_id"] for r in sample_records} != seen:
            raise ValueError("Index/sample coverage mismatch")
        committed = {record["sample_id"]: record for record in sample_records}
        if any(committed[record["sample_id"]] != record for record in records):
            raise ValueError("Index/sample metadata mismatch")
        for split in manifest.get("empty_splits", []):
            warnings.append(f"Empty {split} split; pilot cannot evaluate this split")
        repeated = sum(len(splits) > 1 for splits in physical_splits.values())
        if repeated:
            warnings.append(f"{repeated} physical states repeat across splits; episode isolation does not imply unseen-state evaluation")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "warnings": warnings, "sample_count": len(records),
            "cross_split_repeated_physical_states": sum(len(splits) > 1 for splits in physical_splits.values()) if "physical_splits" in locals() else 0,
            "valid_path_count": sum(bool(r.get("metrics", {}).get("valid")) for r in records)}


def export_dataset(root, destination=None):
    root = Path(root)
    report = validate_dataset(root)
    if not report["valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(report["errors"]))
    destination = Path(destination) if destination else root.parent / f"{root.name}.zip"
    fd, temporary = tempfile.mkstemp(prefix=".tmp-export-", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.name.startswith(".tmp-"):
                    archive.write(path, path.relative_to(root).as_posix())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
