"""Shared model profile loading, validation, and run selection."""
from __future__ import annotations

import re
from pathlib import Path

from . import REPO_ROOT
from .harness import NUM_CTX

DEFAULT_PROFILES_DIR = REPO_ROOT / "profiles"


def canonical_model_name(name: str) -> str:
    """Treat an omitted tag and :latest as the same installed model."""
    if name.endswith(":latest"):
        return name[: -len(":latest")]
    return name


def display_source(path: Path) -> str:
    """Prefer a repo-relative path so reports stay portable."""
    repo_root = REPO_ROOT
    resolved = path.resolve()
    if resolved.is_relative_to(repo_root):
        return str(resolved.relative_to(repo_root))
    return str(resolved)


# Keys are matched exactly. A different case, hyphen, or space is an error.
PROFILE_FIELDS = (
    "model",
    "num_ctx",
    "thinking",
    "temperature",
    "top_p",
    "top_k",
    "weight_precision",
    "weight_quant",
    "kv_cache",
    "repeat_penalty",
)
PROFILE_FIELD_NAMES = frozenset(PROFILE_FIELDS)


def require_profile_field(label: str, source: str, number: int) -> str:
    """Return the key when it is one of the exact profile field names."""
    if label not in PROFILE_FIELD_NAMES:
        allowed = ", ".join(PROFILE_FIELDS)
        raise ValueError(
            f"{source}:{number}: unknown field {label!r}; expected an exact key: {allowed}"
        )
    return label


def parse_profile_text(text: str, source: str) -> dict:
    """Parse a declared profile. Each line is 'key=value', as in a .env file."""
    if not text.strip():
        raise ValueError(f"{source}: profile is empty")
    fields: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{source}:{number}: expected 'key=value'")
        label, value = line.split("=", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            raise ValueError(f"{source}:{number}: expected 'key=value'")
        key = require_profile_field(label, source, number)
        if key in fields:
            raise ValueError(f"{source}:{number}: duplicate field {label}")
        fields[key] = value
    if "model" not in fields:
        raise ValueError(f"{source}: missing Model")
    return {"model": fields["model"], "source": source, "text": text, "fields": fields}


def parse_profile_file(path: Path) -> dict:
    return parse_profile_text(path.read_text(encoding="utf-8"), display_source(path))


def parse_num_ctx(value: str, source: str) -> int:
    """Parse a profile num_ctx as one positive integer. Commas are not allowed."""
    text = value.strip()
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(
            f"{source}: num_ctx must be a positive integer with no commas (got {value!r})"
        )
    return int(text)


def profile_num_ctx(profile: dict | None) -> int:
    """Context window for one setup. A missing num_ctx line uses NUM_CTX."""
    if profile is None:
        return NUM_CTX
    raw = profile.get("fields", {}).get("num_ctx")
    if raw is None:
        return NUM_CTX
    return parse_num_ctx(raw, profile.get("source") or "profile")


def parse_thinking(value: str, source: str) -> bool | str:
    """Parse a profile thinking line into the Ollama think argument."""
    text = value.strip().lower()
    if text in {"true", "enabled", "on", "yes"}:
        return True
    if text in {"false", "disabled", "off", "no"}:
        return False
    if text in {"low", "medium", "high"}:
        return text
    raise ValueError(
        f"{source}: thinking must be true, false, low, medium, or high (got {value!r})"
    )


def profile_think(profile: dict | None) -> bool | str | None:
    """Think argument for one setup. A missing line uses THINK from shared/harness.py."""
    if profile is None:
        return None
    raw = profile.get("fields", {}).get("thinking")
    if raw is None:
        return None
    return parse_thinking(raw, profile.get("source") or "profile")


_PLAIN_DECIMAL = re.compile(r"(?:0|[1-9]\d*)(?:\.\d+)?")


def parse_plain_decimal(value: str, source: str, field: str) -> float:
    """Parse a profile decimal written as digits, with an optional fraction."""
    text = value.strip()
    if not _PLAIN_DECIMAL.fullmatch(text):
        raise ValueError(
            f"{source}: {field} must be a non-negative decimal with no commas "
            f"(got {value!r})"
        )
    return float(text)


def parse_temperature(value: str, source: str) -> float:
    return parse_plain_decimal(value, source, "temperature")


def parse_top_p(value: str, source: str) -> float:
    number = parse_plain_decimal(value, source, "top_p")
    if number > 1:
        raise ValueError(
            f"{source}: top_p must be a decimal from 0 through 1 (got {value!r})"
        )
    return number


def parse_top_k(value: str, source: str) -> int:
    """Parse a profile top_k as one positive integer. Commas are not allowed."""
    text = value.strip()
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(
            f"{source}: top_k must be a positive integer with no commas (got {value!r})"
        )
    return int(text)


def profile_sampling(profile: dict | None) -> dict:
    """Sampling options for one setup. Missing lines keep the harness defaults."""
    if profile is None:
        return {}
    fields = profile.get("fields", {})
    source = profile.get("source") or "profile"
    sampling: dict = {}
    if "temperature" in fields:
        sampling["temperature"] = parse_temperature(fields["temperature"], source)
    if "top_p" in fields:
        sampling["top_p"] = parse_top_p(fields["top_p"], source)
    if "top_k" in fields:
        sampling["top_k"] = parse_top_k(fields["top_k"], source)
    return sampling


def load_profiles(directory: Path) -> list[dict]:
    """Read every *.profile file. A missing directory means nothing was recorded.

    Several files may name the same model. Those are different setups, not duplicates.
    """
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"Profiles path is not a directory: {directory}")
    problems: list[str] = []
    found: list[dict] = []
    for path in sorted(directory.glob("*.profile")):
        try:
            found.append(parse_profile_file(path))
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError("Profile check failed:\n" + "\n".join(problems))
    return found


def load_profile_paths(paths: list[Path]) -> list[dict]:
    """Parse the profile files chosen for this run. The same model may appear more than once."""
    problems: list[str] = []
    found: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        try:
            if not path.is_file():
                raise ValueError(f"Profile not found: {path}")
            profile = parse_profile_file(path)
        except ValueError as error:
            problems.append(str(error))
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            problems.append(f"Profile listed more than once: {profile['source']}")
            continue
        seen.add(resolved)
        found.append(profile)
    if problems:
        raise ValueError("Profile check failed:\n" + "\n".join(problems))
    return found


def model_names_match(left: str, right: str) -> bool:
    return canonical_model_name(left) == canonical_model_name(right)


def profiles_for_model(profiles: list[dict], model_name: str) -> list[dict]:
    return [profile for profile in profiles if model_names_match(profile["model"], model_name)]


def unique_profile(profiles: list[dict], model_name: str) -> dict | None:
    """Return the only profile for a model. Several matches must be chosen with --profile."""
    matches = profiles_for_model(profiles, model_name)
    if len(matches) > 1:
        listed = "\n".join(profile["source"] for profile in matches)
        raise ValueError(
            f"Multiple profiles match {model_name}:\n{listed}\n"
            "Pass --profile with the file for this run."
        )
    return matches[0] if matches else None


def weight_tokens(value: str) -> set[str]:
    """Split a weight-precision note into comparable quant tokens."""
    tokens: set[str] = set()
    current: list[str] = []
    for char in value:
        if char.isalnum() or char == "_":
            current.append(char.lower())
        elif current:
            tokens.add("".join(current))
            current = []
    if current:
        tokens.add("".join(current))
    return tokens


def declared_weight(fields: dict) -> str | None:
    return fields.get("weight_precision") or fields.get("weight_quant")


def quantization_note(declared: str | None, installed: str | None) -> str | None:
    """Warn when the written quant does not match the installed Ollama model."""
    if not declared or not installed:
        return None
    if installed.lower() in weight_tokens(declared):
        return None
    return (
        f"Declared weight quant is {declared}; installed quantization is {installed}."
    )


def recorded_profile(profile: dict, installed: dict) -> dict:
    """Snapshot the profile chosen for one installed model before inference."""
    recorded = {
        "model": profile["model"],
        "matched_model": installed["name"],
        "source": profile["source"],
        "text": profile["text"],
        "fields": profile["fields"],
        "run_key": profile["source"],
    }
    note = quantization_note(
        declared_weight(profile["fields"]),
        installed.get("quantization_level"),
    )
    if note:
        recorded["quantization_note"] = note
    return recorded


def print_recorded_profile(recorded: dict | None, model_name: str) -> None:
    if recorded is None:
        print(f"No declared profile for {model_name}", flush=True)
        return
    print(f"Declared profile ({recorded['source']}):", flush=True)
    print(recorded["text"].rstrip(), flush=True)
    note = recorded.get("quantization_note")
    if note:
        print(note, flush=True)


def assign_run_slots(
    selected: list[dict], loaded: list[dict], explicit: list[dict]
) -> list[dict]:
    """One slot per setup. Two profiles for one model stay separate runs."""
    problems: list[str] = []
    selected_keys = {canonical_model_name(model["name"]) for model in selected}
    for profile in explicit:
        if canonical_model_name(profile["model"]) not in selected_keys:
            problems.append(
                f"Profile {profile['source']} declares {profile['model']}, which is not selected"
            )
    slots = []
    for model in selected:
        pinned = profiles_for_model(explicit, model["name"]) if explicit else []
        if pinned:
            chosen: list[dict | None] = list(pinned)
        else:
            try:
                one = unique_profile(loaded, model["name"])
            except ValueError as error:
                problems.append(str(error))
                continue
            chosen = [one]
        for profile in chosen:
            try:
                num_ctx = profile_num_ctx(profile)
                think = profile_think(profile)
                sampling = profile_sampling(profile)
            except ValueError as error:
                problems.append(str(error))
                continue
            if think is not None and "thinking" not in (model.get("capabilities") or []):
                source = (profile or {}).get("source") or "profile"
                problems.append(
                    f"{source}: thinking is set but {model['name']} does not "
                    "list the thinking capability"
                )
                continue
            slot = dict(model)
            slot["num_ctx"] = num_ctx
            slot["think"] = think
            slot["sampling"] = sampling
            if profile is None:
                slot["run_key"] = model["name"]
                slot["profile_source"] = None
                slot["declared_profile"] = None
                print_recorded_profile(None, model["name"])
            else:
                recorded = recorded_profile(profile, model)
                slot["run_key"] = recorded["run_key"]
                slot["profile_source"] = recorded["source"]
                slot["declared_profile"] = recorded
                print_recorded_profile(recorded, model["name"])
            slots.append(slot)
    if problems:
        raise ValueError("Profile check failed:\n" + "\n".join(problems))
    return slots
