"""Shared validation of installed Ollama model formats."""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
import tomllib

from ollama import Client


@dataclass(frozen=True)
class ModelFormat:
    label: str
    pattern: re.Pattern[str]
    markers: tuple[str, ...] = ()
    renderers: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    hint: str = ""
    parsers: tuple[str, ...] | None = None
    required_stops: tuple[str, ...] = ()
    allowed_stops: tuple[str, ...] | None = None


def load_model_formats(path: Path) -> tuple[ModelFormat, ...]:
    """Load declarative rules, rejecting typos that would silently weaken checks."""
    with path.open("rb") as source:
        data = tomllib.load(source)
    if set(data) != {"formats"} or not isinstance(data["formats"], list) or not data["formats"]:
        raise ValueError(f"{path}: expected at least one [[formats]] rule")
    known = {"label", "name_pattern", "markers", "renderers", "forbidden", "hint",
             "parsers", "required_stops", "allowed_stops"}
    rules = []
    labels = set()
    for entry in data["formats"]:
        if not isinstance(entry, dict) or set(entry) - known:
            raise ValueError(f"{path}: unknown model format fields")
        for key in ("label", "name_pattern"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"{path}: {key} must be a nonempty string")
        if entry["label"] in labels:
            raise ValueError(f"{path}: duplicate model format {entry['label']!r}")
        labels.add(entry["label"])
        options = {}
        for key in known - {"label", "name_pattern", "hint"}:
            if key in entry:
                value = entry[key]
                if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
                    raise ValueError(f"{path}: {key} must be an array of nonempty strings")
                options[key] = tuple(value)
        if not isinstance(entry.get("hint", ""), str):
            raise ValueError(f"{path}: hint must be a string")
        try:
            pattern = re.compile(entry["name_pattern"], re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"{path}: invalid name_pattern: {error}") from error
        rule = ModelFormat(entry["label"], pattern, hint=entry.get("hint", ""), **options)
        if not rule.markers and not rule.renderers:
            raise ValueError(f"{path}: {rule.label} needs markers or renderers")
        if rule.markers and rule.renderers:
            raise ValueError(f"{path}: {rule.label} must choose markers or renderers")
        if rule.allowed_stops is not None and not set(rule.required_stops) <= set(rule.allowed_stops):
            raise ValueError(f"{path}: required_stops must be included in allowed_stops")
        rules.append(rule)
    return tuple(rules)


MODEL_FORMATS = load_model_formats(Path(__file__).with_name("model_formats.toml"))


def is_bare_prompt_template(template: str | None) -> bool:
    """True when Ollama will send chat text without native role/turn markers."""
    collapsed = re.sub(r"\s+", "", template or "")
    return collapsed in {"{{.Prompt}}", "{{.Prompt}}{{.Response}}", ""}


def installed_renderer(modelfile: str | None) -> str | None:
    """Return the Modelfile RENDERER name when Ollama frames chat with one."""
    return installed_directive(modelfile, "RENDERER")


def installed_directive(modelfile: str | None, directive: str) -> str | None:
    """Read top-level instructions, ignoring text inside multiline template/system blocks."""
    if not isinstance(modelfile, str):
        return None
    instructions = re.sub(r'""".*?"""', '""', modelfile, flags=re.DOTALL)
    for line in instructions.splitlines():
        parts = line.strip().split(None, 1)
        if parts and parts[0].upper() == directive:
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError(f"empty {directive} directive")
            values = shlex.split(parts[1], comments=True)
            if len(values) != 1:
                raise ValueError(f"invalid {directive} directive")
            return values[0]
    return None


def expected_chat_framing(model: str, formats: tuple[ModelFormat, ...] | None = None) -> ModelFormat | None:
    """Return the native framing registered for this installed model name."""
    for framing in MODEL_FORMATS if formats is None else formats:
        if framing.pattern.search(model):
            return framing
    return None


def framing_expectation(framing: ModelFormat) -> str:
    if framing.hint:
        return framing.hint
    if framing.renderers:
        return f"{framing.label} expects RENDERER {', '.join(framing.renderers)}."
    return f"{framing.label} expects {', '.join(framing.markers)}."


def chat_template_error(
    model: str, template: str | None, renderer: str | None = None,
    *, formats: tuple[ModelFormat, ...] | None = None,
) -> str | None:
    """Return an error if the installed framing is not the one this model was trained with."""
    framing = expected_chat_framing(model, formats)
    if framing is None:
        return (
            f"No expected chat template is registered for {model!r}. "
            "Preflight will not prompt a model whose native framing is unknown."
        )
    text = template or ""
    installed = text.strip() or "(empty)"
    expectation = framing_expectation(framing)
    if framing.renderers:
        if renderer not in framing.renderers:
            shown = renderer or "(none)"
            return (
                f"Installed Ollama model {model!r} has RENDERER {shown}. "
                f"{expectation} "
                f"Installed template: {installed}"
            )
        return None
    if renderer:
        return (
            f"Installed Ollama model {model!r} has RENDERER {renderer!r}, "
            "which replaces the chat template. "
            f"{expectation} "
            f"Installed template: {installed}"
        )
    forbidden = [marker for marker in framing.forbidden if marker in text]
    if forbidden:
        return (
            f"Installed Ollama template for {model!r} contains "
            f"{', '.join(forbidden)}, which is the wrong framing. "
            f"{expectation} "
            f"Installed template: {installed}"
        )
    missing = [marker for marker in framing.markers if marker not in text]
    if missing:
        return (
            f"Installed Ollama template for {model!r} is missing "
            f"{', '.join(missing)}. {expectation} "
            f"Installed template: {installed}"
        )
    return None


def installed_model_error(
    model: str, info, *, formats: tuple[ModelFormat, ...] | None = None,
) -> str | None:
    """Apply template, parser, and stop policies without model-specific branches."""
    modelfile = getattr(info, "modelfile", None)
    try:
        renderer = installed_renderer(modelfile)
        parser = installed_directive(modelfile, "PARSER")
    except ValueError as error:
        return f"{model!r} has malformed Modelfile instructions: {error}"
    error = chat_template_error(model, info.template or "", renderer, formats=formats)
    if error:
        return error
    framing = expected_chat_framing(model, formats)
    assert framing is not None  # Unknown models already failed template validation.
    if framing.parsers is not None:
        if not isinstance(modelfile, str):
            return f"{model!r} is missing Modelfile metadata needed to validate its parser. {framing.hint}"
        if not framing.parsers and parser:
            return f"{model!r} must use plain-text output without a PARSER directive. {framing.hint}"
        if framing.parsers and parser not in framing.parsers:
            return f"{model!r} expects PARSER {', '.join(framing.parsers)}; installed: {parser!r}. {framing.hint}"
    if not framing.required_stops and framing.allowed_stops is None:
        return None
    parameters = getattr(info, "parameters", None)
    stops = []
    if isinstance(parameters, str):
        try:
            for line in parameters.splitlines():
                head = line.strip().split(None, 1)
                if head and head[0].lower() == "stop":
                    # /api/show quotes strings with escapes; preserve newline,
                    # backslash, and Unicode stop sequences from that metadata.
                    if len(head) == 2 and head[1].startswith('"'):
                        value = json.loads(head[1])
                        if not isinstance(value, str):
                            raise ValueError("invalid stop parameter")
                        stops.append(value)
                        continue
                    parts = shlex.split(line)
                    if len(parts) != 2:
                        raise ValueError("invalid stop parameter")
                    stops.append(parts[1])
        except ValueError:
            return f"{model!r} has malformed stop parameters. {framing.hint}"
    required = set(framing.required_stops)
    allowed = set(framing.allowed_stops) if framing.allowed_stops is not None else None
    actual = set(stops)
    if allowed == required and actual != required:
        return (
            f"{model!r} must use only stop {', '.join(framing.required_stops) or '(none)'}; "
            f"installed stops: {stops!r}. {framing.hint}"
        )
    if missing := required - actual:
        return f"{model!r} is missing required stops {sorted(missing)!r}. {framing.hint}"
    if allowed is not None and (unexpected := actual - allowed):
        return f"{model!r} has unexpected stops {sorted(unexpected)!r}. {framing.hint}"
    return None


def inspect_installed_model(client: Client, model: str) -> tuple[str, list[str], str | None]:
    """Read /api/show and refuse models whose framing is not the trained one."""
    info = client.show(model)
    template = info.template or ""
    capabilities = list(info.capabilities or [])
    renderer = installed_renderer(getattr(info, "modelfile", None))
    error = installed_model_error(model, info)
    if error:
        raise ValueError(error)
    return template, capabilities, renderer
