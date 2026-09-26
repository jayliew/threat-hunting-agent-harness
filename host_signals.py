"""Sample host pressure while one Ollama chat call is running.

Placement comes from the same data as `ollama ps`. Memory pressure and swap
are macOS readings. Probe failures are ignored so a hunt still finishes.
"""

from __future__ import annotations

import math
import sys
import threading
from time import perf_counter
from typing import Callable

from ollama import Client


SAMPLE_INTERVAL_SECONDS = 1.0
SWAP_GROWTH_BYTES = 100 * 1024 * 1024
PRESSURE_NORMAL = 1
PRESSURE_WARNING = 2
PRESSURE_CRITICAL = 4
_PRESSURE_NAMES = {
    PRESSURE_NORMAL: "normal",
    PRESSURE_WARNING: "warning",
    PRESSURE_CRITICAL: "critical",
}
_PRESSURE_RANK = {
    PRESSURE_NORMAL: 0,
    PRESSURE_WARNING: 1,
    PRESSURE_CRITICAL: 2,
}
SIGNAL_CPU_GPU_SPLIT = "cpu_gpu_split"
SIGNAL_CPU_ONLY = "cpu_only"
SIGNAL_MEMORY_PRESSURE = "memory_pressure"
SIGNAL_SWAP = "swap"
SIGNAL_ORDER = (
    SIGNAL_CPU_GPU_SPLIT,
    SIGNAL_CPU_ONLY,
    SIGNAL_MEMORY_PRESSURE,
    SIGNAL_SWAP,
)
SIGNAL_LABELS = {
    SIGNAL_CPU_GPU_SPLIT: "CPU/GPU split",
    SIGNAL_CPU_ONLY: "100% CPU",
    SIGNAL_MEMORY_PRESSURE: "memory pressure",
    SIGNAL_SWAP: "swap",
}

SampleReader = Callable[[str], dict]


def empty_host() -> dict:
    return {
        "placement": None,
        "pressure_baseline": None,
        "pressure_peak": None,
        "swap_used_bytes_start": None,
        "swap_used_bytes_end": None,
        "signals": [],
    }


def processor_label(size: int, size_vram: int) -> str | None:
    """Match `ollama ps`: CPU% is ceil, GPU% is floor, of size_vram / size."""
    if size <= 0:
        return None
    if size_vram <= 0:
        return "100% CPU"
    if size_vram >= size:
        return "100% GPU"
    gpu_ratio = size_vram / size
    cpu = math.ceil(100 - gpu_ratio * 100)
    gpu = math.floor(gpu_ratio * 100)
    return f"{cpu}%/{gpu}% CPU/GPU"


def _field(model, name: str):
    if isinstance(model, dict):
        return model.get(name)
    return getattr(model, name, None)


def _same_model(requested: str, name: str, model: str) -> bool:
    names = {name, model}
    if requested in names:
        return True
    if ":" not in requested and f"{requested}:latest" in names:
        return True
    return False


def processor_for(models, requested: str) -> str | None:
    for running in models or []:
        name = str(_field(running, "name") or "")
        model = str(_field(running, "model") or "")
        if not _same_model(requested, name, model):
            continue
        size = _field(running, "size")
        size_vram = _field(running, "size_vram")
        if size is None or size_vram is None:
            continue
        try:
            return processor_label(int(size), int(size_vram))
        except (TypeError, ValueError):
            return None
    return None


def _libc():
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.sysctlbyname.restype = ctypes.c_int
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    return libc, ctypes


def read_pressure_level() -> int | None:
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical."""
    if sys.platform != "darwin":
        return None
    try:
        libc, ctypes = _libc()
        value = ctypes.c_int32(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        rc = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            ctypes.c_size_t(0),
        )
        if rc != 0:
            return None
        return int(value.value)
    except Exception:
        return None


def read_swap_used_bytes() -> int | None:
    """vm.swapusage xsu_used, in bytes."""
    if sys.platform != "darwin":
        return None
    try:
        libc, ctypes = _libc()

        class XswUsage(ctypes.Structure):
            _fields_ = [
                ("xsu_total", ctypes.c_uint64),
                ("xsu_avail", ctypes.c_uint64),
                ("xsu_used", ctypes.c_uint64),
                ("xsu_pagesize", ctypes.c_uint32),
                ("xsu_encrypted", ctypes.c_int),
            ]

        usage = XswUsage()
        size = ctypes.c_size_t(ctypes.sizeof(usage))
        rc = libc.sysctlbyname(
            b"vm.swapusage",
            ctypes.byref(usage),
            ctypes.byref(size),
            None,
            ctypes.c_size_t(0),
        )
        if rc != 0:
            return None
        return int(usage.xsu_used)
    except Exception:
        return None


def read_placement(model: str) -> str | None:
    try:
        running = Client(timeout=2).ps()
    except Exception:
        return None
    return processor_for(getattr(running, "models", None), model)


def read_host_sample(model: str) -> dict:
    return {
        "placement": read_placement(model),
        "pressure": read_pressure_level(),
        "swap_used_bytes": read_swap_used_bytes(),
    }


class HostMonitor:
    """Sample once at start, about once a second, and once when stopped."""

    def __init__(
        self,
        model: str,
        *,
        reader: SampleReader | None = None,
        interval: float = SAMPLE_INTERVAL_SECONDS,
    ):
        self.model = model
        self.reader = reader or read_host_sample
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._started = perf_counter()
        self._capture()
        self._thread = threading.Thread(
            target=self._loop, name="host-signals", daemon=True
        )
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.interval + 2)
        if thread is None or not thread.is_alive():
            self._capture()
        with self._lock:
            return list(self.samples)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._capture()

    def _capture(self) -> None:
        elapsed = perf_counter() - self._started
        try:
            reading = self.reader(self.model)
        except Exception:
            reading = {}
        if not isinstance(reading, dict):
            reading = {}
        sample = {
            "t": elapsed,
            "placement": reading.get("placement"),
            "pressure": reading.get("pressure"),
            "swap_used_bytes": reading.get("swap_used_bytes"),
        }
        with self._lock:
            self.samples.append(sample)


def _gpu_percent(label: str) -> int:
    if label == "100% GPU":
        return 100
    if label == "100% CPU":
        return 0
    cpu_and_gpu = label.split(" ", 1)[0]
    return int(cpu_and_gpu.split("/")[1].rstrip("%"))


def _pressure_name(level) -> str | None:
    try:
        return _PRESSURE_NAMES.get(int(level))
    except (TypeError, ValueError):
        return None


def _peak_pressure(samples: list[dict]) -> int | None:
    known = []
    for sample in samples:
        level = sample.get("pressure")
        if level in _PRESSURE_RANK:
            known.append(level)
    if not known:
        return None
    return max(known, key=lambda level: _PRESSURE_RANK[level])


def _placement_warning(labels: list[str]) -> str | None:
    non_gpu = [label for label in labels if label != "100% GPU"]
    if not non_gpu:
        return None
    ordered = sorted(dict.fromkeys(non_gpu), key=_gpu_percent)
    shown = ", ".join(ordered)
    return f"Host: Ollama reports {shown} rather than 100% GPU."


def _pressure_warning(baseline: int | None, peak: int | None) -> str | None:
    if peak not in (PRESSURE_WARNING, PRESSURE_CRITICAL):
        return None
    peak_name = _PRESSURE_NAMES[peak]
    if baseline not in _PRESSURE_NAMES:
        return f"Host: memory pressure reached {peak_name} during inference."
    if baseline == peak:
        return f"Host: memory pressure stayed at {peak_name} during inference."
    if baseline == PRESSURE_NORMAL:
        return f"Host: memory pressure entered {peak_name} during inference."
    if baseline == PRESSURE_WARNING and peak == PRESSURE_CRITICAL:
        return "Host: memory pressure rose from warning to critical during inference."
    return f"Host: memory pressure reached {peak_name} during inference."


def _swap_edges(
    samples: list[dict], steady_start: float
) -> tuple[int | None, int | None]:
    known = []
    for sample in samples:
        used = sample.get("swap_used_bytes")
        elapsed = sample.get("t")
        if not isinstance(used, int) or isinstance(used, bool):
            continue
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
            continue
        known.append((float(elapsed), used))
    before = [item for item in known if item[0] <= steady_start]
    if not before:
        return None, None
    start_t, start_used = before[-1]
    after = [item for item in known if item[0] >= steady_start and item[0] > start_t]
    if not after:
        return None, None
    return start_used, after[-1][1]


def host_signal_labels(signals: list[str]) -> str:
    labels = [SIGNAL_LABELS[name] for name in SIGNAL_ORDER if name in signals]
    return ", ".join(labels) if labels else "—"


def interpret_host_samples(
    samples: list[dict],
    *,
    load_seconds: float | None,
    prompt_eval_seconds: float | None,
) -> dict:
    """Turn samples into report warnings. Missing probes add no warning."""
    labels = [sample.get("placement") for sample in samples if sample.get("placement")]
    baseline = samples[0].get("pressure") if samples else None
    if baseline not in _PRESSURE_RANK:
        baseline = None
    peak = _peak_pressure(samples)
    swap_start = swap_end = None
    if load_seconds is not None and prompt_eval_seconds is not None:
        swap_start, swap_end = _swap_edges(
            samples, float(load_seconds) + float(prompt_eval_seconds)
        )

    signals = []
    warnings = []
    placement_warning = _placement_warning(labels)
    if placement_warning:
        warnings.append(placement_warning)
        if any(label == "100% CPU" for label in labels):
            signals.append(SIGNAL_CPU_ONLY)
        if any(label not in ("100% GPU", "100% CPU") for label in labels):
            signals.append(SIGNAL_CPU_GPU_SPLIT)
    pressure_warning = _pressure_warning(baseline, peak)
    if pressure_warning:
        warnings.append(pressure_warning)
        signals.append(SIGNAL_MEMORY_PRESSURE)
    growth = None
    if swap_start is not None and swap_end is not None:
        growth = swap_end - swap_start
        if growth >= SWAP_GROWTH_BYTES:
            mib = round(growth / (1024 * 1024))
            warnings.append(f"Host: swap grew {mib} MiB during generation.")
            signals.append(SIGNAL_SWAP)

    non_gpu = [label for label in labels if label != "100% GPU"]
    if non_gpu:
        placement = min(dict.fromkeys(non_gpu), key=_gpu_percent)
    elif "100% GPU" in labels:
        placement = "100% GPU"
    else:
        placement = None

    host = empty_host()
    host["placement"] = placement
    host["pressure_baseline"] = _pressure_name(baseline)
    host["pressure_peak"] = _pressure_name(peak)
    host["swap_used_bytes_start"] = swap_start
    host["swap_used_bytes_end"] = swap_end
    host["signals"] = [name for name in SIGNAL_ORDER if name in signals]
    return {"warnings": warnings, "host": host}


def apply_host_signals(result: dict, samples: list[dict]) -> None:
    timing = result.get("timing") or {}
    try:
        interpreted = interpret_host_samples(
            samples,
            load_seconds=timing.get("load_duration_seconds"),
            prompt_eval_seconds=timing.get("prompt_eval_duration_seconds"),
        )
    except Exception:
        return
    result["host"] = interpreted["host"]
    result["warnings"].extend(interpreted["warnings"])
