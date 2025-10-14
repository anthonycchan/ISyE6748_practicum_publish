# utils_mem.py
import os
import sys
import time
import threading
import platform
from contextlib import contextmanager

# ---------------------------
# Optional deps / globals
# ---------------------------
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil, _PROC = None, None

_IS_WINDOWS = os.name == "nt"
_IS_DARWIN = platform.system() == "Darwin"


# ---------------------------
# Helpers (platform fallbacks)
# ---------------------------
def _procfs_rss_bytes_fallback():
    """
    Best-effort RSS from /proc or resource when psutil is missing.
    Works on Linux (/proc/self/statm) and as a last resort on Unix via resource.
    """
    # Linux procfs
    try:
        with open("/proc/self/statm", "r") as f:
            parts = f.read().strip().split()
            if len(parts) >= 2:
                pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return pages * page_size
    except Exception:
        pass

    # Generic Unix fallback via resource (ru_maxrss is KB on Linux, bytes on macOS)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        ru_bytes = ru.ru_maxrss * (1 if _IS_DARWIN else 1024)
        # Note: ru_maxrss is *peak*; for lack of a better option we return it.
        return int(ru_bytes)
    except Exception:
        return None


def _win_workingset_bytes():
    """
    Windows-only: use ctypes to query current Working Set and Peak Working Set
    when psutil is missing or unavailable.

    Returns: (working_set_bytes, peak_working_set_bytes) or (None, None) on failure.
    """
    if not _IS_WINDOWS:
        return (None, None)

    try:
        import ctypes
        from ctypes import wintypes

        # Structures and API setup
        # Based on PROCESS_MEMORY_COUNTERS from psapi.h
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        GetCurrentProcess = kernel32.GetCurrentProcess
        GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        hp = GetCurrentProcess()
        ok = GetProcessMemoryInfo(hp, ctypes.byref(counters), counters.cb)
        if not ok:
            return (None, None)

        return (int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize))

    except Exception:
        return (None, None)


def _rss_bytes():
    """
    Return resident memory usage in bytes (best effort, cross-platform).

    Priority:
      1) psutil rss (all platforms)
      2) Windows: WorkingSet via ctypes as an rss proxy
      3) Linux/Unix: /proc + resource fallback
    """
    # Preferred: psutil
    if _PROC is not None:
        try:
            return int(_PROC.memory_info().rss)
        except Exception:
            # fall through to platform-specific paths
            pass

    # Windows: use Working Set as a proxy for RSS
    if _IS_WINDOWS:
        ws, _peak_ws = _win_workingset_bytes()
        return ws  # may be None; caller handles it

    # Unix-like fallback paths
    return _procfs_rss_bytes_fallback()


def _workingset_bytes_windows():
    """
    Windows-only helper to read current Working Set (and keep Peak for printing).
    Returns tuple (working_set_bytes, peak_working_set_bytes).
    On non-Windows returns (None, None).
    """
    if not _IS_WINDOWS:
        return (None, None)

    # If psutil is available, we can still prefer its rss as WS proxy,
    # but also fetch true WS via ctypes to expose as an extra metric.
    ws_now, ws_peak = _win_workingset_bytes()
    return (ws_now, ws_peak)


def _cuda_peak_bytes():
    """Read CUDA peak allocated memory if torch is available and a GPU exists."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except Exception:
        pass
    return 0


# ---------------------------
# Public API
# ---------------------------
class MemRecord:
    def __init__(self):
        self.peak_rss_bytes = 0
        self.peak_workingset_bytes = 0  # meaningful on Windows
        self.peak_cuda_bytes = 0


@contextmanager
def peak_ram(prefix=None, label=None, interval=0.02, with_cuda=True, track_workingset=True):
    """
    Sample process memory usage while executing a code block.

    Args:
        prefix (str): Printed prefix label (e.g., pipeline name).
        label (str): Printed secondary label (e.g., "rank=40").
        interval (float): Sampling period in seconds.
        with_cuda (bool): If True, reset and report CUDA peak allocated memory.
        track_workingset (bool): If True on Windows, also track Working Set.
    """
    rec = MemRecord()
    stop = [False]
    t = None
    cuda_enabled = False

    try:
        # CUDA setup (non-fatal if torch isn't present)
        if with_cuda:
            try:
                import torch
                cuda_enabled = torch.cuda.is_available()
                if cuda_enabled:
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                cuda_enabled = False

        # Make an initial synchronous sample to avoid "race-to-zero" peaks
        first_rss = _rss_bytes()
        if first_rss is not None:
            rec.peak_rss_bytes = max(rec.peak_rss_bytes, int(first_rss))

        if track_workingset and _IS_WINDOWS:
            ws_now, ws_peak = _workingset_bytes_windows()
            if ws_now is not None:
                rec.peak_workingset_bytes = max(rec.peak_workingset_bytes, int(ws_now))
            if ws_peak is not None:
                rec.peak_workingset_bytes = max(rec.peak_workingset_bytes, int(ws_peak))

        # Background sampler
        def _sampler():
            local_peak_rss = rec.peak_rss_bytes or 0
            local_peak_ws = rec.peak_workingset_bytes or 0

            while not stop[0]:
                v = _rss_bytes()
                if v is not None and v > local_peak_rss:
                    local_peak_rss = v

                if track_workingset and _IS_WINDOWS:
                    ws_now, ws_peak = _workingset_bytes_windows()
                    if ws_now is not None and ws_now > local_peak_ws:
                        local_peak_ws = ws_now
                    if ws_peak is not None and ws_peak > local_peak_ws:
                        local_peak_ws = ws_peak

                time.sleep(interval)

            rec.peak_rss_bytes = local_peak_rss
            if track_workingset and _IS_WINDOWS:
                rec.peak_workingset_bytes = local_peak_ws

        t = threading.Thread(target=_sampler, daemon=True)
        t.start()

        yield rec

    finally:
        # Stop sampler
        stop[0] = True
        if t is not None:
            t.join(timeout=0.25)

        # CUDA peak
        if cuda_enabled:
            rec.peak_cuda_bytes = _cuda_peak_bytes()

        # ---------------------------
        # Formatting & Printing
        # ---------------------------
        def _fmt_mb(v):
            return f"{v / (1024 ** 2):.1f}"

        # RSS
        rss_line = "n/a"
        if rec.peak_rss_bytes is not None and rec.peak_rss_bytes > 0:
            rss_line = _fmt_mb(rec.peak_rss_bytes)

        # Working Set (Windows only)
        ws_line = None
        if _IS_WINDOWS and rec.peak_workingset_bytes and rec.peak_workingset_bytes > 0:
            ws_line = _fmt_mb(rec.peak_workingset_bytes)

        # CUDA
        cuda_line = None
        if cuda_enabled:
            cuda_line = _fmt_mb(rec.peak_cuda_bytes if rec.peak_cuda_bytes else 0)

        # Compose output line
        line = f"[{prefix}]" if prefix else "[peak_ram]"
        if label:
            line += f" {label}"
        line += f" peak_rss_mb={rss_line}"
        if ws_line is not None:
            line += f" | peak_workingset_mb={ws_line}"
        if cuda_line is not None:
            line += f" | peak_cuda_mb={cuda_line}"

        sys.stdout.write(line + "\n")
        sys.stdout.flush()
