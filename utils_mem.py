# utils_mem.py
from contextlib import contextmanager
from types import SimpleNamespace
import threading, time, os, sys

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil, _PROC = None, None

def _rss_bytes():
    if _PROC is None: return None
    try: return int(_PROC.memory_info().rss)
    except Exception: return None

def _cuda_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            return True
    except Exception:
        pass
    return False

def _cuda_read():
    try:
        import torch
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return 0

@contextmanager
def peak_ram(prefix=None, label=None, interval=0.02, with_cuda=True):
    rec = SimpleNamespace(
        label=label,
        peak_rss_bytes=0,
        peak_cuda_bytes=0,
        peak_workingset_bytes=None,  # may be filled on Windows
    )
    # start values
    rec.peak_rss_bytes = _rss_bytes() or 0
    stop = [False]

    cuda_enabled = _cuda_reset() if with_cuda else False

    def _sampler():
        local_peak = rec.peak_rss_bytes
        while not stop[0]:
            v = _rss_bytes()
            if v is not None and v > local_peak:
                local_peak = v
            time.sleep(interval)
        rec.peak_rss_bytes = local_peak

    t = None
    try:
        if psutil is not None:
            t = threading.Thread(target=_sampler, daemon=True)
            t.start()
        yield rec                    # <<< you can access this after the with-block
    finally:
        stop[0] = True
        if t is not None:
            t.join(timeout=0.2)
        if cuda_enabled:
            rec.peak_cuda_bytes = _cuda_read()
        # optional: Windows lifetime peak working set
        try:
            mi = _PROC.memory_full_info()
            if hasattr(mi, "peak_wset"):
                rec.peak_workingset_bytes = int(mi.peak_wset)
        except Exception:
            pass
        # concise print for logs
        mb = rec.peak_rss_bytes / (1024**2) if rec.peak_rss_bytes else None
        gmb = rec.peak_cuda_bytes / (1024**2) if rec.peak_cuda_bytes else 0
        line = f"[{prefix}]{' '+label if label else ''} peak_rss_mb="
        line += f"{mb:.1f}" if mb is not None else "?"
        if rec.peak_workingset_bytes:
            line += f" | peak_workingset_mb={rec.peak_workingset_bytes/(1024**2):.1f}"
        if with_cuda:
            line += f" | peak_cuda_mb={gmb:.1f}"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
