import os
import sys
import time
import threading
from contextlib import contextmanager

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil, _PROC = None, None


def _procfs_rss_bytes_fallback():
    """Best-effort RSS from /proc or resource when psutil is missing."""
    try:
        with open("/proc/self/statm", "r") as f:
            parts = f.read().strip().split()
            if len(parts) >= 2:
                pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return pages * page_size
    except Exception:
        pass

    try:
        import resource, platform
        ru = resource.getrusage(resource.RUSAGE_SELF)
        ru_bytes = ru.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        return int(ru_bytes)
    except Exception:
        return None


def _rss_bytes():
    """Return resident memory usage in bytes."""
    if _PROC is not None:
        try:
            return int(_PROC.memory_info().rss)
        except Exception:
            pass
    return _procfs_rss_bytes_fallback()


def _cuda_read():
    """Read CUDA allocated memory if torch is available and a GPU exists."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except Exception:
        pass
    return 0


class MemRecord:
    def __init__(self):
        self.peak_rss_bytes = 0
        self.peak_workingset_bytes = 0
        self.peak_cuda_bytes = 0


@contextmanager
def peak_ram(prefix=None, label=None, interval=0.02, with_cuda=True):
    rec = MemRecord()
    stop = [False]
    t = None
    cuda_enabled = False

    try:
        if with_cuda:
            import torch
            cuda_enabled = torch.cuda.is_available()
            if cuda_enabled:
                torch.cuda.reset_peak_memory_stats()

        def _sampler():
            local_peak = rec.peak_rss_bytes
            while not stop[0]:
                v = _rss_bytes()
                if v is not None and v > local_peak:
                    local_peak = v
                time.sleep(interval)
            rec.peak_rss_bytes = local_peak

        t = threading.Thread(target=_sampler, daemon=True)
        t.start()

        yield rec

    finally:
        stop[0] = True
        if t is not None:
            t.join(timeout=0.2)
        if cuda_enabled:
            rec.peak_cuda_bytes = _cuda_read()

        mb = rec.peak_rss_bytes / (1024 ** 2) if rec.peak_rss_bytes else None
        gmb = rec.peak_cuda_bytes / (1024 ** 2) if rec.peak_cuda_bytes else 0

        line = f"[{prefix}]{' ' + label if label else ''} peak_rss_mb="
        line += f"{mb:.1f}" if mb is not None else "n/a"
        if rec.peak_workingset_bytes:
            line += f" | peak_workingset_mb={rec.peak_workingset_bytes / (1024 ** 2):.1f}"
        if with_cuda and cuda_enabled:
            line += f" | peak_cuda_mb={gmb:.1f}"

        sys.stdout.write(line + "\n")
        sys.stdout.flush()
