"""Safe torch device selection.

Old or unsupported GPUs (for example a GTX 1050, compute capability ``sm_61``)
crash with ``cudaErrorNoKernelImageForDevice`` when a torch build that was not
compiled for that architecture tries to launch a kernel on them. To keep the
package working out of the box, device selection defaults to CUDA *only* when
the installed torch build actually supports the present GPU; otherwise it uses
Apple MPS (if available) or the CPU.

Override with ``EPISCOPE_DEVICE=cpu|cuda|mps`` or by passing ``device=`` to the
embedder / reranker.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_device(requested: Optional[str] = None) -> str:
    """Return a safe torch device string.

    Resolution order: explicit ``requested`` → ``EPISCOPE_DEVICE`` → ``"auto"``.
    """
    pref = (requested or os.environ.get("EPISCOPE_DEVICE") or "auto").strip().lower()
    if pref in {"cpu", "cuda", "mps"}:
        return pref

    try:
        import torch

        if torch.cuda.is_available() and _cuda_is_supported(torch):
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception as exc:  # torch missing or probe failed → CPU is always safe
        logger.debug("Device auto-detection fell back to CPU: %s", exc)
    return "cpu"


def _cuda_is_supported(torch) -> bool:
    """True only if the GPU's compute capability is covered by this torch build."""
    try:
        major, minor = torch.cuda.get_device_capability(0)
        capability = major * 10 + minor
        built = [
            int(arch[3:])
            for arch in torch.cuda.get_arch_list()
            if arch.startswith("sm_") and arch[3:].isdigit()
        ]
        if not built:
            return True  # cannot determine; trust torch
        if capability < min(built):
            logger.warning(
                "GPU compute capability sm_%d is older than this torch build "
                "(built for %s); using CPU instead. Set EPISCOPE_DEVICE=cuda to "
                "force the GPU.",
                capability,
                sorted(built),
            )
            return False
        return True
    except Exception:
        return False
