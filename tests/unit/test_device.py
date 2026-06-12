from __future__ import annotations

import sys

from episcope.rag.device import resolve_device


def test_explicit_request_wins(monkeypatch) -> None:
    monkeypatch.delenv("EPISCOPE_DEVICE", raising=False)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_env_var_selects_device(monkeypatch) -> None:
    monkeypatch.setenv("EPISCOPE_DEVICE", "cpu")
    assert resolve_device() == "cpu"


def test_auto_returns_a_valid_device(monkeypatch) -> None:
    monkeypatch.delenv("EPISCOPE_DEVICE", raising=False)
    assert resolve_device() in {"cpu", "cuda", "mps"}


class _FakeCuda:
    def __init__(self, capability, arch_list):
        self._cap = capability
        self._arch = arch_list

    def is_available(self):
        return True

    def get_device_capability(self, index=0):
        return self._cap

    def get_arch_list(self):
        return self._arch


def _fake_torch(capability, arch_list):
    class _FakeMps:
        @staticmethod
        def is_available():
            return False

    class _FakeBackends:
        mps = _FakeMps

    class _FakeTorch:
        cuda = _FakeCuda(capability, arch_list)
        backends = _FakeBackends

    return _FakeTorch


def test_auto_falls_back_to_cpu_for_unsupported_gpu(monkeypatch) -> None:
    # A GTX 1050 (sm_61) with a torch build for sm_75+ must NOT pick CUDA.
    monkeypatch.delenv("EPISCOPE_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch((6, 1), ["sm_75", "sm_80", "sm_90"])
    )
    assert resolve_device() == "cpu"


def test_auto_uses_cuda_for_supported_gpu(monkeypatch) -> None:
    monkeypatch.delenv("EPISCOPE_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch((8, 6), ["sm_75", "sm_80", "sm_86"])
    )
    assert resolve_device() == "cuda"
