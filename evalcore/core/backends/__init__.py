"""Backends: the only part of the harness that knows how a model is called.
The contract is in base.py."""

from .base import (
    DECODE_KEYS,
    Backend,
    Fatal,
    Request,
    Response,
    Transient,
    batch_with_retries,
    decode_signature,
    filled,
    merge_params,
    refuse_dropped,
    refuse_unread,
    request_key,
    require,
    with_retries,
)
from .cache import Cache
from .check import CheckReport, check_backend
from .echo import EchoBackend
from .function import FunctionBackend
from .hf import HFBackend
from .http import ADAPTERS, HTTPBackend, VLLMBackend
from .probe import FunctionProbe, Probe

__all__ = [
    "ADAPTERS",
    "DECODE_KEYS",
    "Backend",
    "Cache",
    "CheckReport",
    "EchoBackend",
    "Fatal",
    "FunctionBackend",
    "FunctionProbe",
    "HFBackend",
    "HTTPBackend",
    "Probe",
    "Request",
    "Response",
    "Transient",
    "VLLMBackend",
    "batch_with_retries",
    "check_backend",
    "decode_signature",
    "filled",
    "merge_params",
    "refuse_dropped",
    "refuse_unread",
    "request_key",
    "require",
    "with_retries",
]
