"""Backends: the only part of the harness that knows how a model is called.

Contract.  A backend maps a list of `Request` to a list of `Response` of the
same length and order.  It may raise `Transient` (retry) or `Fatal` (do not).
It must NOT know about cells, items, clusters, or metrics -- everything above
this line is backend-agnostic, which is what lets one core serve both a
rate-limited HTTP API and a local batched forward pass.

Batching is expressed by `complete_batch`; the default maps `complete` over
the list.  An HTTP backend leaves that alone and gets its throughput from the
runner's thread pool.  A local HF backend overrides it and gets its
throughput from a real batched forward, with the runner at one worker.
"""

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
                   request_key,
                   require,
                   with_retries,
)
from .cache import Cache
from .check import CheckReport, check_backend
from .echo import EchoBackend
from .function import FunctionBackend
from .hf import HFBackend
from .http import ADAPTERS, HTTPBackend
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
                   "batch_with_retries",
                   "check_backend",
                   "decode_signature",
                   "filled",
                   "merge_params",
                   "refuse_dropped",
                   "request_key",
                   "require",
                   "with_retries",
]
