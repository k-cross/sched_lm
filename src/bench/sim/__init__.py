"""Offline discrete-event simulator for prefix-cache routing.

This package models the piece the external ``llm-d-inference-sim`` cannot: explicit
per-node KV-cache state, queueing, and a KV-block interconnect, so that the observable
timing of a request *derives* from node state instead of being a constant knob. That
makes the three routing regimes -- wait for the hot node, transfer its KV blocks, or
recompute the prefix -- fall out of an ``argmin`` over candidate nodes rather than being
hand-set. See ``cost.py`` for the model and ``policies.py`` for the routers compared.
"""
