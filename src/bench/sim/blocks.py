"""Block-hash prefix model, mirroring how real vLLM/EPP hash KV blocks.

A prompt is chunked into fixed-size token blocks and each block gets a hash that folds
in the hash of the block before it. Because the hash is a *rolling* function of the whole
prefix, two prompts that share a leading run of tokens produce identical leading block
hashes and diverge at exactly the first block where their tokens differ -- which is what
lets the router match on the longest common *prefix* (not a set intersection).
"""

from collections.abc import Container

# vLLM's default KV block is 16 tokens; keep the same default here.
DEFAULT_BLOCK_SIZE = 16

# Sentinel folded into the first block so block 0 depends on "no parent", matching the
# way vLLM seeds its prefix hash chain.
_ROOT = 1469598103934665603  # FNV-1a offset basis, an arbitrary fixed seed


def hash_blocks(tokens: list[int], block_size: int = DEFAULT_BLOCK_SIZE) -> list[int]:
    """Return the rolling per-block hash chain for a token sequence.

    Only whole blocks are hashed; a trailing partial block is ignored, matching vLLM,
    which can only cache complete blocks. ``h[i] = hash(h[i-1], tuple(block_i))``.
    """
    hashes: list[int] = []
    parent = _ROOT
    n_full = len(tokens) // block_size
    for i in range(n_full):
        block = tuple(tokens[i * block_size : (i + 1) * block_size])
        parent = hash((parent, block))
        hashes.append(parent)
    return hashes


def matched_prefix(req_blocks: list[int], cache: "Container[int]") -> int:
    """Length of the longest leading run of ``req_blocks`` resident in ``cache``.

    Prefix caching can only reuse a *contiguous* prefix: once a block is missing every
    later block must be (re)computed even if its hash happens to be cached, so we stop at
    the first miss rather than counting total membership.
    """
    matched = 0
    for h in req_blocks:
        if h in cache:
            matched += 1
        else:
            break
    return matched
