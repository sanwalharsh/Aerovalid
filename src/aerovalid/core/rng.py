"""Deterministic RNG management (FR-COR-006). A single master seed is spawned into
independent per-component generators via `numpy.random.SeedSequence.spawn`, never by
hand-mixing seeds. No module-level RNG state exists anywhere in the codebase.
"""

from __future__ import annotations

import numpy as np


def spawn_generators(master_seed: int, labels: list[str]) -> dict[str, np.random.Generator]:
    """
    Derive one independent `Generator` per label from a single master seed.

    Parameters
    ----------
    master_seed : int
        The study's master seed (`StudyConfig.study.seed`).
    labels : list[str]
        Names of the components needing an independent stream, e.g. flight IDs.

    Returns
    -------
    dict[str, numpy.random.Generator]
        One PCG64 generator per label. Same `master_seed` and `labels` (same order)
        always yield bit-identical streams (FR-COR-006).
    """
    seed_seq = np.random.SeedSequence(master_seed)
    children = seed_seq.spawn(len(labels))
    return {
        label: np.random.Generator(np.random.PCG64(child))
        for label, child in zip(labels, children, strict=True)
    }


def child_seed(master_seed: int, index: int) -> int:
    """
    Derive a single reproducible integer seed for component `index` of `master_seed`.

    Useful where an API wants a plain `int` seed rather than a `Generator` (e.g. a
    third-party library's `random_state=`). Not bit-identical to the `Generator`
    `spawn_generators` would build for the same index -- only the two functions'
    independence/determinism properties agree, not their output streams.

    Parameters
    ----------
    master_seed : int
        The study's master seed.
    index : int
        Zero-based index of the component (e.g. flight index).

    Returns
    -------
    int
        A 32-bit unsigned integer seed, deterministic in (`master_seed`, `index`) and
        independent across different `index` values.
    """
    seed_seq = np.random.SeedSequence(master_seed)
    (child,) = seed_seq.spawn(index + 1)[index : index + 1]
    return int(child.generate_state(1, dtype=np.uint32)[0])
