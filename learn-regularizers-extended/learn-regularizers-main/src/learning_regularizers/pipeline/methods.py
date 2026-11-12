"""The six compared training variants and the run grid they induce.

    1  Optimizee + B
    2  Optimizee + B + lambda R
    3  Optimizee + LO
    4  Optimizee + LO + lambda_smooth R_smooth
    5  Optimizee + LO + lambda_reg R
    6  Optimizee + LO + lambda_smooth R_smooth + lambda_reg R      (proposed)

B in {adam, sgd} applies to methods 1-2; R in {l1, l2, sam, gam, gsam} to methods 2, 5, 6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class MethodSpec:
    id: int
    key: str
    label: str
    uses_lo: bool
    uses_smooth: bool
    uses_reg: bool

    @property
    def uses_base_optimizer(self) -> bool:
        return not self.uses_lo


METHODS = {
    1: MethodSpec(1, "B", "Optimizee + B", False, False, False),
    2: MethodSpec(2, "B+R", "Optimizee + B + lambda R", False, False, True),
    3: MethodSpec(3, "LO", "Optimizee + LO", True, False, False),
    4: MethodSpec(4, "LO+Rsmooth", "Optimizee + LO + lambda_smooth R_smooth", True, True, False),
    5: MethodSpec(5, "LO+R", "Optimizee + LO + lambda_reg R", True, False, True),
    6: MethodSpec(6, "LO+Rsmooth+R", "Optimizee + LO + lambda_smooth R_smooth + lambda_reg R (proposed)",
                  True, True, True),
}


@dataclass(frozen=True)
class RunKey:
    """One trained entity: a baseline optimizer setting or one learned optimizer."""
    method: int
    base_optimizer: Optional[str]   # only for methods 1-2
    regularizer: Optional[str]      # only for methods 2, 5, 6

    @property
    def spec(self) -> MethodSpec:
        return METHODS[self.method]

    @property
    def run_id(self) -> str:
        return f"M{self.method}_{self.base_optimizer or 'LO'}_{self.regularizer or 'noreg'}"

    def as_dict(self) -> dict:
        return {"method": self.method, "base_optimizer": self.base_optimizer,
                "regularizer": self.regularizer, "run_id": self.run_id}


def run_grid(methods: List[int], base_optimizers: List[str], regularizers: List[str]) -> Iterator[RunKey]:
    for m in methods:
        spec = METHODS[m]
        bs = base_optimizers if spec.uses_base_optimizer else [None]
        rs = regularizers if spec.uses_reg else [None]
        for b in bs:
            for r in rs:
                yield RunKey(m, b, r)


def cell_run_key(method: int, base_optimizer: str, regularizer: str) -> RunKey:
    """RunKey that fills table cell (method, R) of the table for base optimizer B.

    Methods without R (1, 3, 4) have the same value in every R slice; methods without B
    (3-6) have the same value in both B tables.
    """
    spec = METHODS[method]
    return RunKey(method,
                  base_optimizer if spec.uses_base_optimizer else None,
                  regularizer if spec.uses_reg else None)
