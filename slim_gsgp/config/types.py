from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch

if TYPE_CHECKING:
    from uuid import UUID

    from slim_gsgp.algorithms.GP.representations.population import Population
    from slim_gsgp.algorithms.GP.representations.tree import Tree


# type FunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class FunctionType(TypedDict):
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    arity: int


type FunctionDef = dict[str, FunctionType]
type XOFunc = Callable[[tuple, tuple, int, int], tuple]

type ConstantFunc = Callable[[], torch.Tensor]
type FitnessFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
type InitializerFunc = Callable[
    [
        int,
        int,
        FunctionDef,
        dict[str, int],
        dict[str, ConstantFunc],
        float,
    ],
    torch.Tensor,
]
type MutatorFunc = Callable[[tuple | str, int | None], tuple | str]
type FindEliteFunc = Callable[[Population, int], tuple[list[Tree], Tree]]
type DepthFunc = Callable[[tuple | str], int]
type TreeSelector = Callable[[Population], Tree]


class GPSolveParameters(TypedDict):
    log: int
    verbose: int
    test_elite: bool
    run_info: list[str | UUID] | None
    ffunction: FitnessFunc | str
    n_jobs: int
    max_depth: int | None
    n_elites: int
    elitism: bool
    n_iter: int
    log_path: NotRequired[str]
    depth_calculator: NotRequired[DepthFunc]


class GPParameters(TypedDict):
    initializer: InitializerFunc | str
    selector: TreeSelector
    crossover: XOFunc
    settings_dict: dict[str, float]
    p_xo: float
    p_m: NotRequired[float]
    pop_size: int
    seed: int
    mutator: NotRequired[MutatorFunc]
    find_elit_func: NotRequired[FindEliteFunc]


class GPPiInit(TypedDict):
    FUNCTIONS: FunctionDef
    CONSTANTS: dict[str, ConstantFunc]
    TERMINALS: NotRequired[dict[str, int]]
    p_c: float
    init_depth: int
    init_pop_size: int
