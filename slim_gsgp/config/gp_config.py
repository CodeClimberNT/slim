# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Sets up the configuration dictionaries for the execution of the GP algorithm."""

from functools import partial

import torch

from slim_gsgp.algorithms.GP.operators.crossover_operators import crossover_trees
from slim_gsgp.config.types import (
    ConstantFunc,
    FitnessFunc,
    FunctionDef,
    GPParameters,
    GPPiInit,
    GPSolveParameters,
    InitializerFunc,
)
from slim_gsgp.evaluators.fitness_functions import (
    mae,
    mae_int,
    mse,
    rmse,
    signed_errors,
)
from slim_gsgp.initializers.initializers import full, grow, rhh
from slim_gsgp.selection.selection_algorithms import tournament_selection_min
from slim_gsgp.utils.utils import protected_div

# Define functions and constants
FUNCTIONS: dict[str, FunctionDef] = {
    "add": {"function": torch.add, "arity": 2},
    "subtract": {"function": torch.sub, "arity": 2},
    "multiply": {"function": torch.mul, "arity": 2},
    "divide": {"function": protected_div, "arity": 2},
}


CONSTANTS: dict[str, ConstantFunc] = {
    "constant_2": partial(torch.tensor, 2.0),
    "constant_3": partial(torch.tensor, 3.0),
    "constant_4": partial(torch.tensor, 4.0),
    "constant_5": partial(torch.tensor, 5.0),
    "constant__1": partial(torch.tensor, -1.0),
}

# Set parameters
settings_dict: dict[str, float] = {"p_test": 0.2}


# GP solve parameters
gp_solve_parameters: GPSolveParameters = {
    "log": 1,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": "rmse",
    "n_jobs": 1,
    "max_depth": 17,
    "n_elites": 1,
    "elitism": True,
    "n_iter": 1000,
}


gp_parameters: GPParameters = {
    "initializer": "rhh",
    "selector": tournament_selection_min(2),
    "crossover": crossover_trees(FUNCTIONS),
    "settings_dict": settings_dict,
    "p_xo": 0.8,
    "pop_size": 100,
    "seed": 74,
}


gp_pi_init: GPPiInit = {
    "FUNCTIONS": FUNCTIONS,
    "CONSTANTS": CONSTANTS,
    "p_c": 0,
    "init_depth": 6,
    "init_pop_size": 0,
}


fitness_function_options: dict[str, FitnessFunc] = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors,
}

initializer_options: dict[str, InitializerFunc] = {
    "rhh": rhh,
    "grow": grow,
    "full": full,
}
