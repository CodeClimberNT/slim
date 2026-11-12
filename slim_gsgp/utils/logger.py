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
from __future__ import annotations

import csv
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from uuid import UUID

    import torch

    from slim_gsgp.config.types import GPParameters, GPPiInit, GPSolveParameters


def log_settings(
    path: str,
    settings: tuple[GPSolveParameters, GPParameters, GPPiInit, dict[str, float]],
    unique_run_id: UUID,
) -> None:
    """
    Log the settings to a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    settings : list[dict]
        Dictionary of settings.
    unique_run_id : UUID
        Unique identifier for the run.

    """
    settings_dict = merge_settings(*settings)
    del settings_dict["TERMINALS"]

    infos = [unique_run_id, settings_dict]

    with Path(path).open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(infos)


def merge_settings(
    sd1: GPSolveParameters,
    sd2: GPParameters,
    sd3: GPPiInit,
    sd4: dict[str, float],
) -> dict[str, GPSolveParameters | GPParameters | GPPiInit | float]:
    """
    Merge multiple settings dictionaries into one.

    Parameters
    ----------
    sd1 : dict
        First settings dictionary.
    sd2 : dict
        Second settings dictionary.
    sd3 : dict
        Third settings dictionary.
    sd4 : dict
        Fourth settings dictionary.

    Returns
    -------
    dict
        Merged settings dictionary.

    """
    return {**sd1, **sd2, **sd3, **sd4}


def logger(
    path: str,
    generation: int,
    elite_fit: float | torch.Tensor,
    timing: float,
    nodes: float,
    additional_infos: list[float | str] | None = None,
    run_info: list[float | str] | None = None,
    seed: int = 0,
) -> None:
    """
    Logs information into a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    generation : int
        Current generation number.
    elite_fit : float
        Elite's validation fitness value.
    timing : float
        Time taken for the process.
    nodes : int
        Count of nodes in the population.
    additional_infos : list, optional
        Population's test fitness value(s) and diversity measurements. Defaults to None.
    run_info : list, optional
        Information about the run. Defaults to None.
    seed : int, optional
        The seed used in random, numpy, and torch libraries. Defaults to 0.

    """
    if not Path(Path(path).parent).is_dir():
        Path(Path(path).parent).mkdir()
    with Path(path).open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        infos = copy(run_info) if run_info is not None else []
        infos.extend([seed, generation, float(elite_fit), timing, nodes])

        if additional_infos is not None:
            try:
                additional_infos[0] = float(additional_infos[0])
            except TypeError:
                additional_infos[0] = "None"
            infos.extend(additional_infos)

        writer.writerow(infos)


def drop_experiment_from_logger(experiment_id: str | int, log_path: str) -> None:
    """
    Remove an experiment from the logger CSV file. If the given experiment_id is -1, the last saved experiment is removed.

    Parameters
    ----------
    experiment_id : str or int
        The experiment ID to be removed. If -1, the most recent experiment is removed.
    log_path : str
        Path to the file containing the logging information.

    """
    logger_data = pd.read_csv(log_path)

    # If we choose to remove the last stored experiment
    if experiment_id == -1:
        # Find the experiment id of the last row in the CSV file
        experiment_id = logger_data.iloc[-1, 1]

    # Exclude the logger data with the chosen id
    to_keep = logger_data[logger_data.iloc[:, 1] != experiment_id]
    # Save the new excluded dataset
    to_keep.to_csv(log_path, index=False, header=None)
