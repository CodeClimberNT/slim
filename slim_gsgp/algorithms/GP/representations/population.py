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
"""Population class implementation for evaluating genetic programming individuals."""

import torch
from joblib import Parallel, delayed

from slim_gsgp.algorithms.GP.representations.tree import Tree
from slim_gsgp.algorithms.GP.representations.tree_utils import _execute_tree
from slim_gsgp.config.types import FitnessFunc
from slim_gsgp.evaluators.fitness_functions import se


class Population:
    def __init__(self, pop: list[Tree]) -> None:
        """
        Initializes a population of Trees.

        This constructor sets up the population with a list of Tree objects,
        calculating the size of the population and the total node count.

        Parameters
        ----------
        pop : List
            The list of individual Tree objects that make up the population.

        """
        self.population: list[Tree] = pop
        self.size: int = len(pop)
        self.nodes_count: int = sum(ind.node_count for ind in pop)
        self.fit: list[torch.Tensor] = []

    def evaluate(
        self,
        ffunction: FitnessFunc,
        X: torch.Tensor,
        y: torch.Tensor,
        n_jobs: int = 1,
        *,
        new_fitness: bool = True,
    ) -> None:
        """
        Evaluates the population given a certain fitness function, input data (X), and target data (y).

        Attributes a fitness tensor to the population.

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individuals.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        n_jobs : int
            The maximum number of concurrently running jobs for joblib parallelization.
        new_fitness : bool
            Create or not a fitness matrix based on the current population and dataset provided.

        Raises
        ------
        ValueError
            When the fitness matrix does not match with the `ffunction` results

        """
        # Evaluates individuals' semantics
        y_pred: list[torch.Tensor] = Parallel(n_jobs=n_jobs)(
            delayed(_execute_tree)(
                individual.repr_,
                X,
                individual.FUNCTIONS,
                individual.TERMINALS,
                individual.CONSTANTS,
            )
            for individual in self.population
        )

        # Evaluate fitnesses
        self.fit: list[torch.Tensor] = [
            ffunction(y, y_pred_ind) for y_pred_ind in y_pred
        ]

        if new_fitness:
            # Create matrix M of square errors
            # TODO: ASSUMPTION -> The list stays ordered!
            # Shape: (number of individuals, number of samples)
            # print(X.shape)
            # sys.exit()
            M = torch.empty((len(y_pred), X.shape[0]))
            for row in range(M.shape[0]):
                M[row] = se(y, y_pred[row])

            ind_fitnesses = M.mean(dim=1)
            fc_fitnesses = M.mean(dim=0)
            for row in range(M.shape[0]):
                # TODO: Assumption! consider MSE
                # as the only fitness function used here
                if not torch.allclose(ind_fitnesses[row], self.fit[row]):
                    raise ValueError(
                        f"Fitness Matrix has discrepancies with the fitness calculated: "
                        f"{ind_fitnesses[row]=}, {self.fit[row]=}"
                    )
            new_fitnesses = (M * fc_fitnesses.unsqueeze(0)).mean(dim=1)
            self.fit = list(new_fitnesses.unbind(dim=0))

        # Assign individuals' fitness
        for i, f in enumerate(self.fit):
            self.population[i].fitness = f
