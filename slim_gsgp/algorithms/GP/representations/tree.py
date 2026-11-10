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
"""Tree class implementation for representing tree structures in genetic programming."""

import torch

from slim_gsgp.algorithms.GP.representations.tree_utils import (
    _execute_tree,
    flatten,
    tree_depth,
)
from slim_gsgp.config.types import (
    ConstantFunc,
    FitnessFunc,
    FunctionDef,
)


class Tree:
    """
    The Tree class representing the candidate solutions in genetic programming.

    Attributes
    ----------
    repr_ : tuple or str
        Representation of the tree structure.
    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.
    depth : int
        Depth of the tree.
    fitness : float
        Fitness value of the tree.
    test_fitness : float
        Test fitness value of the tree.
    node_count : int
        Number of nodes in the tree.

    """

    TERMINALS: dict[str, int]
    FUNCTIONS: dict[str, FunctionDef]
    CONSTANTS: dict[str, ConstantFunc]

    def __init__(self, repr_: tuple):
        """
        Initializes a Tree object.

        Parameters
        ----------
        repr_ : tuple
            Representation of the tree structure.

        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        self.repr_ = repr_
        self.depth = tree_depth(Tree.FUNCTIONS)(repr_)
        self.fitness = None
        self.test_fitness = None
        self.node_count = len(list(flatten(self.repr_)))

    def apply_tree(self, inputs: torch.Tensor):
        """
        Evaluates the tree on input vectors.

        Parameters
        ----------
        inputs : tuple
            Input vectors.

        Returns
        -------
        float
            Output of the evaluated tree.

        """
        return _execute_tree(
            repr_=self.repr_,
            X=inputs,
            FUNCTIONS=self.FUNCTIONS,
            TERMINALS=self.TERMINALS,
            CONSTANTS=self.CONSTANTS,
        )

    def evaluate(
        self,
        ffunction: FitnessFunc,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        testing=False,
        new_data=False,
    ) -> float | None:
        """
        Evaluates the tree given a fitness function, input data (X), and target data (y).

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individual.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        testing : bool, optional
            Flag indicating if the data is testing data. Default is False.
        new_data : bool, optional
            Flag indicating that the input data is new and the model is being used outside the training process.

        Returns
        -------
        None
            If the data is training or testing data, the fitness value is attributed to the individual.
        float
            If exposed to new data, the fitness value is returned.

        """
        # getting the predictions (i.e., semantics) of the individual
        preds = self.apply_tree(X)

        # if new (testing data) is being used, return the fitness value for this new data
        if new_data:
            return float(ffunction(y, preds))

        # if not, attribute the fitness value to the individual
        if testing:
            self.test_fitness = ffunction(y, preds)
        else:
            self.fitness = ffunction(y, preds)
        return None

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the tree semantics (output) for the given input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data to predict.

        Returns
        -------
        torch.Tensor
            The predicted output for the input data.

        Notes
        -----
        This function delegates the actual prediction task to the `apply_tree` method,
        which is assumed to be another method in the same class. The `apply_tree` method
        should be defined to handle the specifics of how predictions are made based on
        the tree structure used in this model.

        """
        return self.apply_tree(X)

    def get_tree_representation(self, indent=""):
        """
        Returns the tree representation as a string with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.

        Returns
        -------
        str
            Returns the tree representation with the chosen indentation.

        """
        representation = []

        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            representation.append(indent + f"{function_name}(\n")

            # if the function has an arity of 2, process both left and right subtrees
            if Tree.FUNCTIONS[function_name]["arity"] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                representation.extend((
                    Tree(left_subtree).get_tree_representation(indent + "  "),
                    Tree(right_subtree).get_tree_representation(indent + "  "),
                ))
            # if the function has an arity of 1, process the left subtree
            else:
                left_subtree = self.repr_[1]
                representation.append(
                    Tree(left_subtree).get_tree_representation(indent + "  "),
                )

            representation.append(indent + ")\n")
        else:  # If it's a terminal node
            representation.append(indent + f"{self.repr_}\n")

        return "".join(representation)

    def print_tree_representation(self, indent=""):
        """
        Prints the tree representation with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.

        """
        print(self.get_tree_representation(indent=indent))
