"""Simplified Multi-Objective Grey Wolf Optimizer (MO-GWO).

This implementation is designed for hyperparameter search over a discrete
search space. Each candidate ("wolf") is a dictionary of hyperparameters.

We assume a user-provided objective function:
    objectives = objective_fn(hparams)
which returns a tuple of floats, e.g.:
    (1 - val_auc, calibration_error, complexity_penalty)

MO-GWO then seeks a set of non-dominated solutions ('Pareto front').
"""
import random
import copy
from typing import Callable, Dict, List, Tuple


class MOGreyWolfOptimizer:
    def __init__(
        self,
        search_space: Dict[str, List],
        objective_fn: Callable[[Dict], Tuple[float, float, float]],
        num_wolves: int = 8,
        num_iters: int = 10,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.num_wolves = num_wolves
        self.num_iters = num_iters
        self.rng = random.Random(seed)
        self.population = []
        self.objectives = []

    def _random_candidate(self) -> Dict:
        return {k: self.rng.choice(v) for k, v in self.search_space.items()}

    def _init_population(self):
        self.population = [self._random_candidate() for _ in range(self.num_wolves)]
        self.objectives = [self.objective_fn(p) for p in self.population]

    @staticmethod
    def _dominates(a, b):
        """Return True if objective vector a dominates b (lower is better)."""
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def _pareto_front(self):
        front_indices = []
        for i, obj_i in enumerate(self.objectives):
            dominated = False
            for j, obj_j in enumerate(self.objectives):
                if i == j:
                    continue
                if self._dominates(obj_j, obj_i):
                    dominated = True
                    break
            if not dominated:
                front_indices.append(i)
        return front_indices

    def _encode(self, candidate: Dict):
        """Encode candidate dict as vector of indices in search space lists."""
        vec = []
        for k, values in self.search_space.items():
            vec.append(values.index(candidate[k]))
        return vec

    def _decode(self, vec):
        candidate = {}
        for (k, values), idx in zip(self.search_space.items(), vec):
            candidate[k] = values[max(0, min(len(values) - 1, idx))]
        return candidate

    def _clip_vec(self, vec):
        clipped = []
        for i, (k, values) in enumerate(self.search_space.items()):
            max_idx = len(values) - 1
            clipped.append(max(0, min(max_idx, vec[i])))
        return clipped

    def optimize(self):
        self._init_population()

        for it in range(self.num_iters):
            front_indices = self._pareto_front()
            leaders = [self.population[i] for i in front_indices]
            leader_vecs = [self._encode(c) for c in leaders]

            new_population = []
            new_objectives = []

            for i, candidate in enumerate(self.population):
                x_vec = self._encode(candidate)
                x_new = [float(x) for x in x_vec]

                for leader_vec in leader_vecs:
                    for d in range(len(x_new)):
                        r1 = self.rng.random()
                        r2 = self.rng.random()
                        A = 2 * r1 - 1
                        C = 2 * r2
                        D = abs(C * leader_vec[d] - x_new[d])
                        a = 2 - 2 * (it / max(1, self.num_iters - 1))
                        x_new[d] = leader_vec[d] - A * D * a

                x_new = self._clip_vec([int(round(v)) for v in x_new])
                new_candidate = self._decode(x_new)
                new_obj = self.objective_fn(new_candidate)

                new_population.append(new_candidate)
                new_objectives.append(new_obj)

            self.population = new_population
            self.objectives = new_objectives

        # Return Pareto front
        front_indices = self._pareto_front()
        pareto_solutions = [(self.population[i], self.objectives[i]) for i in front_indices]
        return pareto_solutions
