from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from probability.discrete.conditional import Conditional
from probability.discrete.discrete import Discrete
from probability.plotting import new_subplots


class MarkovChain(object):
    """
    A Discrete-Time Markov Chain (DTMC) with a finite state space.

    Governed by the Markov property:
        P(S_t = j | S_{t-1} = i, S_{t-2}, ..., S_0) = P(S_t = j | S_{t-1} = i) = T_{ij}

    Supports:
      - n-step transition matrices T^n
      - Stationary distributions pi * T = pi
      - Trajectory simulation / sampling
      - Absorbing chain analysis (fundamental matrix N = (I - Q)^-1, time to absorption, absorption probabilities)
      - Transition graph visualization (.draw)
      - Unrolling into a temporal Bayesian Network (.to_bayesian_network)

    Links:
      - https://en.wikipedia.org/wiki/Markov_chain
      - https://en.wikipedia.org/wiki/Absorbing_Markov_chain
    """

    def __init__(
            self,
            transitions: Union[DataFrame, np.ndarray, Dict[Tuple[str, str], float], Conditional],
            states: Optional[List[str]] = None,
            initial_distribution: Optional[Union[Discrete, Series, Dict[str, float], np.ndarray]] = None
    ):
        """
        Create a new Markov Chain.

        :param transitions: Transition probabilities. Rows are current state S_{t-1},
                            columns are next state S_t. Can be a DataFrame, 2D array,
                            dict of (s_from, s_to): prob, or a Conditional CPT.
        :param states: Optional list of state names if transitions is a numpy array.
        :param initial_distribution: Optional initial state distribution P(S_0).
        """
        if isinstance(transitions, Conditional):
            df = transitions.data
            self._states = list(df.index.unique())
            # Convert to square DataFrame
            t_df = df.unstack()
            self._transition_matrix = t_df.reindex(index=self._states, columns=self._states).fillna(0.0)
        elif isinstance(transitions, DataFrame):
            self._transition_matrix = transitions.copy().astype(float)
            self._states = list(self._transition_matrix.index)
        elif isinstance(transitions, dict):
            # Parse dict of {(from_state, to_state): prob}
            all_states = sorted(list({s for pair in transitions.keys() for s in pair}))
            self._states = states or all_states
            mat = np.zeros((len(self._states), len(self._states)), dtype=float)
            s_map = {s: i for i, s in enumerate(self._states)}
            for (u, v), p in transitions.items():
                mat[s_map[u], s_map[v]] = float(p)
            self._transition_matrix = DataFrame(mat, index=self._states, columns=self._states)
        elif isinstance(transitions, (np.ndarray, list)):
            mat = np.asarray(transitions, dtype=float)
            if states is None:
                states = [f'S{i}' for i in range(mat.shape[0])]
            self._states = list(states)
            self._transition_matrix = DataFrame(mat, index=self._states, columns=self._states)
        else:
            raise TypeError(f"Unsupported transitions type: {type(transitions)}")

        # Validate stochasticity (rows sum to 1)
        row_sums = self._transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            # Normalize rows if minor floating point discrepancy
            self._transition_matrix = self._transition_matrix.div(row_sums, axis=0)

        # Initial distribution
        if initial_distribution is None:
            # Default to uniform
            n_states = len(self._states)
            self._initial_distribution = Series(1.0 / n_states, index=self._states)
        elif isinstance(initial_distribution, Discrete):
            self._initial_distribution = initial_distribution.data.copy()
        elif isinstance(initial_distribution, (Series, dict)):
            self._initial_distribution = Series(initial_distribution)[self._states].astype(float)
            self._initial_distribution /= self._initial_distribution.sum()
        elif isinstance(initial_distribution, (np.ndarray, list)):
            arr = np.asarray(initial_distribution, dtype=float)
            arr = arr / arr.sum()
            self._initial_distribution = Series(arr, index=self._states)

    @property
    def states(self) -> List[str]:
        """List of state names."""
        return list(self._states)

    @property
    def transition_matrix(self) -> DataFrame:
        """The transition matrix T where T_ij = P(S_t = j | S_{t-1} = i)."""
        return self._transition_matrix

    @property
    def initial_distribution(self) -> Discrete:
        """The initial distribution P(S_0)."""
        return Discrete.from_probs(
            data=dict(self._initial_distribution),
            variables='state_0'
        )

    def transition(self, from_state: str, to_state: str) -> float:
        """Return transition probability P(S_t = to_state | S_{t-1} = from_state)."""
        return float(self._transition_matrix.loc[from_state, to_state])

    def n_step_transition_matrix(self, n: int) -> DataFrame:
        """
        Compute n-step transition probability matrix T^n.

        :param n: Number of time steps (n >= 0).
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0:
            return DataFrame(np.eye(len(self._states)), index=self._states, columns=self._states)
        vals = np.linalg.matrix_power(self._transition_matrix.values, n)
        return DataFrame(vals, index=self._states, columns=self._states)

    def state_distribution(self, n_steps: int) -> Discrete:
        """
        Compute the state distribution P(S_n) after `n_steps` given initial distribution P(S_0).
        """
        tn = self.n_step_transition_matrix(n_steps)
        dist = self._initial_distribution.values @ tn.values
        return Discrete.from_probs(
            data=dict(zip(self._states, dist)),
            variables=f'state_{n_steps}'
        )

    def stationary_distribution(self) -> Discrete:
        """
        Compute the stationary distribution pi satisfying pi * T = pi and sum(pi) = 1.
        """
        T = self._transition_matrix.values
        n = len(self._states)
        M = T.T - np.eye(n)
        M[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0

        try:
            pi = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            pi, _, _, _ = np.linalg.lstsq(M, b, rcond=None)

        pi = np.maximum(pi, 0.0)
        pi = pi / pi.sum()

        return Discrete.from_probs(
            data=dict(zip(self._states, pi)),
            variables='state'
        )

    def is_absorbing(self, state: str) -> bool:
        """Return True if `state` is an absorbing state (P(state | state) == 1.0)."""
        return np.isclose(self.transition(state, state), 1.0)

    @property
    def absorbing_states(self) -> List[str]:
        """List of all absorbing states."""
        return [s for s in self._states if self.is_absorbing(s)]

    @property
    def transient_states(self) -> List[str]:
        """List of all transient states."""
        return [s for s in self._states if not self.is_absorbing(s)]

    def is_absorbing_chain(self) -> bool:
        """Return True if chain has at least one absorbing state accessible from transient states."""
        return len(self.absorbing_states) > 0

    def fundamental_matrix(self) -> DataFrame:
        """
        For an absorbing Markov chain, compute the fundamental matrix N = (I - Q)^-1.
        N_ij is the expected number of times state j is visited starting from transient state i.
        """
        trans = self.transient_states
        if not trans:
            raise ValueError("No transient states in chain.")
        Q = self._transition_matrix.loc[trans, trans].values
        I = np.eye(len(trans))
        N = np.linalg.inv(I - Q)
        return DataFrame(N, index=trans, columns=trans)

    def expected_time_to_absorption(
            self,
            from_state: Optional[str] = None
    ) -> Union[float, Series]:
        """
        Compute expected number of steps until absorption from transient state(s).
        t = N * 1
        """
        N = self.fundamental_matrix()
        times = N.sum(axis=1)
        if from_state is not None:
            if from_state in self.absorbing_states:
                return 0.0
            return float(times.loc[from_state])
        return times

    def absorption_probabilities(self) -> DataFrame:
        """
        Compute absorption probabilities B = N * R.
        B_ik is the probability of absorbing into absorbing state k starting from transient state i.
        """
        trans = self.transient_states
        absorb = self.absorbing_states
        if not absorb or not trans:
            raise ValueError("Must have both transient and absorbing states.")
        N = self.fundamental_matrix().values
        R = self._transition_matrix.loc[trans, absorb].values
        B = N @ R
        return DataFrame(B, index=trans, columns=absorb)

    def simulate(
            self,
            n_steps: int,
            initial_state: Optional[str] = None,
            random_state: Optional[int] = None
    ) -> List[str]:
        """
        Simulate a sequence of state transitions.

        :param n_steps: Number of steps to simulate.
        :param initial_state: Optional starting state. If None, samples from initial distribution.
        :param random_state: Optional random seed for reproducibility.
        """
        rng = np.random.default_rng(random_state)
        if initial_state is not None:
            curr = initial_state
        else:
            curr = rng.choice(self._states, p=self._initial_distribution.values)

        trajectory = [curr]
        for _ in range(n_steps):
            probs = self._transition_matrix.loc[curr].values
            curr = rng.choice(self._states, p=probs)
            trajectory.append(curr)

        return trajectory

    def to_bayesian_network(self, n_steps: int) -> 'BayesianNetwork':
        """
        Unroll the Markov chain into a temporal Bayesian Network of length `n_steps`.
        Variables: S_0, S_1, ..., S_{n_steps}.
        """
        from probability.discrete.bayesian_network import BayesianNetwork

        s0 = Discrete.from_probs(
            data=dict(self._initial_distribution),
            variables='S_0'
        )
        nodes = [s0]

        for t in range(1, n_steps + 1):
            prev_var = f'S_{t-1}'
            curr_var = f'S_{t}'
            # Build CPT dictionary
            cpt_dict = {}
            for u in self._states:
                for v in self._states:
                    cpt_dict[(v, u)] = self.transition(u, v)

            cond = Conditional.from_probs(
                data=cpt_dict,
                joint_variables=curr_var,
                conditional_variables=prev_var
            )
            nodes.append(cond)

        return BayesianNetwork(nodes)

    def draw(
            self,
            ax: Optional[plt.Axes] = None,
            layout: str = 'circular',
            node_color: str = '#A0CBE2',
            node_size: int = 2500,
            font_size: int = 11,
            edge_color: str = '#4A4A4A',
            **kwargs
    ) -> plt.Axes:
        """
        Draw the state transition diagram for the Markov chain.
        """
        if ax is None:
            fig, ax = new_subplots(1, 1)

        G = nx.MultiDiGraph()
        for u in self._states:
            G.add_node(u)
        for u in self._states:
            for v in self._states:
                p = self.transition(u, v)
                if p > 0.001:
                    G.add_edge(u, v, weight=p, label=f"{p:.2f}")

        if layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        else:
            pos = nx.shell_layout(G)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=node_size)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight='bold')

        # Draw directed edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_color, arrowstyle='-|>',
            arrowsize=20, node_size=node_size,
            connectionstyle='arc3,rad=0.15'
        )

        ax.set_title("Markov Chain State Transition Diagram")
        ax.axis('off')
        return ax

    def __repr__(self) -> str:
        return f"MarkovChain(states={self._states}, n_states={len(self._states)})"
