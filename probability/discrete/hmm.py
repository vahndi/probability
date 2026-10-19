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
import numpy as np
from pandas import DataFrame, Series

from probability.discrete.conditional import Conditional
from probability.discrete.discrete import Discrete
from probability.plotting import new_subplots


class HMM(object):
    """
    Hidden Markov Model (HMM) for discrete state spaces and discrete observations.

    Model:
      - Latent states: Z_t in {s_1, ..., s_K}
      - Observations: X_t in {o_1, ..., o_M}
      - Transition matrix: A_ij = P(Z_t = s_j | Z_{t-1} = s_i)
      - Emission matrix: B_jk = P(X_t = o_k | Z_t = s_j)
      - Initial distribution: pi_i = P(Z_0 = s_i)

    Core Inference Algorithms (BRML Ch 23 & PRML Ch 13):
      - Scaled Forward algorithm: exact sequence likelihood P(X_{1:T}) and filtering P(Z_t | x_{1:t})
      - Backward algorithm: exact smoothing P(Z_t | x_{1:T})
      - Viterbi algorithm: most probable hidden state sequence argmax P(z_{1:T} | x_{1:T})
      - Forward-backward simulation / sampling
      - State probability visualization (.plot_states)

    Links:
      - https://en.wikipedia.org/wiki/Hidden_Markov_model
      - https://en.wikipedia.org/wiki/Viterbi_algorithm
      - https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    """

    def __init__(
            self,
            states: List[str],
            emissions: List[Any],
            transitions: Union[DataFrame, np.ndarray, Dict[Tuple[str, str], float]],
            emission_matrix: Union[DataFrame, np.ndarray, Dict[Tuple[str, Any], float]],
            initial_distribution: Optional[Union[Discrete, Series, Dict[str, float], np.ndarray]] = None
    ):
        """
        Create a new Hidden Markov Model.

        :param states: Names of hidden states Z.
        :param emissions: Names / values of observable emissions X.
        :param transitions: Transition probabilities P(Z_t | Z_{t-1}).
        :param emission_matrix: Emission probabilities P(X_t | Z_t). Rows are hidden states, columns are emissions.
        :param initial_distribution: Initial hidden state distribution P(Z_0).
        """
        self._states = list(states)
        self._emissions = list(emissions)
        self._state_map = {s: i for i, s in enumerate(self._states)}
        self._emission_map = {e: i for i, e in enumerate(self._emissions)}
        K = len(self._states)
        M = len(self._emissions)

        # Parse transition matrix A
        if isinstance(transitions, DataFrame):
            self._A = transitions.loc[self._states, self._states].values.astype(float)
        elif isinstance(transitions, dict):
            mat = np.zeros((K, K), dtype=float)
            for (u, v), p in transitions.items():
                mat[self._state_map[u], self._state_map[v]] = float(p)
            self._A = mat
        else:
            self._A = np.asarray(transitions, dtype=float)

        # Normalize A rows
        row_sums = self._A.sum(axis=1, keepdims=True)
        self._A = np.divide(self._A, row_sums, out=np.zeros_like(self._A), where=row_sums != 0)

        # Parse emission matrix B
        if isinstance(emission_matrix, DataFrame):
            self._B = emission_matrix.loc[self._states, self._emissions].values.astype(float)
        elif isinstance(emission_matrix, dict):
            mat = np.zeros((K, M), dtype=float)
            for (s, o), p in emission_matrix.items():
                mat[self._state_map[s], self._emission_map[o]] = float(p)
            self._B = mat
        else:
            self._B = np.asarray(emission_matrix, dtype=float)

        # Normalize B rows
        row_sums = self._B.sum(axis=1, keepdims=True)
        self._B = np.divide(self._B, row_sums, out=np.zeros_like(self._B), where=row_sums != 0)

        # Initial distribution pi
        if initial_distribution is None:
            self._pi = np.full(K, 1.0 / K, dtype=float)
        elif isinstance(initial_distribution, Discrete):
            self._pi = np.array([initial_distribution.p(**{initial_distribution.variables[0]: s}) for s in self._states])
        elif isinstance(initial_distribution, (Series, dict)):
            self._pi = np.array([float(initial_distribution[s]) for s in self._states])
            self._pi /= self._pi.sum()
        else:
            self._pi = np.asarray(initial_distribution, dtype=float)
            self._pi /= self._pi.sum()

    @property
    def states(self) -> List[str]:
        """List of hidden states."""
        return list(self._states)

    @property
    def emissions(self) -> List[Any]:
        """List of emission symbols."""
        return list(self._emissions)

    @property
    def transition_matrix(self) -> DataFrame:
        """The transition matrix A."""
        return DataFrame(self._A, index=self._states, columns=self._states)

    @property
    def emission_matrix(self) -> DataFrame:
        """The emission matrix B."""
        return DataFrame(self._B, index=self._states, columns=self._emissions)

    @property
    def initial_distribution(self) -> Discrete:
        """The initial hidden state distribution."""
        return Discrete.from_probs(
            data=dict(zip(self._states, self._pi)),
            variables='hidden_state_0'
        )

    def _parse_observations(self, observations: Iterable[Any]) -> List[int]:
        """Convert observations list to index integers."""
        indices = []
        for obs in observations:
            if obs not in self._emission_map:
                raise KeyError(f"Observation '{obs}' not in emission vocabulary: {self._emissions}")
            indices.append(self._emission_map[obs])
        return indices

    def _forward(self, obs_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled forward algorithm.
        Returns (scaled_alpha, scale_factors).
        """
        T = len(obs_indices)
        K = len(self._states)
        alpha = np.zeros((T, K), dtype=float)
        c = np.zeros(T, dtype=float)

        # t = 0
        alpha[0] = self._pi * self._B[:, obs_indices[0]]
        c[0] = np.sum(alpha[0])
        if c[0] > 0:
            alpha[0] /= c[0]

        # t = 1 .. T-1
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self._A) * self._B[:, obs_indices[t]]
            c[t] = np.sum(alpha[t])
            if c[t] > 0:
                alpha[t] /= c[t]

        return alpha, c

    def _backward(self, obs_indices: List[int], c: np.ndarray) -> np.ndarray:
        """
        Scaled backward algorithm.
        Returns scaled_beta.
        """
        T = len(obs_indices)
        K = len(self._states)
        beta = np.zeros((T, K), dtype=float)

        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            denom = c[t + 1] if c[t + 1] > 0 else 1.0
            beta[t] = (self._A @ (self._B[:, obs_indices[t + 1]] * beta[t + 1])) / denom

        return beta

    def log_likelihood(self, observations: Iterable[Any]) -> float:
        """
        Compute exact log-likelihood ln P(X_{1:T}) of the observation sequence.
        """
        obs_idx = self._parse_observations(observations)
        if not obs_idx:
            return 0.0
        _, c = self._forward(obs_idx)
        # Avoid log(0)
        c_clean = np.where(c > 0, c, 1e-300)
        return float(np.sum(np.log(c_clean)))

    def likelihood(self, observations: Iterable[Any]) -> float:
        """
        Compute exact marginal probability P(X_{1:T}) of the observation sequence.
        """
        return float(np.exp(self.log_likelihood(observations)))

    def filter(self, observations: Iterable[Any]) -> DataFrame:
        """
        Compute filtered state distributions P(Z_t | x_{1:t}) for t = 1 .. T.

        :param observations: Sequence of observations.
        :return: DataFrame of size T x K with filtered probabilities.
        """
        obs_idx = self._parse_observations(observations)
        if not obs_idx:
            return DataFrame(columns=self._states)
        alpha, _ = self._forward(obs_idx)
        return DataFrame(alpha, index=range(1, len(obs_idx) + 1), columns=self._states)

    def smooth(self, observations: Iterable[Any]) -> DataFrame:
        """
        Compute smoothed posterior marginal state distributions P(Z_t | x_{1:T}) for t = 1 .. T.

        :param observations: Sequence of observations.
        :return: DataFrame of size T x K with smoothed probabilities.
        """
        obs_idx = self._parse_observations(observations)
        if not obs_idx:
            return DataFrame(columns=self._states)
        alpha, c = self._forward(obs_idx)
        beta = self._backward(obs_idx, c)
        gamma = alpha * beta
        # Normalize rows
        gamma_sums = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gamma_sums, out=np.zeros_like(gamma), where=gamma_sums != 0)
        return DataFrame(gamma, index=range(1, len(obs_idx) + 1), columns=self._states)

    def decode(self, observations: Iterable[Any]) -> Tuple[List[str], float]:
        """
        Find the most probable sequence of hidden states via the Viterbi algorithm.
        argmax_{z_{1:T}} P(z_{1:T} | x_{1:T})

        :param observations: Sequence of observations.
        :return: (most_probable_hidden_states, log_prob)
        """
        obs_idx = self._parse_observations(observations)
        T = len(obs_idx)
        K = len(self._states)

        if T == 0:
            return [], 0.0

        # Log probabilities with protection against log(0)
        log_pi = np.log(np.maximum(self._pi, 1e-300))
        log_A = np.log(np.maximum(self._A, 1e-300))
        log_B = np.log(np.maximum(self._B, 1e-300))

        V = np.zeros((T, K), dtype=float)
        ptr = np.zeros((T, K), dtype=int)

        # t = 0
        V[0] = log_pi + log_B[:, obs_idx[0]]

        # t = 1 .. T-1
        for t in range(1, T):
            for j in range(K):
                candidates = V[t - 1] + log_A[:, j]
                best_prev = int(np.argmax(candidates))
                ptr[t, j] = best_prev
                V[t, j] = candidates[best_prev] + log_B[j, obs_idx[t]]

        # Backtrack
        best_last = int(np.argmax(V[-1]))
        max_log_prob = float(V[-1, best_last])

        state_indices = [best_last]
        for t in range(T - 1, 0, -1):
            prev_state = ptr[t, state_indices[-1]]
            state_indices.append(prev_state)

        state_indices.reverse()
        decoded_states = [self._states[i] for i in state_indices]
        return decoded_states, max_log_prob

    def sample(
            self,
            n_steps: int,
            random_state: Optional[int] = None
    ) -> Tuple[List[str], List[Any]]:
        """
        Generate a synthetic trajectory of hidden states and corresponding emissions.

        :param n_steps: Number of time steps.
        :param random_state: Optional random seed.
        :return: (hidden_states_sequence, emissions_sequence)
        """
        rng = np.random.default_rng(random_state)
        hidden_seq: List[str] = []
        emiss_seq: List[Any] = []

        curr_state_idx = rng.choice(len(self._states), p=self._pi)
        curr_state = self._states[curr_state_idx]
        hidden_seq.append(curr_state)

        curr_emiss = rng.choice(self._emissions, p=self._B[curr_state_idx])
        emiss_seq.append(curr_emiss)

        for _ in range(1, n_steps):
            curr_state_idx = rng.choice(len(self._states), p=self._A[curr_state_idx])
            curr_state = self._states[curr_state_idx]
            hidden_seq.append(curr_state)

            curr_emiss = rng.choice(self._emissions, p=self._B[curr_state_idx])
            emiss_seq.append(curr_emiss)

        return hidden_seq, emiss_seq

    def plot_states(
            self,
            observations: Iterable[Any],
            smoothed: bool = True,
            true_states: Optional[List[str]] = None,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ) -> plt.Axes:
        """
        Plot state probabilities over time for the given observation sequence.

        :param observations: Sequence of observations.
        :param smoothed: If True, plot smoothed marginals P(Z_t | x_{1:T}).
                         If False, plot filtered marginals P(Z_t | x_{1:t}).
        :param true_states: Optional list of true hidden states to overlay as background blocks.
        :param ax: Matplotlib axes.
        """
        if ax is None:
            fig, ax = new_subplots(1, 1)

        probs_df = self.smooth(observations) if smoothed else self.filter(observations)
        title_prefix = "Smoothed P(Z_t | x_{1:T})" if smoothed else "Filtered P(Z_t | x_{1:t})"

        for state in self._states:
            ax.plot(probs_df.index, probs_df[state], label=f"State: {state}", **kwargs)

        if true_states is not None:
            # Highlight true states with subtle background shading
            for i, st in enumerate(true_states):
                s_idx = self._state_map[st]
                ax.axvspan(i + 0.5, i + 1.5, color=f"C{s_idx}", alpha=0.15)

        ax.set_xlabel("Time step t")
        ax.set_ylabel("Posterior Probability")
        ax.set_title(f"HMM {title_prefix}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
        return ax

    def to_bayesian_network(self, n_steps: int) -> 'BayesianNetwork':
        """
        Unroll the HMM into a temporal Bayesian Network of length `n_steps`.
        Nodes: Z_0, ..., Z_{n_steps} and X_0, ..., X_{n_steps}.
        """
        from probability.discrete.bayesian_network import BayesianNetwork

        z0 = Discrete.from_probs(
            data=dict(zip(self._states, self._pi)),
            variables='Z_0'
        )
        nodes = [z0]

        # Initial emission X_0 | Z_0
        x0_dict = {}
        for s in self._states:
            for o in self._emissions:
                x0_dict[(o, s)] = float(self._B[self._state_map[s], self._emission_map[o]])
        x0 = Conditional.from_probs(x0_dict, joint_variables='X_0', conditional_variables='Z_0')
        nodes.append(x0)

        for t in range(1, n_steps + 1):
            z_prev = f'Z_{t-1}'
            z_curr = f'Z_{t}'
            x_curr = f'X_{t}'

            # Transition Z_t | Z_{t-1}
            trans_dict = {}
            for u in self._states:
                for v in self._states:
                    trans_dict[(v, u)] = float(self._A[self._state_map[u], self._state_map[v]])
            z_node = Conditional.from_probs(trans_dict, joint_variables=z_curr, conditional_variables=z_prev)
            nodes.append(z_node)

            # Emission X_t | Z_t
            emiss_dict = {}
            for s in self._states:
                for o in self._emissions:
                    emiss_dict[(o, s)] = float(self._B[self._state_map[s], self._emission_map[o]])
            x_node = Conditional.from_probs(emiss_dict, joint_variables=x_curr, conditional_variables=z_curr)
            nodes.append(x_node)

        return BayesianNetwork(nodes)

    def __repr__(self) -> str:
        return f"HMM(states={self._states}, emissions={self._emissions})"
