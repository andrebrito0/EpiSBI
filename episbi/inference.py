from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence
import tempfile

import numpy as np
import pandas as pd
import pyabc
import torch
import torch.nn as nn

from .embedding import LSTMembedding


class SBIEngine:
    def __init__(self, density_estimator: str = "maf", device: str = "cpu", batch_size: int = 256):
        self.de_type = density_estimator
        self.device = device
        self.batch_size = batch_size

    def _param_names(self, prior):
        return list(prior.names) if hasattr(prior, "names") else None

    def _discrete_param_names(self, prior):
        return list(prior.discrete_names) if hasattr(prior, "discrete_names") else []

    def _reorder_parameter_frame(self, values, param_names: Optional[Sequence[str]] = None):
        if param_names is None or not isinstance(values, pd.DataFrame):
            return values
        ordered = [name for name in param_names if name in values.columns]
        remaining = [name for name in values.columns if name not in ordered]
        return values[ordered + remaining]

    def _resample_parameter_particles(self, values, weights, num_samples: int, random_state: int = 0):
        if not isinstance(values, pd.DataFrame):
            return values
        weights = np.asarray(weights, dtype=float)
        weights = None if weights.sum() <= 0 else weights / weights.sum()
        return values.sample(n=num_samples, replace=True, weights=weights, random_state=random_state).reset_index(drop=True)

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _named_dict_to_array(self, values: Dict):
        columns = []
        for value in values.values():
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr[None]
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            if arr.ndim != 1:
                raise ValueError(f"Named observed values must be one-dimensional, got {arr.shape}.")
            columns.append(arr)
        return np.stack(columns, axis=1)

    def _observed_array(self, values):
        if not isinstance(values, dict):
            return values
        if "observed" in values:
            return values["observed"]
        if "data" in values:
            return values["data"]
        if "transitions" in values:
            return values["transitions"]
        if "compartments" in values:
            return values["compartments"]
        return self._named_dict_to_array(values)

    def _obs_to_tensor(self, obs_data):
        return self._to_tensor(self._observed_array(obs_data))

    def _obs_to_pyabc_dict(self, obs_data):
        if isinstance(obs_data, dict):
            if "observed" in obs_data:
                return {"observed": np.asarray(obs_data["observed"], dtype=np.float32)}
            if "data" in obs_data:
                return {"observed": np.asarray(obs_data["data"], dtype=np.float32)}
            return {key: np.asarray(value, dtype=np.float32) for key, value in obs_data.items()}
        return {"observed": np.asarray(obs_data, dtype=np.float32)}

    def _sim_to_array(self, sim_output):
        return np.asarray(self._observed_array(sim_output), dtype=np.float32)

    def _match_shape(self, simulated, observed):
        simulated = np.asarray(simulated, dtype=np.float32)
        observed = np.asarray(observed, dtype=np.float32)
        if simulated.shape == observed.shape:
            return simulated
        if simulated.ndim == 2 and simulated.shape[-1] == 1 and observed.ndim == 1:
            simulated = simulated[:, 0]
        elif simulated.ndim == 1 and observed.ndim == 2 and observed.shape[-1] == 1:
            simulated = simulated[:, None]
        if simulated.shape == observed.shape:
            return simulated
        if simulated.size == observed.size:
            return simulated.reshape(observed.shape)
        raise ValueError(
            "Simulator output shape does not match obs_data. "
            f"Got simulator shape {simulated.shape} and obs_data shape {observed.shape}."
        )

    def _sim_to_pyabc_dict(self, sim_output, obs_dict: Dict):
        if isinstance(sim_output, dict) and set(obs_dict).issubset(sim_output):
            return {key: self._match_shape(sim_output[key], obs_dict[key]) for key in obs_dict}

        sim_value = self._observed_array(sim_output)
        if len(obs_dict) == 1:
            obs_key = list(obs_dict.keys())[0]
            return {obs_key: self._match_shape(sim_value, obs_dict[obs_key])}

        if isinstance(sim_output, dict):
            return {key: self._match_shape(sim_output[key], obs_dict[key]) for key in obs_dict}

        raise ValueError(f"Simulator output keys do not match obs_data keys: {sorted(obs_dict)}.")

    def _make_pyabc_simulator(self, simulator_func: Callable, obs_dict: Dict, simulator_kwargs: Optional[Dict] = None):
        simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs

        def wrapped_simulator(theta):
            sim_output = simulator_func(theta, **simulator_kwargs)
            return self._sim_to_pyabc_dict(sim_output, obs_dict)

        return wrapped_simulator

    def _get_neural_net(self, use_lstm: bool = False, input_dim: int = 1, embedding_net: Optional[nn.Module] = None):
        from sbi.neural_nets import posterior_nn

        if embedding_net is None:
            embedding_net = LSTMembedding(input_dim=input_dim).to(self.device) if use_lstm else nn.Identity()
        return posterior_nn(model=self.de_type, embedding_net=embedding_net)

    def _prepare_npe_xs(self, xs: torch.Tensor, thetas: torch.Tensor, input_dim: int, use_sequence_embedding: bool):
        if xs.ndim == 1:
            raise ValueError(f"xs must include a simulation axis and a data axis, got {tuple(xs.shape)}.")

        if xs.ndim == 2:
            if xs.shape[0] == thetas.shape[0]:
                return xs[:, :, None] if use_sequence_embedding and input_dim == 1 else xs
            if xs.shape[1] == thetas.shape[0]:
                xs = xs.T
                return xs[:, :, None] if use_sequence_embedding and input_dim == 1 else xs
        elif xs.ndim == 3:
            if xs.shape[0] != thetas.shape[0] and xs.shape[1] == thetas.shape[0]:
                xs = xs.transpose(0, 1)
            if xs.shape[0] != thetas.shape[0]:
                raise ValueError(f"thetas.shape={tuple(thetas.shape)} and xs.shape={tuple(xs.shape)} are not aligned.")
            if use_sequence_embedding:
                if xs.shape[-1] != input_dim:
                    raise ValueError(f"input_dim={input_dim} but xs has {xs.shape[-1]} output channel(s).")
                return xs
            if xs.shape[-1] == 1:
                return xs.squeeze(-1)
            raise ValueError("Use run_npe_lstm() or provide embedding_net for multi-output time-series xs.")

        raise ValueError(f"Could not align xs with thetas. Got thetas.shape={tuple(thetas.shape)} and xs.shape={tuple(xs.shape)}.")

    def _prepare_nre_xs(self, xs: torch.Tensor, thetas: torch.Tensor, x_obs: torch.Tensor):
        if xs.ndim == 1:
            raise ValueError(f"xs must include a simulation axis and a data axis, got {tuple(xs.shape)}.")
        if xs.ndim == 2:
            if xs.shape[0] != thetas.shape[0] and xs.shape[1] == thetas.shape[0]:
                xs = xs.T
        elif xs.ndim == 3:
            if xs.shape[0] != thetas.shape[0] and xs.shape[1] == thetas.shape[0]:
                xs = xs.transpose(0, 1)
            xs = xs.reshape(xs.shape[0], -1)
        else:
            raise ValueError(f"xs must have 2 or 3 dimensions, got {tuple(xs.shape)}.")

        if x_obs.ndim > 1:
            x_obs = x_obs.reshape(-1)
        if x_obs.numel() == int(np.prod(xs.shape[1:])):
            x_obs = x_obs.reshape(xs.shape[1:])
        if tuple(x_obs.shape) != tuple(xs.shape[1:]):
            raise ValueError(f"obs_data shape {tuple(x_obs.shape)} does not match one x shape {tuple(xs.shape[1:])}.")
        return xs, x_obs

    def _validate_sbi_inputs(self, thetas: torch.Tensor, xs: torch.Tensor, x_obs: torch.Tensor):
        if thetas.ndim != 2:
            raise ValueError(f"thetas must have shape (n_simulations, n_parameters), got {tuple(thetas.shape)}.")
        if xs.ndim < 2:
            raise ValueError(f"xs must have shape (n_simulations, ...), got {tuple(xs.shape)}.")
        if thetas.shape[0] != xs.shape[0]:
            raise ValueError(f"thetas and xs have different simulation counts: {tuple(thetas.shape)} vs {tuple(xs.shape)}.")
        if tuple(x_obs.shape) != tuple(xs.shape[1:]):
            raise ValueError(
                "obs_data must have the same shape as one simulated x. "
                f"Got obs_data shape {tuple(x_obs.shape)} but one x has shape {tuple(xs.shape[1:])}."
            )

    def run_abc(
        self,
        obs_data,
        prior,
        simulator_func: Callable,
        num_simulations: int = 1000,
        population_size: int = 100,
        num_samples: int = 10000,
        eps_alpha: float = 0.2,
        transition_scaling: float = 0.5,
        simulator_kwargs: Optional[Dict] = None,
        resample_seed: int = 0,
    ):
        print("[*] Running SMC-ABC...")

        param_names = self._param_names(prior)
        discrete_param_names = self._discrete_param_names(prior)
        prior = prior.pyabc if hasattr(prior, "pyabc") else prior
        obs_dict = self._obs_to_pyabc_dict(obs_data)
        abc_simulator = self._make_pyabc_simulator(simulator_func, obs_dict, simulator_kwargs)

        distance = pyabc.AdaptivePNormDistance(p=2, scale_function=pyabc.distance.std, all_particles_for_scale=True)
        eps = pyabc.QuantileEpsilon(initial_epsilon="from_sample", alpha=eps_alpha)
        transition = pyabc.MultivariateNormalTransition(scaling=transition_scaling)
        abc = pyabc.ABCSMC(abc_simulator, prior, distance, eps=eps, transitions=transition, population_size=population_size)
        db_path = "sqlite:///" + tempfile.mkstemp(suffix=".db")[1]
        abc.new(db_path, obs_dict)

        history = abc.run(max_total_nr_simulations=num_simulations)
        df, weights = history.get_distribution()
        df = self._reorder_parameter_frame(df, param_names)

        if discrete_param_names:
            samples = self._resample_parameter_particles(df, weights, num_samples, random_state=resample_seed)
            for name in discrete_param_names:
                if name in samples:
                    samples[name] = samples[name].round().astype(int)
        else:
            kde = pyabc.transition.MultivariateNormalTransition()
            kde.fit(df, weights)
            samples = self._reorder_parameter_frame(kde.rvs(num_samples), param_names)

        return samples, history

    def run_npe(
        self,
        obs_data,
        prior=None,
        thetas=None,
        xs=None,
        use_lstm: bool = False,
        input_dim: int = 1,
        learning_rate: float = 1e-3,
        num_samples: int = 10000,
        batch_size: Optional[int] = None,
        embedding_net: Optional[nn.Module] = None,
        show_train_summary: bool = True,
    ):
        from sbi.inference import NPE

        batch_size = self.batch_size if batch_size is None else batch_size
        prior = prior.sbi if hasattr(prior, "sbi") else prior
        print(f"[*] Running NPE (use_lstm={use_lstm}) with batch size {batch_size}...")

        thetas = self._to_tensor(thetas)
        xs = self._to_tensor(xs)
        x_obs = self._obs_to_tensor(obs_data)
        xs = self._prepare_npe_xs(xs, thetas, input_dim, use_lstm or embedding_net is not None)
        if xs.ndim == 3 and x_obs.ndim == 1 and xs.shape[-1] == 1 and x_obs.shape[0] == xs.shape[1]:
            x_obs = x_obs[:, None]
        elif xs.ndim == 2 and x_obs.ndim == 2 and x_obs.shape[-1] == 1 and x_obs.shape[0] == xs.shape[1]:
            x_obs = x_obs[:, 0]
        self._validate_sbi_inputs(thetas, xs, x_obs)

        neural_net = self._get_neural_net(use_lstm=use_lstm, input_dim=input_dim, embedding_net=embedding_net)
        inference = NPE(prior=prior, density_estimator=neural_net, device=self.device)
        density_estimator = inference.append_simulations(thetas, xs).train(
            training_batch_size=batch_size,
            learning_rate=learning_rate,
            show_train_summary=show_train_summary,
        )
        posterior = inference.build_posterior(density_estimator)
        samples = posterior.sample((num_samples,), x=x_obs)
        return {"posterior": posterior, "samples": samples, "density_estimator": density_estimator, "x_obs": x_obs}

    def run_npe_lstm(
        self,
        obs_data,
        prior=None,
        thetas=None,
        xs=None,
        input_dim: int = 1,
        learning_rate: float = 1e-3,
        num_samples: int = 10000,
        batch_size: Optional[int] = None,
        show_train_summary: bool = True,
    ):
        return self.run_npe(
            obs_data=obs_data,
            prior=prior,
            thetas=thetas,
            xs=xs,
            use_lstm=True,
            input_dim=input_dim,
            learning_rate=learning_rate,
            num_samples=num_samples,
            batch_size=batch_size,
            show_train_summary=show_train_summary,
        )

    def run_nre(
        self,
        obs_data,
        prior=None,
        thetas=None,
        xs=None,
        learning_rate: float = 1e-3,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        classifier=None,
        show_train_summary: bool = True,
    ):
        from sbi.inference import NRE

        batch_size = self.batch_size if batch_size is None else batch_size
        prior = prior.sbi if hasattr(prior, "sbi") else prior
        print(f"[*] Running NRE with batch size {batch_size}...")

        thetas = self._to_tensor(thetas)
        xs = self._to_tensor(xs)
        x_obs = self._obs_to_tensor(obs_data)
        xs, x_obs = self._prepare_nre_xs(xs, thetas, x_obs)
        self._validate_sbi_inputs(thetas, xs, x_obs)

        inference_kwargs = {"prior": prior, "device": self.device}
        if classifier is not None:
            inference_kwargs["classifier"] = classifier
        inference = NRE(**inference_kwargs)
        density_estimator = inference.append_simulations(thetas, xs).train(
            training_batch_size=batch_size,
            learning_rate=learning_rate,
            show_train_summary=show_train_summary,
        )
        posterior = inference.build_posterior(density_estimator)
        result = {"posterior": posterior, "density_estimator": density_estimator, "x_obs": x_obs}
        if num_samples is not None and num_samples > 0:
            result["samples"] = posterior.sample((num_samples,), x=x_obs)
        return result

    def run_pnpe(
        self,
        obs_data,
        prior,
        simulator_func: Callable,
        num_simulations: int = 10000,
        num_samples: int = 10000,
        batch_size: Optional[int] = None,
        population_size: int = 100,
        abc_fraction: float = 0.5,
        use_lstm: bool = False,
        input_dim: int = 1,
        learning_rate: float = 1e-3,
        show_train_summary: bool = True,
        simulator_kwargs: Optional[Dict] = None,
    ):
        batch_size = self.batch_size if batch_size is None else batch_size
        abc_sims = int(num_simulations * abc_fraction)
        print("[*] PNPE Stage 1: ABC preconditioning...")

        refined_thetas, history = self.run_abc(
            obs_data=obs_data,
            prior=prior,
            simulator_func=simulator_func,
            num_simulations=abc_sims,
            population_size=population_size,
            num_samples=num_simulations - abc_sims,
            simulator_kwargs=simulator_kwargs,
        )

        print("[*] PNPE Stage 2: Training NPE with preconditioned samples...")
        theta_values = refined_thetas.values if isinstance(refined_thetas, pd.DataFrame) else np.asarray(refined_thetas)
        thetas = torch.as_tensor(theta_values, dtype=torch.float32, device=self.device)
        kwargs = {} if simulator_kwargs is None else simulator_kwargs
        xs = [torch.as_tensor(self._sim_to_array(simulator_func(theta, **kwargs)), dtype=torch.float32) for theta in theta_values]
        xs = torch.stack(xs).to(self.device)

        result = self.run_npe(
            obs_data=obs_data,
            prior=prior,
            thetas=thetas,
            xs=xs,
            use_lstm=use_lstm,
            input_dim=input_dim,
            learning_rate=learning_rate,
            num_samples=num_samples,
            batch_size=batch_size,
            show_train_summary=show_train_summary,
        )
        result["abc_history"] = history
        return result

    def run_bsl(
        self,
        obs_data,
        prior,
        simulator_func: Callable,
        n_sim_raw: int = 50,
        min_keep: int = 10,
        filter_func: Optional[Callable] = None,
        n_mcmc: int = 1000,
        burn: int = 200,
        prop_scale: Optional[np.ndarray] = None,
        init_theta: Optional[np.ndarray] = None,
        seed: int = 0,
        param_names: Optional[Sequence[str]] = None,
        simulator_kwargs: Optional[Dict] = None,
        pass_seed: bool = True,
        n_jobs: int = 1,
        backend: str = "loky",
        output_key: str = "observed",
    ):
        # Bayesian synthetic likelihood (Wood 2010): at each proposed theta, simulate
        # a batch, model the summaries jointly as multivariate normal, and use that
        # as an approximate likelihood inside Metropolis-Hastings. Unlike run_abc,
        # this does not degrade as more summaries are added, since it models them
        # jointly rather than distance-matching them one at a time.
        #
        # filter_func(row) -> bool lets the caller condition on a subset of simulated
        # outcomes before estimating the covariance (e.g. "outbreak took off"), which
        # is often necessary since raw summaries from take-off/fizzle dynamics are
        # rarely close to normal; conditioning on a realised outcome repairs this and
        # is principled rather than a post-hoc fix, since only realised outcomes are
        # ever observed.
        from scipy.linalg import cholesky, solve_triangular, LinAlgError
        from .simulation import _theta_to_input, _sim_to_array

        if n_mcmc <= burn:
            raise ValueError("n_mcmc must be greater than burn.")

        simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
        if param_names is None:
            param_names = self._param_names(prior)

        obs_arr = _sim_to_array(obs_data, output_key=output_key).reshape(-1).astype(np.float64)
        low = np.asarray(prior.low, dtype=np.float64)
        high = np.asarray(prior.high, dtype=np.float64)
        prop_scale = (
            np.asarray(prop_scale, dtype=np.float64) if prop_scale is not None else 0.05 * (high - low)
        )

        rng = np.random.default_rng(seed)

        def _simulate_batch(theta, batch_seed):
            theta_input = _theta_to_input(theta, param_names)
            batch_rng = np.random.default_rng(batch_seed)
            sim_seeds = batch_rng.integers(0, 2**32 - 1, size=n_sim_raw, dtype=np.uint32)

            def _one(sim_seed):
                kwargs = dict(simulator_kwargs)
                if pass_seed and "seed" not in kwargs:
                    kwargs["seed"] = int(sim_seed)
                arr = _sim_to_array(simulator_func(theta_input, **kwargs), output_key=output_key)
                return arr.reshape(-1)

            if n_jobs == 1:
                rows = [_one(s) for s in sim_seeds]
            else:
                try:
                    from joblib import Parallel, delayed
                except ImportError as exc:
                    raise ImportError("run_bsl(n_jobs != 1) requires joblib.") from exc
                rows = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_one)(s) for s in sim_seeds)
            return np.stack(rows, axis=0).astype(np.float64)

        def _log_synth_lik(theta, batch_seed, scale):
            S = _simulate_batch(theta, batch_seed)
            if filter_func is not None:
                keep = np.array([filter_func(row) for row in S], dtype=bool)
                S = S[keep]
            if len(S) < min_keep:
                return -1e10
            Sk = S / scale
            mu = Sk.mean(axis=0)
            Sig = np.cov(Sk, rowvar=False) + np.eye(Sk.shape[1]) * 1e-6
            try:
                L = cholesky(Sig, lower=True)
            except LinAlgError:
                return -1e10
            z = solve_triangular(L, (obs_arr / scale) - mu, lower=True)
            k = Sk.shape[1]
            return -0.5 * k * np.log(2 * np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(z**2)

        if init_theta is None:
            init_theta = (low + high) / 2
        else:
            init_theta = np.asarray(
                prior.to_array(init_theta) if hasattr(prior, "to_array") else init_theta, dtype=np.float64
            )
        init_theta = np.clip(init_theta, low, high)

        print(f"[*] Running BSL-MCMC: {n_mcmc} steps x {n_sim_raw} sims/step...")
        pilot = _simulate_batch(init_theta, seed)
        scale = pilot.std(axis=0)
        scale[scale == 0] = 1.0

        cur = init_theta
        cur_ll = _log_synth_lik(cur, seed + 1, scale)
        chain = np.zeros((n_mcmc, len(param_names)))
        acc = 0
        for it in range(n_mcmc):
            prop = rng.normal(cur, prop_scale)
            if np.all(prop >= low) and np.all(prop <= high):
                prop_ll = _log_synth_lik(prop, seed + 2 + it, scale)
                if np.log(rng.random()) < prop_ll - cur_ll:
                    cur, cur_ll = prop, prop_ll
                    acc += 1
            chain[it] = cur

        samples = pd.DataFrame(chain[burn:], columns=param_names)
        print(f"      acceptance rate: {100 * acc / n_mcmc:.1f}%")
        return {"samples": samples, "chain": chain, "acceptance_rate": acc / n_mcmc, "scale": scale}

    def sample_posterior(self, posterior, obs_data, num_samples: int = 10000):
        return posterior.sample((num_samples,), x=self._obs_to_tensor(obs_data))
