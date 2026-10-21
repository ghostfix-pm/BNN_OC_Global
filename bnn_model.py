"""
Heteroscedastic Bayesian neural network for chlorophyll-a retrieval.

Shared definitions used by ``train.py`` and ``predict_global.py``:
the network architecture, its hyperparameters, the input preprocessing,
and posterior predictive inference with the epistemic/aleatoric
uncertainty decomposition.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.nn import PyroModule, PyroSample

# The band definition lives in data_utils (NumPy-only) so that the data
# contract can be validated without importing PyTorch or Pyro; re-exported
# here for the callers that expect it in this module.
from data_utils import BANDS_NM, N_BANDS, RRS_FLOOR, RRS_VARIABLES  # noqa: F401

# ---------------------------------------------------------------------------
# Hyperparameters of the published model
# ---------------------------------------------------------------------------
PRIOR_SCALE = 0.3          # std of the zero-mean Gaussian weight priors (s)
LOGVAR_BIAS_PRIOR = -4.6   # prior mean of the log-variance output bias (sigma ~ 0.1)
HIDDEN = 32                # units in every hidden layer (h)
RANK = 20                  # rank of the low-rank variational posterior (r)
BETA_NLL = 0.5             # Beta-NLL weighting exponent (beta)
SIGMA_MIN = 1e-4           # lower bound on the predicted aleatoric std

# File names written by train.py and read by predict_global.py
CHECKPOINT_FILE = "bnn_best_checkpoint.pt"
X_MEAN_FILE = "x_train_mu.npy"
X_STD_FILE = "x_train_std.npy"


class Swish(nn.Module):
    """Swish activation x * sigmoid(beta * x) with a learnable beta."""

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class BNNChlModel(PyroModule):
    """
    Bayesian neural network with a heteroscedastic Gaussian output.

    A shared two-layer backbone feeds two separate heads:
      - mean head:        fc3_mu  -> fc_mu      (predictive mean of log10 Chl-a)
      - uncertainty head: fc3_ale -> fc_logvar  (log of the aleatoric variance)
    All weights and biases are random variables with Gaussian priors.
    """

    def __init__(self, h1=HIDDEN, h2=HIDDEN, prior_scale=PRIOR_SCALE):
        super().__init__()
        ps = prior_scale

        def make_linear(in_f, out_f):
            layer = PyroModule[nn.Linear](in_f, out_f)
            layer.weight = PyroSample(
                dist.Normal(0., ps).expand([out_f, in_f]).to_event(2))
            layer.bias = PyroSample(
                dist.Normal(0., ps).expand([out_f]).to_event(1))
            return layer

        # Shared backbone
        self.fc1_shared = make_linear(len(BANDS_NM), h1)
        self.fc2_shared = make_linear(h1, h2)

        # Mean head
        self.fc3_mu = make_linear(h2, h2)
        self.fc_mu = make_linear(h2, 1)

        # Uncertainty head; the log-variance bias has a prior mean of -4.6
        self.fc3_ale = make_linear(h2, h2)
        self.fc_logvar = PyroModule[nn.Linear](h2, 1)
        self.fc_logvar.weight = PyroSample(
            dist.Normal(0., ps).expand([1, h2]).to_event(2))
        self.fc_logvar.bias = PyroSample(
            dist.Normal(LOGVAR_BIAS_PRIOR, ps).expand([1]).to_event(1))

        # One learnable Swish per hidden layer
        self.act1 = Swish()
        self.act2 = Swish()
        self.act3_mu = Swish()
        self.act3_ale = Swish()

    def forward(self, x, y=None):
        h = self.act1(self.fc1_shared(x))
        h = self.act2(self.fc2_shared(h))

        h_mu = self.act3_mu(self.fc3_mu(h))
        mu = self.fc_mu(h_mu).squeeze(-1)

        h_ale = self.act3_ale(self.fc3_ale(h))
        logvar = self.fc_logvar(h_ale).squeeze(-1)
        sigma = torch.exp(0.5 * logvar).clamp(min=SIGMA_MIN)

        pyro.deterministic("mu", mu)
        pyro.deterministic("ale_std", sigma)

        if y is not None:
            # Beta-NLL (Seitzer et al., 2022): log-likelihood weighted by the
            # stop-gradient of sigma^(2*beta)
            sigma_sg = sigma.detach() ** (2 * BETA_NLL)
            log_prob = dist.Normal(mu, sigma).log_prob(y)
            with pyro.plate("data", x.shape[0]):
                pyro.factor("obs", (sigma_sg * log_prob).sum())
        else:
            with pyro.plate("data", x.shape[0]):
                pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        return mu


def build_model_and_guide():
    """Return a new model and its low-rank multivariate normal guide."""
    model = BNNChlModel()
    guide = AutoLowRankMultivariateNormal(model, rank=RANK)
    return model, guide


def snapshot_params():
    """Copy of the current (unconstrained) parameters in the Pyro param store."""
    return {k: v.detach().clone() for k, v in pyro.get_param_store().named_parameters()}


def restore_params(snapshot):
    """
    Restore a snapshot in place. The model and guide hold the same tensor
    objects as the param store, so copying into them restores the network
    itself; replacing the store entries (clear + pyro.param) would not.
    """
    with torch.no_grad():
        for k, v in pyro.get_param_store().named_parameters():
            v.copy_(snapshot[k])


def sync_activations(model):
    """
    Copy the Swish parameters held in the Pyro parameter store into the model.

    The Swish modules are plain torch modules: the network uses the values
    stored on the modules themselves, not the values in the parameter store.
    When a model is loaded from disk this makes the network use the loaded
    values.
    """
    store = pyro.get_param_store()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Swish):
                module.beta.copy_(store[f"{name}$$$beta"].detach())


# ---------------------------------------------------------------------------
# Preprocessing and inference
# ---------------------------------------------------------------------------
def _as_sample_matrix(samples, n_rows, site):
    """
    Reshape a posterior predictive site to (num_samples, n_rows).

    The auto guide fixes its prototype batch shape on its first call, so every
    evaluation site carries that (otherwise redundant) size-1 batch dimension:
    with a guide prototyped on a single observation row, ``mu`` comes back as
    (num_samples, 1, n_rows) instead of (num_samples, n_rows). Reducing the
    wrong axis is a silent error, so the shape is verified here.
    """
    samples = samples.detach()
    if samples.dim() < 2:
        raise ValueError(
            f"the predictive site {site!r} has shape {tuple(samples.shape)}: expected at "
            "least (num_samples, n_rows)")
    if samples.shape[-1] != n_rows:
        raise ValueError(
            f"the predictive site {site!r} holds {samples.shape[-1]} values per sample but "
            f"{n_rows} input rows were passed")
    batch_shape = tuple(samples.shape[1:-1])
    if any(dim != 1 for dim in batch_shape):
        raise ValueError(
            f"the predictive site {site!r} has shape {tuple(samples.shape)}: the extra batch "
            f"dimensions {batch_shape} are not supported. Build the predictive from a guide "
            "whose prototype was set with a single observation row.")
    return samples.reshape(samples.shape[0], n_rows)


def predictive_moments(predictive, x, seed=None):
    """
    Posterior predictive mean and the epistemic and aleatoric 1-sigma
    uncertainties of a batch of standardised inputs (log10 units), decomposed
    with the law of total variance.

    Returns three tensors with one value per input row. The Monte Carlo seed is
    only set when one is given, so by default repeated calls draw independent
    samples instead of replaying the same draws.
    """
    x = torch.as_tensor(x).float()
    n_rows = int(x.shape[0])
    if seed is not None:
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
    with torch.no_grad():
        samples = predictive(x)

    mu_s = _as_sample_matrix(samples["mu"].cpu(), n_rows, "mu")            # (S, n_rows)
    sigma_s = _as_sample_matrix(samples["ale_std"].cpu(), n_rows, "ale_std")
    return mu_s.mean(dim=0), mu_s.var(dim=0), (sigma_s ** 2).mean(dim=0)


def standardise(rrs, x_mean, x_std):
    """
    Rrs (sr-1, shape (N, 4)) -> standardised log10 Rrs.
    Rows with any non-finite band are returned as zeros and flagged in the
    returned mask so they can be skipped.
    """
    rrs = np.asarray(rrs, dtype=np.float64)
    if rrs.ndim != 2 or rrs.shape[1] != N_BANDS:
        raise ValueError(f"Rrs must be a 2-D array with {N_BANDS} bands, got shape {rrs.shape}")
    invalid = ~np.isfinite(rrs).all(axis=1)
    x = (np.log10(np.clip(np.nan_to_num(rrs, nan=1.0), RRS_FLOOR, None)) - x_mean) / x_std
    x[invalid] = 0.0
    return x.astype(np.float32), invalid


def predict(predictive, x, invalid, seed=42):
    """
    Posterior predictive inference on standardised inputs.

    Returns the predictive mean of log10(Chl-a) and the epistemic and
    aleatoric 1-sigma uncertainties (log10 units), decomposed with the law of
    total variance; invalid rows are NaN.

    ``seed`` reproduces the published dataset (42); pass None to draw fresh
    Monte Carlo samples and leave the global random state untouched.
    """
    n = len(invalid)
    mu = np.full(n, np.nan, dtype=np.float32)
    epi = np.full(n, np.nan, dtype=np.float32)
    ale = np.full(n, np.nan, dtype=np.float32)
    valid_idx = np.flatnonzero(~invalid)
    if valid_idx.size == 0:
        return mu, epi, ale

    mean, var_epi, var_ale = predictive_moments(predictive, x[valid_idx], seed=seed)
    mu[valid_idx] = mean.numpy()
    epi[valid_idx] = var_epi.sqrt().numpy()
    ale[valid_idx] = var_ale.sqrt().numpy()
    return mu, epi, ale


def load_trained_model(model_dir, num_samples=100):
    """
    Load the trained posterior and the training standardisation statistics.

    Returns (predictive, x_mean, x_std).
    """
    checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found: train the model first (python train.py --rrs "
            f"<rrs.npy> --chl <chl.npy> --out {model_dir}) or point --model-dir at a folder "
            "written by it")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "params" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no 'params' entry: it was not written by train.py")

    def statistics(name):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found: {model_dir} is not a complete model folder written by train.py")
        values = np.load(path)
        if values.shape != (N_BANDS,):
            raise ValueError(
                f"{path} has shape {values.shape}, expected ({N_BANDS},) for the "
                f"{N_BANDS} input bands")
        return values

    x_mean, x_std = statistics(X_MEAN_FILE), statistics(X_STD_FILE)

    pyro.clear_param_store()
    model, guide = build_model_and_guide()
    params = ckpt["params"]
    for key, value in params.items():
        pyro.param(key, value)
    # Initialise the guide once, as in training (with an observed target), so
    # that it binds to the loaded parameters
    with torch.no_grad():
        guide(torch.zeros(1, N_BANDS), torch.zeros(1))
    sync_activations(model)

    # A checkpoint that does not cover every parameter of the architecture
    # would silently leave the missing ones at their initial values.
    missing = sorted(set(pyro.get_param_store().keys()) - set(params))
    if missing:
        raise ValueError(
            f"{checkpoint_path} does not contain {missing}: it was written by a different "
            "architecture or hyper-parameters than the ones built here")

    predictive = Predictive(model, guide=guide, num_samples=num_samples,
                            return_sites=["mu", "ale_std"])
    return predictive, x_mean, x_std
