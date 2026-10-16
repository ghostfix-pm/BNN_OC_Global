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

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
BANDS_NM = (443, 490, 510, 560)                 # Rrs input bands (nm)
RRS_VARIABLES = tuple(f"Rrs_{b}" for b in BANDS_NM)
RRS_FLOOR = 1e-6                                # Rrs (sr-1) is clipped here before log10

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
def standardise(rrs, x_mean, x_std):
    """
    Rrs (sr-1, shape (N, 4)) -> standardised log10 Rrs.
    Rows with any non-finite band are returned as zeros and flagged in the
    returned mask so they can be skipped.
    """
    rrs = np.asarray(rrs, dtype=np.float64)
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
    """
    n = len(invalid)
    mu = np.full(n, np.nan, dtype=np.float32)
    epi = np.full(n, np.nan, dtype=np.float32)
    ale = np.full(n, np.nan, dtype=np.float32)
    valid_idx = np.flatnonzero(~invalid)
    if valid_idx.size == 0:
        return mu, epi, ale

    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    with torch.no_grad():
        s = predictive(torch.from_numpy(x[valid_idx]))
    mu_s = s["mu"].cpu()          # (S, N_valid)
    sig_s = s["ale_std"].cpu()    # (S, N_valid)

    mu[valid_idx] = mu_s.mean(dim=0).numpy().ravel()
    epi[valid_idx] = mu_s.var(dim=0).sqrt().numpy().ravel()
    ale[valid_idx] = (sig_s ** 2).mean(dim=0).sqrt().numpy().ravel()
    return mu, epi, ale


def load_trained_model(model_dir, num_samples=100):
    """
    Load the trained posterior and the training standardisation statistics.

    Returns (predictive, x_mean, x_std).
    """
    ckpt = torch.load(os.path.join(model_dir, CHECKPOINT_FILE),
                      map_location="cpu", weights_only=True)
    x_mean = np.load(os.path.join(model_dir, X_MEAN_FILE))
    x_std = np.load(os.path.join(model_dir, X_STD_FILE))

    pyro.clear_param_store()
    model, guide = build_model_and_guide()
    for k, v in ckpt["params"].items():
        pyro.param(k, v)
    # Initialise the guide once, as in training (with an observed target), so
    # that it binds to the loaded parameters
    with torch.no_grad():
        guide(torch.zeros(1, len(BANDS_NM)), torch.zeros(1))
    sync_activations(model)

    predictive = Predictive(model, guide=guide, num_samples=num_samples,
                            return_sites=["mu", "ale_std"])
    return predictive, x_mean, x_std
