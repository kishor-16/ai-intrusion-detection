"""
SHAP-value estimation via Expected Gradients — pure PyTorch, no shap package.

This is the SAME estimator shap.GradientExplainer implements (Expected
Gradients, Erion et al. 2021, "Improving performance of deep learning models
with axiomatic attribution priors and expected gradients"): SHAP values are
approximated by E_{b~background, a~U(0,1)} [ (x - b) * ∇f(b + a(x - b)) ].

We implement it directly because the shap package's numba/llvmlite native
dependencies crash silently on some Windows setups — an unacceptable risk for
a live demo. Pure torch = if training worked, explanations work.

Used by: ml-inference-service/explain/precompute_shap.py (offline class
profiles) and ml-inference-service/main.py /explain (live per-instance).
"""

import torch


def expected_gradients(net, x, background, target_class, k=64, seed=42):
    """SHAP-value estimate for one input.

    net: eval-mode torch module (logits out)
    x: (F,) float tensor — scaled feature vector
    background: (N, F) float tensor — scaled training sample
    target_class: int — class whose score is attributed
    k: number of (baseline, interpolation) samples
    returns: (F,) signed attribution tensor
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, background.shape[0], (k,), generator=g)
    b = background[idx]                                    # (k, F)
    alpha = torch.rand(k, 1, generator=g)                  # (k, 1)
    pts = (b + alpha * (x.unsqueeze(0) - b)).detach().requires_grad_(True)
    net.zero_grad(set_to_none=True)
    out = net(pts)[:, target_class].sum()
    out.backward()
    attr = ((x.unsqueeze(0) - b) * pts.grad).mean(dim=0)   # (F,)
    return attr.detach()
