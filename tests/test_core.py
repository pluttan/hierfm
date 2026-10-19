import numpy as np
from scipy.stats import norm

from hierfm import covariance as cov
from hierfm import dependence, metrics
from hierfm import hierarchy as hier
from hierfm import reconcile as rec


def small_tree():
    recs = [{"id": "T", "parent": None}, {"id": "A", "parent": "T"}, {"id": "B", "parent": "T"},
            {"id": "A1", "parent": "A"}, {"id": "A2", "parent": "A"}, {"id": "B1", "parent": "B"},
            {"id": "C", "parent": "T"}]                   # C is a leaf on the middle level
    return hier.from_nodes(["top", "mid", "leaf"], recs)


def test_summing_matrix():
    h = small_tree()
    assert h.nodes[0] == "T" and set(h.bottom) == {"A1", "A2", "B1", "C"}
    assert h.S[0].sum() == h.m
    assert np.allclose(h.S[[h.nodes.index(b) for b in h.bottom]], np.eye(h.m))


def test_every_method_gives_coherent_samples():
    h = small_tree()
    rng = np.random.default_rng(0)
    E = rng.standard_normal((60, h.n))
    prior = cov.prior_corr(h, "block", rho=0.4)
    x = rng.standard_normal((50, 24, h.n))
    for m in ["bu", "ols", "wls_struct", "wls_var", "mint_sample", "mint_shrink", "prior_shrink", "prior_only"]:
        G, _ = rec.method_matrix(m, h, E, prior)
        y = rec.apply(h, G, x)
        bottom = y[..., [h.nodes.index(b) for b in h.bottom]]
        assert np.allclose(y, bottom @ h.S.T), m
        assert np.allclose(h.S @ G @ h.S, h.S), m             # projection leaves coherent vectors alone


def test_shrink_weight_falls_with_history():
    rng = np.random.default_rng(1)
    C = np.array([[1, .6, .3], [.6, 1, .2], [.3, .2, 1]])
    L = np.linalg.cholesky(C)
    lams = [cov.shrunk(rng.standard_normal((n, 3)) @ L.T, np.eye(3))[1] for n in (10, 100, 3000)]
    assert lams[0] > lams[1] > lams[2]


def test_crps_matches_closed_form():
    rng = np.random.default_rng(2)
    s = rng.standard_normal((20000, 1, 1))
    y = np.array([[0.7]])
    exact = 0.7 * (2 * norm.cdf(0.7) - 1) + 2 * norm.pdf(0.7) - 1 / np.sqrt(np.pi)
    assert abs(metrics.crps(s, y)[0, 0] - exact) < 0.01
    assert np.isclose(metrics.crps(np.full((5, 1, 1), 2.0), np.array([[3.0]]))[0, 0], 1.0)


def test_schaake_keeps_marginals_and_copies_ranks():
    rng = np.random.default_rng(3)
    s = rng.standard_normal((100, 4, 3))
    tpl = rng.standard_normal((100, 4, 3))
    tpl[..., 1] = tpl[..., 0]                               # node 1 moves exactly with node 0
    out = dependence.reorder(s, tpl, rng)
    assert np.allclose(np.sort(out, 0), np.sort(s, 0))
    r0, r1 = out[..., 0].argsort(0).argsort(0), out[..., 1].argsort(0).argsort(0)
    assert (r0 == r1).all()


def test_independent_sum_is_too_narrow_and_copula_fixes_it():
    """The mechanism of the paper on a toy: two correlated nodes, independent draws undercover the sum."""
    rng = np.random.default_rng(4)
    levels = np.linspace(0.05, 0.95, 19)
    q = np.broadcast_to(norm.ppf(levels), (2, 1, 19)).copy()
    truth = rng.multivariate_normal([0, 0], [[1, .9], [.9, 1]], 2000).sum(1)
    corr = np.array([[1, .9], [.9, 1]])
    cov_ind = cov_cop = 0
    for k, t in enumerate(truth):
        ind = dependence.draw(levels, q, 200, "independent", rng).sum(-1)[:, 0]
        cop = dependence.draw(levels, q, 200, "gauss_copula", rng, corr=corr).sum(-1)[:, 0]
        cov_ind += np.quantile(ind, .1) <= t <= np.quantile(ind, .9)
        cov_cop += np.quantile(cop, .1) <= t <= np.quantile(cop, .9)
    assert cov_ind / len(truth) < 0.72 < cov_cop / len(truth)
