import numpy as np
import pytest
from scipy import sparse

from leetfly.connectome import nulls
from leetfly.data import splits
from leetfly.data.leetcode import clean_text
from leetfly.features.antennal_lobe import Nose
from leetfly.features.odor import analyze, nmf_transform
from leetfly.model.dopamine import DopamineReadout, online_fit
from leetfly.model.mushroom_body import kwta


def random_wiring(rng, n_kc=300, n_glom=20, claws=5):
    rows = np.repeat(np.arange(n_kc), claws)
    cols = np.concatenate([rng.choice(n_glom, claws, replace=False) for _ in range(n_kc)])
    vals = rng.integers(5, 40, size=len(rows)).astype(float)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_kc, n_glom))


@pytest.mark.parametrize("balanced", [True, False])
def test_closed_form_equals_online_rule(balanced):
    rng = np.random.default_rng(0)
    codes = rng.random((120, 60)) < 0.1
    y = rng.random((120, 5)) < 0.3
    y[0] = True  # every class has positives and negatives
    y[1] = False
    closed = DopamineReadout(eta=3.0, balanced=balanced).fit(codes, y)
    w_plus, w_minus = online_fit(codes, y, eta=3.0, balanced=balanced)
    np.testing.assert_allclose(closed.w_plus, w_plus, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(closed.w_minus, w_minus, rtol=1e-9, atol=1e-12)


def test_order_of_training_does_not_matter():
    rng = np.random.default_rng(1)
    codes = rng.random((80, 40)) < 0.1
    y = rng.random((80, 3)) < 0.4
    perm = rng.permutation(80)
    a = online_fit(codes, y, 2.0, True)
    b = online_fit(codes[perm], y[perm], 2.0, True)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_kwta_exact_sparsity_and_positive_only():
    rng = np.random.default_rng(2)
    drive = rng.random((50, 400))
    assert (kwta(drive, 20).sum(axis=1) == 20).all()
    drive[:, 5:] = 0  # only 5 KCs have any input
    assert (kwta(drive, 20).sum(axis=1) == 5).all()


def test_degree_preserving_null_keeps_both_degree_sequences():
    rng = np.random.default_rng(3)
    w = random_wiring(rng)
    shuffled = nulls.degree_preserving(w, rng)
    b0, b1 = (w > 0), (shuffled > 0)
    np.testing.assert_array_equal(np.asarray(b0.sum(1)).ravel(), np.asarray(b1.sum(1)).ravel())
    np.testing.assert_array_equal(np.asarray(b0.sum(0)).ravel(), np.asarray(b1.sum(0)).ravel())
    assert sorted(w.data) == sorted(shuffled.data)  # weights travel with edges
    assert (b0 != b1).nnz > 0.5 * b0.nnz  # and it actually rewired


def test_uniform_null_keeps_each_kcs_inputs_and_weights():
    rng = np.random.default_rng(4)
    w = random_wiring(rng)
    u = nulls.uniform(w, rng)
    for i in range(w.shape[0]):
        assert sorted(w[i].data) == sorted(u[i].data)


def test_nose_places_receptors_on_glomeruli():
    nose = Nose(perm=np.array([2, 0, 1]), gain=np.array([1.0, 2.0, 0.5]))
    x = nose.glomerular_input(np.array([[10.0, 20.0, 30.0]]))
    # receptor 1 -> glom 0 (gain 1), receptor 2 -> glom 1 (gain 2), receptor 0 -> glom 2 (gain 0.5)
    np.testing.assert_allclose(x, [[20.0, 60.0, 5.0]])


def test_nmf_transform_is_nonnegative_and_fits():
    rng = np.random.default_rng(5)
    h = rng.random((4, 30))
    w_true = rng.random((10, 4))
    w = nmf_transform(w_true @ h, h, iters=2000)
    assert (w >= 0).all()
    np.testing.assert_allclose(w @ h, w_true @ h, rtol=0.05)


def test_text_cleaning_restores_magnitudes():
    s = clean_text("1 <= n <= 105\n-109 <= x <= 109\nreturn it modulo 109 + 7. Better than O(n2)?")
    assert "pow5" in s and "-pow9" in s and "O(n^2)" in s and "pow9 + 7" in s
    assert "n^2" in analyze(s) and "pow5" in analyze(s)


def test_family_keys_group_sequels_but_not_variables():
    keys = splits.family_keys(
        ["two-sum", "two-sum-ii-input-array-is-sorted", "jump-game", "jump-game-vii",
         "minimum-number-of-operations-to-make-x-and-y-equal", "minimum-number-of-operations-to-make-array-empty"]
    )
    assert keys[0] == keys[1] and keys[2] == keys[3]
    assert keys[4] != keys[5]
