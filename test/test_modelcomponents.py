import pytest
import sklearn.mixture
import numpy as np
import torch
import h5py

from trackertraincode.neuralnets.modelcomponents import GaussianMixture, LocalToGlobalCoordinateOffset
from trackertraincode.neuralnets.rotrepr import QuatRepr

@pytest.fixture()
def rng() -> np.random.RandomState:
    return np.random.RandomState(seed=123) 


@pytest.fixture()
def sklearn_gmm(rng : np.random.RandomState):
    data = rng.normal(loc=[1.,2.],scale=[3.,4.], size=(100,2))
    gmm = sklearn.mixture.GaussianMixture(n_components=3, covariance_type='diag')
    gmm.fit(data)
    yield gmm


@pytest.fixture()
def gmm(sklearn_gmm : sklearn.mixture.GaussianMixture):
    return GaussianMixture.from_sklearn(sklearn_gmm)


class TestGaussianMixture:
    def test_gmm_log_likelihood(self, gmm : GaussianMixture, sklearn_gmm : sklearn.mixture.GaussianMixture, rng : np.random.RandomState):
        x = rng.uniform(0.,1.,size=(10,2))
        
        x_gmm = gmm(torch.from_numpy(x)).numpy()

        x_sklearn = sklearn_gmm.score_samples(x)

        assert x_gmm.shape==(10,)
        assert x_sklearn.shape==(10,)
        np.testing.assert_allclose(x_gmm, x_sklearn)
    
    def test_save_load_hdf5(self, gmm : GaussianMixture):
        f = h5py.File('foo', 'w', driver='core', backing_store=False)
        g = gmm.save_to_hdf5(f, 'gmm')
        restored = GaussianMixture.from_hdf5(g)
        np.testing.assert_array_equal(gmm.weights, restored.weights)
        np.testing.assert_array_equal(gmm.means, restored.means)
        np.testing.assert_array_equal(gmm.cov, restored.cov)
        np.testing.assert_array_equal(gmm.scales_inv, restored.scales_inv)


def test_local_to_global_transform_offset():
    from scipy.spatial.transform import Rotation

    tr = LocalToGlobalCoordinateOffset()
    tr.p.data[:] = torch.tensor([ 1., 3., 4., 2. ])

    r = Rotation.from_rotvec([ 0., 0.5, 0.])
    q = r.as_quat().astype(np.float32)
    c = np.asarray([ 1., 2., 3. ], dtype=np.float32)

    pred_quat, pred_coord = tr(QuatRepr(torch.from_numpy(q)[None,...]), torch.from_numpy(c)[None,...], set_id=None)
    pred_r = Rotation.from_quat(pred_quat.value.detach().numpy())
    pred_c = pred_coord.detach().numpy()
    expect_scale = 3.*np.exp(2.)
    expect_c = -(r.apply(expect_scale*np.array([ 3., 4., 0. ]))[[2,1]]) + np.array([1.,2.])
    print ((r.inv()*pred_r).magnitude())
    print (pred_c, expect_c, expect_scale)