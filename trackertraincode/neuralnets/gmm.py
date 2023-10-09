'''Gaussian Mixture Model'''
import pickle
import torch

from trackertraincode.gmm_torch.gmm import GaussianMixture


def unpickle_scipy_gmm(filename):
    '''WARNING only "diag" covariance supported.'''
    with open(filename, 'rb') as f:
        scipy_gmm = pickle.load(f)      

    gmm = GaussianMixture(
        n_components=scipy_gmm.n_components,
        n_features=scipy_gmm.means_.shape[-1],
        covariance_type='diag',
        mu_init=torch.from_numpy(scipy_gmm.means_)[None,:,:,],
        var_init=torch.from_numpy(scipy_gmm.covariances_)[None,:,:]
    )
    gmm.pi.data[...] = torch.from_numpy(scipy_gmm.weights_[None,:,None])
    gmm.params_fitted = True
    return gmm