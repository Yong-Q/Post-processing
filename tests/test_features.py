import importlib
import numpy as np
import pytest
from waterpost.static import local_metrics


def test_features_invariant_to_rotation_h_exchange_and_box_images():
    try: enc=importlib.import_module('waterpost.features')
    except ModuleNotFoundError:pytest.fail('Missing production feature encoder')
    rng=np.random.default_rng(9);cell=np.eye(3)*20
    o=rng.uniform(1,8,(12,3));oh=np.tile(np.array([[.96,0,0],[-.24,.93,0]]),(12,1,1))
    a=enc.encode(o,oh,cell,cutoff=7.)
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    oo=(o+20*rng.integers(-2,3,o.shape))@rot
    hh=oh[:,::-1]@rot;cc=cell@rot
    b=enc.encode(oo,hh,cc,cutoff=7.)
    np.testing.assert_allclose(a['invariants'],b['invariants'],rtol=1e-8,atol=1e-8)
    assert len(a['names'])==a['invariants'].shape[1]
    assert np.all(np.isfinite(a['invariants']))


@pytest.mark.parametrize('water_only',[False,True])
def test_features_arbitrary_rotation_and_neighbor_permutation(water_only):
    from waterpost.features import encode
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(731)
    cell=np.diag([18.,19.,20.]);o=rng.uniform(1,9,(10,3))
    oh=np.tile(np.array([[.96,0,0],[-.24,.93,0]]),(10,1,1))
    oh[-1]=np.nan
    a=encode(o,oh,cell,cutoff=7.,neighbor_water_only=water_only)['invariants']
    rot=Rotation.from_rotvec([.71,-.42,1.13]).as_matrix();permutation=rng.permutation(10)
    b=encode(o[permutation]@rot,oh[permutation]@rot,cell@rot,cutoff=7.,neighbor_water_only=water_only)['invariants']
    np.testing.assert_allclose(a[permutation],b,rtol=1e-8,atol=1e-8)
