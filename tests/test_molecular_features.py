"""Physical symmetry regressions for the common molecular representation."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from waterpost.molecular_features import encode_molecules


def sample():
    centers = np.array([[3., 3., 3.], [6., 4., 3.]])
    positions = np.array([[3.,3.,3.],[3.8,3.,3.],[3.,3.7,3.2],
                          [6.,4.,3.],[6.6,4.,3.],[6.,4.8,3.1],[4.,5.,4.]])
    return centers, positions, np.array([3,1,1,2,1,1,1]), np.array([0,0,0,1,1,1,-1]), np.eye(3)*20


def test_contract_and_missing_channels():
    result = encode_molecules(*sample())
    assert result['features'].shape == (2,508)
    assert len(result['names']) == len(set(result['names'])) == 508
    assert result['coefficients'].shape == (2,4,3,16)
    assert result['shape_moments'].shape == (2,4,16)
    assert result['shape_strength'].shape == (2,4,4)
    assert_allclose(result['coefficients'][:,0],0)
    assert np.isfinite(result['features']).all()


def test_global_rotation_translation_and_independent_box_images():
    c,p,e,o,cell = sample()
    base = encode_molecules(c,p,e,o,cell)
    rotation = Rotation.from_rotvec([.4,-.8,.6]).as_matrix()
    shift=np.array([1.7,-.2,2.])
    turned = encode_molecules(c@rotation+shift,p@rotation+shift,e,o,cell@rotation)
    assert_allclose(base['features'],turned['features'],atol=2e-12)
    rng=np.random.default_rng(5)
    imaged=encode_molecules(c+rng.integers(-4,5,c.shape)@cell,p+rng.integers(-4,5,p.shape)@cell,e,o,cell)
    assert_allclose(base['features'],imaged['features'],atol=2e-12)


def test_atom_and_center_permutations():
    c,p,e,o,cell=sample()
    base=encode_molecules(c,p,e,o,cell)
    perm=np.array([6,3,1,4,0,5,2])
    remapped=np.where(o<0,-1,1-o)
    result=encode_molecules(c[::-1],p[perm],e[perm],remapped[perm],cell)
    assert result['names']==base['names']
    assert_allclose(result['features'][::-1],base['features'],atol=1e-13)


def test_rotating_only_central_shape_changes_coupling():
    c,p,e,o,cell=sample()
    base=encode_molecules(c,p,e,o,cell)
    rotated=p.copy()
    rotated[o==0]=(p[o==0]-c[0])@Rotation.from_rotvec([.8,.2,.5]).as_matrix()+c[0]
    result=encode_molecules(c,rotated,e,o,cell)
    coupling=[i for i,name in enumerate(base['names']) if name.startswith('coupling_')]
    powers=[i for i,name in enumerate(base['names']) if name.startswith('env_power_')]
    assert_allclose(base['features'][0,powers],result['features'][0,powers])
    assert np.max(np.abs(base['features'][0,coupling]-result['features'][0,coupling]))>1e-4


def test_self_exclusion_orphan_hydrogen_and_valid_filter():
    c,p,e,o,cell=sample()
    result=encode_molecules(c,p,e,o,cell,valid=[True,False])
    assert np.isnan(result['features'][1]).all()
    assert result['neighbor_counts'][0]==4
    filtered=encode_molecules(c,p,e,o,cell,valid=[True,False],neighbor_valid_only=True)
    assert filtered['neighbor_counts'][0]==0
    assert_allclose(filtered['coefficients'][0],0)
    one=encode_molecules(c[:1],p[:3],e[:3],o[:3],cell)
    assert_allclose(one['coefficients'],0)
    assert_allclose(one['shape_moments'][0,1,0],2/np.sqrt(4*np.pi))


@pytest.mark.parametrize('kind',['co2','ch4','c2h4'])
def test_equivalent_atom_symmetries_without_unique_axes(kind):
    if kind=='co2':
        local=np.array([[0.,0.,0.],[1.2,0.,0.],[-1.2,0.,0.]])
        e=np.array([0,3,3]); perm=[0,2,1]
    elif kind=='ch4':
        local=np.vstack([np.zeros(3),np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*.6])
        e=np.array([0,1,1,1,1]); perm=[0,3,4,2,1]
    else:
        local=np.array([[-.7,0,0],[.7,0,0],[-1.2,.8,0],[-1.2,-.8,0],[1.2,.8,0],[1.2,-.8,0]])
        e=np.array([0,0,1,1,1,1]); perm=[1,0,4,5,2,3]
    p=np.vstack([local,[2.3,1.1,.4]])+4
    elem=np.r_[e,1]; owner=np.r_[np.zeros(len(e),dtype=int),-1]
    base=encode_molecules([[4,4,4]],p,elem,owner,np.eye(3)*20)
    idx=np.r_[perm,len(e)]
    result=encode_molecules([[4,4,4]],p[idx],elem[idx],owner[idx],np.eye(3)*20)
    assert_allclose(base['features'],result['features'],atol=1e-12)
    if kind=='co2':
        assert_allclose(base['shape_strength'][0,:,1],0,atol=1e-30)
    if kind=='ch4':
        assert base['shape_strength'][0,1,3]>0
        assert_allclose(base['shape_strength'][0,1,1:3],0,atol=1e-28)


def test_triclinic_images_and_empty_arrays():
    c,p,e,o,_=sample(); cell=np.array([[12.,0,0],[4.,10.,0],[-2.,1.,11.]])
    base=encode_molecules(c,p,e,o,cell)
    result=encode_molecules(c+np.array([[2,-1,3],[-2,3,1]])@cell,p+np.arange(21).reshape(7,3)@cell,e,o,cell)
    assert_allclose(base['features'],result['features'],atol=1e-11)
    empty=encode_molecules(np.empty((0,3)),np.empty((0,3)),[],[],cell)
    assert empty['features'].shape==(0,508)


@pytest.mark.parametrize('kwargs',[{'cutoff':0},{'nrad':0},{'lmax':-1}])
def test_invalid_basis_parameters(kwargs):
    with pytest.raises(ValueError): encode_molecules(*sample(),**kwargs)
