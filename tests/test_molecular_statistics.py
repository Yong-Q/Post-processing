"""Species-aware static observables and explicit symmetry/applicability checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from waterpost.histograms import Histograms
from waterpost.molecular_statistics import MolecularStatistics


def data(centers,kind='none',elements=(0,),axis=None):
    c=np.asarray(centers,float); n=len(c)
    return dict(centers=c,positions=c.copy(),elements=np.asarray(elements),owner=np.arange(n),
                molecule_valid=np.ones(n,bool),axis_kind=kind,
                axis=np.full((n,3),np.nan) if axis is None else np.asarray(axis,float),
                plane_normal=np.full((n,3),np.nan),geometry_samples={},attached_heavy_index=np.full(n,-1))


def test_tetrahedral_center_packing_is_generic():
    c=np.vstack([np.zeros(3),np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])])
    h=Histograms(); s=MolecularStatistics(h)
    m=s.frame(data(c,elements=[0]*5),np.eye(3)*20,0)
    assert m['coordination'][0]==4
    assert_allclose(m['packing_q'][0],1)
    assert_allclose(m['radial_cv'][0],0)
    assert m['nearest'].shape==(5,4)
    assert np.isinf(np.diag(m['distance'])).all()
    assert 'local/packing_q' in h.items
    assert not any(k.startswith('hbond/') for k in h.items)
    assert not s.applicability['hydrogen_bonds']
    assert not any(k.startswith('orientation/') for k in h.items)


def test_director_reversal_invariance_and_no_signed_p1():
    d=data([[2,2,2],[4,3,2]],'director',[0,0],[[1,0,0],[0,1,0]])
    a=Histograms(); b=Histograms()
    MolecularStatistics(a).frame(d,np.eye(3)*12,0)
    d['axis'][0]*=-1
    MolecularStatistics(b).frame(d,np.eye(3)*12,0)
    names=[k for k in a.items if k.startswith('orientation/')]
    assert names and not any('P1' in k for k in names)
    for name in names: assert_allclose(a.result(name)['counts'],b.result(name)['counts'])
    assert 'orientation/mutual_P2_vs_r' in names


def test_polar_axes_and_invalid_pair_mask():
    d=data([[2,2,2],[4,2,2]],'polar',[3,3],[[1,0,0],[1,0,0]])
    h=Histograms(); s=MolecularStatistics(h); s.frame(d,np.eye(3)*12,0)
    assert h.result('orientation/mutual_P1_vs_r')['n_finite']==2
    d['molecule_valid'][1]=False
    h2=Histograms(); m=MolecularStatistics(h2).frame(d,np.eye(3)*12,0)
    assert h2.result('orientation/mutual_P1_vs_r')['n_finite']==0
    assert np.isnan(m['coordination'][1])


@pytest.mark.parametrize('heavy',[2,3])
def test_hydrogen_bond_global_h_index_periodic_and_validity(heavy):
    # First H crosses the x boundary towards the other molecule's heavy atom.
    c=np.array([[11.4,3,3],[1.9,3,3.]])
    d=data(c,'polar',[heavy,1,1,heavy,1,1],[[1,0,0],[0,1,0]])
    d.update(positions=np.array([[11.4,3,3],[.3,3,3],[11.4,3.9,3],
                                 [1.9,3,3],[1.9,3.9,3],[1.9,3,3.9]]),
             owner=np.array([0,0,0,1,1,1]),attached_heavy_index=np.array([-1,0,0,-1,3,3]))
    h=Histograms(); s=MolecularStatistics(h); m=s.frame(d,np.eye(3)*12,0,expensive=True)
    assert m['hbond_triplets'].tolist()==[[0,1,1]]
    assert m['hbond_donors'].tolist()==[1,0]
    assert m['hbond_acceptors'].tolist()==[0,1]
    assert m['network']['largest_component']==2
    assert s.applicability['hydrogen_bonds']
    d['molecule_valid'][1]=False
    m=s.frame(d,np.eye(3)*12,1)
    assert len(m['hbond_triplets'])==0
    assert np.isnan(m['hbond_acceptors'][1])


def test_geometry_element_labels_pool_equivalents_and_mask_invalid():
    d=data([[1,1,1],[5,1,1]],elements=[0,0,1,1])
    d['molecule_valid'][1]=False
    d['elements']=np.array([0,1,0,1])
    d['geometry_samples']=dict(bond_lengths_A=np.array([1.1,1.2]),bond_molecule_index=np.array([0,1]),
        bond_atom_indices=np.array([[0,1],[2,3]]),angle_cos=np.array([-.3]),
        angle_molecule_index=np.array([0]),angle_atom_indices=np.array([[1,0,1]]),
        torsion_rad=np.array([0.,2*np.pi/3,4*np.pi/3]),torsion_molecule_index=np.zeros(3,int),
        torsion_atom_indices=np.tile([1,0,2,3],(3,1)))
    h=Histograms(); MolecularStatistics(h).frame(d,np.eye(3)*12,0)
    assert h.result('geometry/bond_C_H_A')['n_finite']==1
    assert h.result('geometry/angle_H_C_H_cos')['n_finite']==1
    assert h.result('geometry/torsion_cos3')['n_finite']==3


def test_voronoi_and_single_center_edge_case():
    h=Histograms(); m=MolecularStatistics(h).frame(data([[1,1,1]]),np.eye(3)*10,0,
                        voronoi={'volume':np.array([1000.]),'faces':np.array([6]),'surface_area':np.array([600.])})
    assert m['nearest'].shape==(1,0)
    assert np.isnan(m['packing_q'][0])
    assert_allclose(m['voronoi_shape'],[600**3/(36*np.pi*1000**2)])
    assert h.result('local/voronoi_volume_A3')['n_finite']==1


def test_periodic_image_invariance():
    d=data([[.2,2,2],[11.8,2,2],[4,5,3],[8,5,2],[4,2,8]],elements=[0]*5)
    cell=np.array([[12.,0,0],[2.,11.,0],[1.,2.,10.]])
    a=MolecularStatistics(Histograms()).frame(d,cell,0)
    d['centers']+=np.arange(15).reshape(5,3)@cell
    b=MolecularStatistics(Histograms()).frame(d,cell,0)
    for key in ['distance','coordination','packing_q','lsi','nearest']:
        assert_allclose(a[key],b[key],atol=1e-10,equal_nan=True)


def test_parallel_directions_remain_in_histogram_range_after_normalization():
    d=data([[1,1,1],[3,3,3]],'polar',[0,0],[[1,1,1],[1,1,1]])
    d['plane_normal']=np.ones((2,3))
    h=Histograms(); MolecularStatistics(h).frame(d,np.eye(3)*12,0)
    for name in ['orientation/mutual_P1_vs_r','orientation/mutual_P2_vs_r','orientation/plane_abs_cos_vs_r']:
        assert h.result(name)['n_in_range']==2
