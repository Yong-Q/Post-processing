import itertools
import numpy as np
from waterpost.geometry import mic, unique_radius, water_frame
from waterpost.io import read_dump
from waterpost.static import local_metrics, hydrogen_bonds, periodic_voronoi
from waterpost.collective import modes, window_counts
from waterpost.dynamics import inferred_nojump


def test_triclinic_matches_exhaustive_lattice_search():
    rng=np.random.default_rng(47)
    cell=np.array([[10.,0,0],[4.8,8.,0],[-3.,2.7,9.]])
    vectors=rng.uniform(-2,2,(80,3))@cell
    shifts=np.array(list(itertools.product(range(-5,6),repeat=3)))@cell
    reference=np.min(np.linalg.norm(vectors[:,None,:]-shifts[None,:,:],axis=-1),axis=1)
    np.testing.assert_allclose(np.linalg.norm(mic(vectors,cell),axis=-1),reference,atol=1e-10)


def test_geometry_and_hbond_invariant_to_individual_box_images():
    cell=np.array([[12.,0,0],[3.,11.,0],[-2.,1.,10.]])
    o=np.array([[.1,.2,.3],[2.8,.2,.3],[.1,3.,.3],[.1,.2,3.1],[8,8,8.]])
    oh=np.tile(np.array([[.96,0,0],[-.24,.93,0]]),(5,1,1))
    shifted=o+np.array([[2,-1,0],[-2,0,1],[0,1,-1],[1,1,1],[-1,-1,2]])@cell
    a=local_metrics(o,oh,cell);b=local_metrics(shifted,oh,cell)
    for name in ['q','coordination','nearest','lsi']:
        np.testing.assert_allclose(a[name],b[name],atol=1e-10,equal_nan=True)
    np.testing.assert_array_equal(hydrogen_bonds(o,oh,cell),hydrogen_bonds(shifted,oh,cell))
    _,_,ro,rh=modes(o,oh,cell,nmax=1)
    _,_,so,sh=modes(shifted,oh,cell,nmax=1)
    np.testing.assert_allclose(ro,so,atol=1e-10);np.testing.assert_allclose(rh,sh,atol=1e-10)
    centers=np.array([[.1,.2,.3],[.5,.5,.5]])
    np.testing.assert_array_equal(window_counts(o,cell,centers,[2.,3.]),window_counts(shifted,cell,centers,[2.,3.]))


def test_triclinic_dump_bound_correction(tmp_path):
    # True cell: rows [10,0,0], [4,8,0], [-2,1,9], origin [2,3,4].
    p=tmp_path/'tilt.dump'
    p.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS xy xz yz pp pp pp\n0 16 4\n3 12 -2\n4 13 1\nITEM: ATOMS id mol type xs ys zs\n1 1 4 .2 .2 .2\n2 1 2 .29 .2 .2\n3 1 2 .2 .31 .2\n')
    f=next(read_dump(p))
    np.testing.assert_allclose(f.cell,[[10,0,0],[4,8,0],[-2,1,9]])
    np.testing.assert_allclose(f.origin,[2,3,4])
    np.testing.assert_allclose(f.positions[0],[2.4,1.8,1.8])


def test_nojump_tracks_crossing_not_wrapped_displacement():
    p=np.array([[[9.8,0,0]],[[.2,0,0]],[[.6,0,0]]])
    result=inferred_nojump(p,np.tile(np.eye(3)*10,(3,1,1)))
    np.testing.assert_allclose(result[:,0,0],[9.8,10.2,10.6])


def test_voronoi_tiles_periodic_box():
    import pytest
    pytest.importorskip('freud')
    rng=np.random.default_rng(13);cell=np.diag([12.,13.,14.]);o=rng.random((60,3))@cell
    a=periodic_voronoi(o,cell)
    b=periodic_voronoi(o+np.array([12.,-26.,14.]),cell)
    assert a['volume'].sum()==pytest.approx(np.linalg.det(cell),rel=1e-5)
    np.testing.assert_allclose(a['volume'],b['volume'],rtol=3e-5,atol=3e-5)


def test_molecular_frame_is_h_exchange_equivalent():
    oh=np.array([[[.96,0,0],[-.24,.93,0]]])
    a=water_frame(oh);b=water_frame(oh[:,::-1])
    np.testing.assert_allclose(a[:,0],b[:,0])
    np.testing.assert_allclose(a[:,1:],-b[:,1:])
