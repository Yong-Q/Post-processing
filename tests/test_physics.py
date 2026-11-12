import importlib
import numpy as np
import pytest


def module(name):
    try:
        return importlib.import_module('waterpost.' + name)
    except ModuleNotFoundError:
        pytest.fail('Missing production module waterpost.' + name)


def test_minimum_image_and_triclinic():
    g = module('geometry')
    np.testing.assert_allclose(g.mic(np.array([9., 0, 0]), np.eye(3)*10), [-1, 0, 0])
    cell = np.array([[10., 0, 0], [4., 8, 0], [0, 0, 9.]])
    np.testing.assert_allclose(g.mic(np.array([4.1, 7.9, 0]), cell), [.1, -.1, 0], atol=1e-12)


def test_perfect_tetrahedron_and_rotation():
    g = module('geometry')
    v = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1.]])
    assert g.tetrahedral_q(v) == pytest.approx(1.)
    rot = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    assert g.tetrahedral_q(v @ rot) == pytest.approx(1.)


def test_hbond_direction_and_pbc():
    st = module('static')
    o = np.array([[0,0,0],[2.8,0,0.]])
    oh = np.array([[[1,0,0],[0,1,0]],[[-1,0,0],[0,1,0.]]])
    edges = st.hydrogen_bonds(o, oh, np.eye(3)*10, 3.5, 30)
    assert set(map(tuple, edges)) == {(0,0,1),(1,0,0)}
    edges2 = st.hydrogen_bonds(o, oh[:,::-1], np.eye(3)*10, 3.5, 30)
    assert set(map(tuple, edges2[:,[0,2]])) == {(0,1),(1,0)}


def test_force_torque_known_couple():
    g = module('geometry')
    r = np.array([[0.,0,0],[1.,0,0],[-1.,0,0]])
    f = np.array([[0.,0,0],[0.,1,0],[0.,-1,0]])
    force, torque = g.force_torque(r, f)
    np.testing.assert_allclose(force, [0,0,0])
    np.testing.assert_allclose(torque, [0,0,2])


def test_histogram_probability_and_block_error():
    h = module('histograms').Histograms(block_size=1)
    for frame, values in enumerate([[.2,.2],[.8,.8]]):
        h.add('x', np.asarray(values)[:,None], [np.array([0,.5,1.])], frame)
    result = h.result('x')
    np.testing.assert_allclose(result['probability'], [.5,.5])
    np.testing.assert_allclose(result['density'], [1.,1.])
    assert np.all(result['probability_stderr'] > 0)


def test_ballistic_msd_and_rotational_persistence():
    d = module('dynamics')
    pos = np.zeros((5,2,3));pos[:,:,0] = np.arange(5)[:,None]*.2
    u = np.zeros_like(pos);u[:,:,0]=1
    result=d.motion(pos,u, np.arange(5)*.5, max_lag=2)
    np.testing.assert_allclose(result['msd'], [0,.04,.16],atol=1e-12)
    np.testing.assert_allclose(result['c2'], [1,1,1])
    assert 'diffusion' not in result  # no automatic fit to a ballistic curve


def test_dump_sorts_ids_and_recovers_water_across_pbc(tmp_path):
    io=module('io');g=module('geometry')
    p=tmp_path/'tiny.dump'
    p.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id mol type x y z fx fy fz\n3 1 2 9.8 1.9 1 0 0 0\n1 1 4 9.8 1 1 0 0 0\n2 1 2 0.7 1 1 0 0 0\n')
    frame=next(io.read_dump(p));top=io.WaterTopology(frame,4,2)
    w=top.extract(frame)
    np.testing.assert_allclose(w['oh'][0], [[.9,0,0],[0,.9,0]],atol=1e-12)
    assert w['mol_ids'].tolist()==[1]


def test_ring_definition_triangle_and_chain():
    st=module('static')
    assert st.shortest_rings(3,[(0,1),(1,2),(2,0)],max_size=8)==[(0,1,2)]
    assert st.shortest_rings(3,[(0,1),(1,2)],max_size=8)==[]


def test_ideal_gas_rdf_normalization():
    h=module('histograms')
    edges=np.array([0.,1,2])
    shell=4*np.pi/3*np.diff(edges**3)
    ideal=100*99/1000*shell
    np.testing.assert_allclose(h.rdf_normalize(ideal,edges,100,100,1000,same=True),[1,1])


def test_no_neighbor_is_valid_not_fake_tetrahedron():
    st=module('static')
    o=np.array([[0.,0,0],[7,7,7]])
    oh=np.tile(np.array([[1.,0,0],[0,1,0]]),(2,1,1))
    result=st.local_metrics(o,oh,np.eye(3)*20,3.5,6.)
    assert np.all(result['coordination']==0)
    assert np.all(np.isnan(result['q']))


def test_cli_refuses_existing_output(tmp_path):
    import subprocess,sys
    p=tmp_path/'existing';p.mkdir()
    r=subprocess.run([sys.executable,'-m','waterpost','analyze','--input','missing.dump','--output',str(p)],capture_output=True,text=True)
    assert r.returncode != 0
    assert 'already exists' in r.stderr.lower()
