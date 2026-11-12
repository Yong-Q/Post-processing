import numpy as np
from waterpost.histograms import Histograms
from waterpost.analysis import Statistics
from waterpost.static import local_metrics
from waterpost.collective import rdf_diagnostics


def test_h_voxels_include_h_inside_cutoff_even_if_oxygen_is_outside():
    o=np.array([[0.,0,0],[3.2,0,0]])
    oh=np.array([[[.96,0,0],[-.24,.93,0]],[[-.9,0,0],[.2,.93,0]]])
    cell=np.eye(3)*20;m=local_metrics(o,oh,cell,2.,3.)
    h=Histograms();st=Statistics(h,2.,3.,4.)
    st.frame(o,oh,cell,m,np.empty((0,3),int),0,do_pairs=False)
    assert h.result('bodyframe/O_voxels')['counts'].sum()==0
    assert h.result('bodyframe/H_voxels')['counts'].sum()==4  # two equivalent H-label frames


def test_nan_weight_does_not_poison_histogram():
    h=Histograms()
    h.add('moment',np.array([[.2],[.3]]),[np.array([0.,1.])],0,weights=np.array([2.,np.nan]),expected=np.array([1.]))
    np.testing.assert_allclose(h.result('moment')['g'],[2.])


def test_finite_n_coordination_uses_other_particle_density():
    edges=np.array([0.,1.,2.]);g=np.ones(2)
    out=rdf_diagnostics(edges,g,6/1000,coordination_rho=5/1000)
    np.testing.assert_allclose(out['integrated_coordination'],5/1000*4*np.pi/3*np.array([1.,8.]))
