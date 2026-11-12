import importlib
import numpy as np
import pytest


def test_periodic_oo_counts_across_boundary():
    try:rdf=importlib.import_module('waterpost.rdf')
    except ModuleNotFoundError:pytest.fail('Missing direct RDF-only processing module')
    o=np.array([[.1,1,1],[9.9,1,1]])
    counts=rdf.periodic_oo_counts(o,np.eye(3)*10,np.array([0.,.5,1.,2.]))
    np.testing.assert_array_equal(counts,[2,0,0])


def test_periodic_oo_counts_match_triclinic_bruteforce():
    try:rdf=importlib.import_module('waterpost.rdf')
    except ModuleNotFoundError:pytest.fail('Missing direct RDF-only processing module')
    from waterpost.geometry import mic
    cell=np.array([[10.,0,0],[3.,9.,0],[1.,2.,11.]])
    o=np.array([[.1,.1,.1],[9.9,.1,.1],[3.,4.,5.]])
    edges=np.array([0.,.5,1.,2.,3.])
    d=np.linalg.norm(mic(o[None,:,:]-o[:,None,:],cell),axis=-1);np.fill_diagonal(d,np.inf)
    expected=np.histogram(d[np.isfinite(d)],bins=edges)[0]
    np.testing.assert_array_equal(rdf.periodic_oo_counts(o,cell,edges),expected)


def test_prebinned_counts_preserve_rdf_exposure_and_sample_count():
    from waterpost.histograms import Histograms
    h=Histograms();assert hasattr(h,'add_binned')
    h.add_binned('rdf',np.array([2.,0.]),[np.array([0.,1.,2.])],0,np.array([1.,2.]),total_samples=6)
    out=h.result('rdf')
    np.testing.assert_allclose(out['g'],[2,0])
    assert out['n_samples']==6
    assert out['n_in_range']==2


def test_batch_rdf_only_runs_without_mu_or_local_feature_calculation(tmp_path):
    import subprocess,sys,json,h5py
    from pathlib import Path
    from test_pipeline import write_fixture
    data=tmp_path/'data';data.mkdir();write_fixture(data/'H2O-T986.4-example.dump')
    out=tmp_path/'batch'
    r=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'batch.py'),'--input-dir',str(data),
                      '--pattern','*.dump','--output-root',str(out),'--mode','rdf-only','--rdf-max','5','--bins','40','--max-frames','2'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    report=json.loads((out/'batch_report.json').read_text());assert report[0]['exit_code']==0
    state=Path(report[0]['output'])
    assert not (state/'environment.h5').exists()
    with h5py.File(state/'direct_correlations.h5') as f:
        assert 'c2/c2_r' in f
        assert 'c1/c1' not in f
