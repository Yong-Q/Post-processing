import importlib
import numpy as np
import pytest


def correlations():
    try:return importlib.import_module('waterpost.correlations')
    except ModuleNotFoundError:pytest.fail('Missing c1/c2 module')


def test_ideal_gas_has_zero_direct_correlations():
    c=correlations();edges=np.linspace(0,12,241)
    out=c.oz_from_rdf(edges,np.ones(240),.02)
    np.testing.assert_allclose(out['c2_r'],0,atol=1e-13)
    np.testing.assert_allclose(out['c2_hat'],0,atol=1e-13)
    np.testing.assert_allclose(out['S_k'],1,atol=1e-13)
    assert out['c2_hat_zero']==pytest.approx(0.)


def test_radial_transform_gaussian_and_rho_zero_roundtrip():
    c=correlations();edges=np.linspace(0,12,1201);r=(edges[:-1]+edges[1:])/2
    amplitude=-.2;sigma=1.4;h=amplitude*np.exp(-(r/sigma)**2)
    out=c.oz_from_rdf(edges,1+h,0.)
    expected=amplitude*np.pi**1.5*sigma**3*np.exp(-(out['k']*sigma)**2/4)
    np.testing.assert_allclose(out['h_hat'][:80],expected[:80],atol=2e-9,rtol=1e-8)
    np.testing.assert_allclose(out['c2_r'],h,atol=1e-13)


def test_oz_algebra_and_sign():
    c=correlations();edges=np.linspace(0,12,601);r=(edges[:-1]+edges[1:])/2
    out=c.oz_from_rdf(edges,1-.2*np.exp(-(r/1.4)**2),.02)
    np.testing.assert_allclose(out['c2_hat']*out['S_k'],out['h_hat'],atol=1e-12)
    assert out['c2_hat_zero']<0
    assert out['c2_r'][0]<0


def test_nonphysical_spectrum_is_reported_not_clipped():
    c=correlations();edges=np.linspace(0,10,101)
    out=c.oz_from_rdf(edges,np.zeros(100),.1)
    assert np.any(~out['valid_k'])
    assert np.isnan(out['c2_r']).all()
    assert out['status']=='invalid_positive_k_spectrum'


def test_planar_projection_of_constant_radial_kernel():
    c=correlations();edges=np.linspace(0,5,51);z=np.array([0.,1.,4.,5.,6.])
    out=c.planar_projection(edges,np.full(50,-2.),z)
    np.testing.assert_allclose(out,-2*np.pi*np.maximum(0,25-z*z),atol=1e-12)


def test_c1_pnas_needs_mu_to_fix_offset():
    c=correlations();rho=np.array([.01,.02,0.]);v=np.array([.2,-.1,0.])
    unknown=c.c1_equilibrium(rho,v,reference_volume=1.)
    assert 'c1' not in unknown
    assert unknown['status']=='requires_beta_mu'
    known=c.c1_equilibrium(rho,v,beta_mu=-3.,reference_volume=1.)
    np.testing.assert_allclose(known['c1'][:2],np.log(rho[:2])+v[:2]+3)
    assert np.isnan(known['c1'][2])


def test_bulk_c1_density_integral_keeps_reference_constant_explicit():
    c=correlations();rho=np.array([.01,.02,.05])
    out=c.integrate_bulk_c1(rho,np.full(3,-4.))
    np.testing.assert_allclose(out['delta_c1'],-4*(rho-rho[0]))
    assert 'c1' not in out
    anchored=c.integrate_bulk_c1(rho,np.full(3,-4.),anchor_index=1,anchor_c1=-.1)
    np.testing.assert_allclose(anchored['c1'],-.1-4*(rho-rho[1]))


def test_oz_rejects_nonuniform_grid_and_negative_g():
    c=correlations()
    with pytest.raises(ValueError):c.oz_from_rdf(np.array([0.,1.,3.]),np.ones(2),.01)
    with pytest.raises(ValueError):c.oz_from_rdf(np.array([0.,1.,2.]),np.array([-1.,1.]),.01)


def test_unknown_mu_workflow_writes_c2_and_relative_c1_only(tmp_path):
    import json,subprocess,sys,h5py
    c=correlations();states=[]
    from types import SimpleNamespace
    edges=np.linspace(0,12,241);r=(edges[:-1]+edges[1:])/2
    for index,rho in enumerate([.01,.02]):
        state=tmp_path/f'state_{index}';state.mkdir()
        (state/'report.json').write_text(json.dumps({'number_density_A_minus3':rho,'temperature_K':986.4}))
        with h5py.File(state/'distributions.h5','w') as f:
            f['rdf/OO/edges_0']=edges;f['rdf/OO/g']=1-.2*np.exp(-(r/1.4)**2)
        summary=c.write_correlations(state,state/'direct_correlations.h5')
        assert summary['c1_status']=='requires_beta_mu'
        with h5py.File(state/'direct_correlations.h5') as f:assert 'c1/c1' not in f
        standalone=tmp_path/f'correlations_{index}'
        standalone_report=c.run_correlations(SimpleNamespace(results=str(state),output=str(standalone),oz_window='none',oz_tail_start=.8,oz_min_s=1e-8,
                                                             beta_mu=None,beta_mu_ex=None,reference_volume=1.))
        assert standalone_report.get('temperature_K')==986.4
        states.append(standalone)
    output=tmp_path/'scan'
    result=subprocess.run([sys.executable,'-m','waterpost','c1-scan','--results',*[str(s) for s in states],
                           '--output',str(output),'--model-id','synthetic_same_model','--assume-single-component'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with h5py.File(output/'bulk_c1.h5') as f:
        assert 'c1' not in f
        assert f['delta_c1'][0]==pytest.approx(0)
        assert f['delta_c1'][1]<0
    for state in states:
        p=state/'report.json';report=json.loads(p.read_text());report['temperature_K']=None;report['input']=None;p.write_text(json.dumps(report))
    declared=tmp_path/'declared_temperature'
    result=subprocess.run([sys.executable,'-m','waterpost','c1-scan','--results',*[str(s) for s in states],
                           '--output',str(declared),'--model-id','synthetic_same_model','--assume-single-component','--temperature-K','986.4'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_short_terminal_block_not_used_as_equal_length_error_block(tmp_path):
    import json,h5py
    c=correlations();edges=np.linspace(0,8,81)
    (tmp_path/'report.json').write_text(json.dumps({'number_density_A_minus3':.02,'frames_analyzed':201,'block_frames':100}))
    with h5py.File(tmp_path/'distributions.h5','w') as f:
        f['rdf/OO/edges_0']=edges;f['rdf/OO/g']=np.ones(80)
        f['rdf/OO/block_g']=np.ones((3,80));f['rdf/OO/block_ids']=np.array([0,1,2])
    c.write_correlations(tmp_path,tmp_path/'direct_correlations.h5')
    with h5py.File(tmp_path/'direct_correlations.h5') as f:
        assert 'blocks/stderr_eligible' in f
        np.testing.assert_array_equal(f['blocks/stderr_eligible'][...],[True,True,False])
        assert f['blocks/c2_hat_zero_valid_blocks'][()]==2
