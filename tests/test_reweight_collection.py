import json
import numpy as np
import pytest
from waterpost.reweight_collection import (read_energy, profile_transform,
    rdf_frames, classify_observable, select_window)


def test_energy_format_and_npz_identity(tmp_path):
    p=tmp_path/'x.energy.csv';p.write_text('step,pe_eV,press_bar\n0,-1234567.123456789,2\n1000,-1234566.987654321,3\n')
    z=tmp_path/'x.npz'
    np.savez(z,step=np.array([0,1000]),pe=np.array([-1234567.123456789,-1234566.987654321]),press=[2.,3.],meta=json.dumps({'temperature_K':300,'model_sha256':'abc'}))
    data=read_energy(p,z,np.array([0,1000]));assert data['meta']['model_sha256']=='abc'
    with pytest.raises(ValueError):read_energy(p,z,np.array([0,2000]))


def test_profile_relative_c1_and_empty_bin():
    f=profile_transform(2,10.,np.array([.1,-.1]))
    value=f(np.array([2.,2.]));np.testing.assert_allclose(value,[.1,-.1])
    with pytest.raises(ValueError):f(np.array([0.,4.]))


def test_rdf_periodicity_exact_two_centers():
    cell=np.diag([10.,10.,10.]);centers=np.array([[[.1,0,0],[9.9,0,0]]])
    edges=np.array([0.,.5,1.]);g=rdf_frames(centers,cell,edges)
    shell=4*np.pi/3*.5**3
    np.testing.assert_allclose(g[0],[1000/shell,0])


def test_noise_not_physical_shift_is_acceptance_criterion():
    from waterpost.reweight import block_estimate
    x=np.arange(40.).reshape(20,2);w=np.ones(20)/20
    base=block_estimate(x,w,2)
    # deterministic constant changes can exceed sampling noise; zero added noise.
    target=(base[0]+100,base[1],base[2]+100)
    result=classify_observable(base,target)
    assert result['status']=='diagnostics_passed'


def test_label_temporal_drift_rejected_even_for_equal_weights(tmp_path):
    import h5py
    from waterpost.reweight_collection import _label_estimate
    x=np.r_[np.zeros(1000),np.ones(1000)*100][:,None]
    with h5py.File(tmp_path/'x.h5','w') as h:
        _,info=_label_estimate(h,x,np.ones(2000),10,None)
    assert info['status']=='unqualified'
    assert info['reason']=='label_temporal_window_nonstationary'


def test_short_window_never_claims_sufficient_blocks():
    x=np.arange(20.)
    result=select_window(x,np.column_stack([x,x]),burn_fraction=.2)
    assert result['status']=='unqualified'


def test_pilot_calibration_can_be_checked_without_reusing_holdout():
    from waterpost.reweight_collection import _screen_logq
    x=np.tile(np.arange(200.)[:,None],(1,2))
    logq=np.zeros(200);logq[100:]=np.arange(100.)
    calibration=_screen_logq(logq,x,10,partitions='calibration')
    assert calibration['pass']
    full=_screen_logq(logq,x,10)
    assert not full['pass']


@pytest.mark.parametrize('empty_bins',[False,True])
def test_generation_preserves_provenance_weights_and_separate_c2_scopes(tmp_path,empty_bins):
    import h5py
    from waterpost.reweight_collection import generate_case
    rng=np.random.default_rng(32);n=800;nmol=100;d=tmp_path/'synthetic';d.mkdir()
    counts=np.asarray([[rng.multinomial(nmol,np.ones(24)/24) for axis in range(3)] for frame in range(n)])
    if empty_bins:
        counts[:]=0;counts[:,:,0]=nmol
    fields=np.zeros((n,15));fields[:,0]=counts[:,0]@np.cos(2*np.pi*(np.arange(24)+.5)/24)
    numerator=1+rng.normal(0,.01,(n,508));denominator=np.ones_like(numerator)
    rho_modes=(rng.normal(size=(n,3))+1j*rng.normal(size=(n,3)))*np.sqrt(nmol/2)
    np.savez_compressed(d/'frames.npz',step=np.arange(n)*1000,energy=-100+rng.normal(size=n),
        numerator=numerator,denominator=denominator,watch=rng.normal(size=(n,5)),
        rdf=1+rng.normal(0,.001,(n,24)),rdf_edges=np.linspace(0,8,25),counts=counts,fields=fields,
        rho_modes=rho_modes,k_vectors=np.eye(3)*.5)
    window={'status':'diagnostics_passed','start':0,'stop':n,'block_frames':50,'blocks':16,'statistical_inefficiency_max':1.}
    provenance={'window':window,'source_temperature_K':300.,'number_of_centers':nmol,'volume_A3':10000.,
        'number_density_A_minus3':.01,'feature_names':[str(i) for i in range(508)],'split_group':'same-source-hash'}
    (d/'provenance.json').write_text(json.dumps(provenance))
    candidates=[{'kind':'field','field_index':0,'target_temperature_K':300.,'amplitude_beta0':.001},
        {'kind':'temperature','field_index':None,'target_temperature_K':300.01,'amplitude_beta0':0.}]
    (d/'perturbation_ranges.json').write_text(json.dumps({'candidates_for_label_qualification':candidates}))
    result=generate_case({'case':'synthetic','output':str(tmp_path)})
    assert result['status']=='generated' and result['states']==3
    manifest=json.loads((d/'manifest.json').read_text())['states']
    assert all(x['split_group']=='same-source-hash' for x in manifest)
    assert manifest[0]['origin']=='direct_estimate'
    with h5py.File(d/'states.h5') as h:
        for name in ('0000','0001','0002'):
            assert h[name+'/frame_weights'][:].sum()==pytest.approx(1.)
        if not empty_bins:assert 'profile_12/canonical_projected_c2' in h['0001']
        assert 'bulk_c2' not in h['0001']
        assert not h['0002/bulk_c2'].attrs['zero_mode_training_eligible']
        assert 'local_observables' not in manifest[1]['errors']
        if empty_bins:
            assert 'c1_centered_12' in manifest[1]['errors']
            assert 'canonical_projected_c2_12' in manifest[1]['errors']


def test_prepare_pilot_generate_short_source_remains_unqualified(tmp_path):
    import h5py
    from waterpost.reweight_collection import prepare_case,pilot_case,generate_case
    n=40;nmol=2;case='tiny';source=tmp_path/'source';source.mkdir();energy=tmp_path/'energy';energy.mkdir();(energy/'npz').mkdir()
    output=tmp_path/'output';output.mkdir();rng=np.random.default_rng(3)
    step=np.arange(n)*1000;u=-100+rng.normal(size=n);press=np.zeros(n)
    with h5py.File(source/'environment.h5','w') as h:
        for key,value in {'step':step,'cell':np.tile(np.eye(3)*10,(n,1,1)),
            'centers':rng.uniform(0,10,(n,nmol,3)),'features':np.zeros((n,nmol,508)),
            'axis':np.ones((n,nmol,3))/np.sqrt(3),'molecule_valid':np.ones((n,nmol),bool)}.items():h[key]=value
        h.create_dataset('feature_names',data=np.asarray([str(i) for i in range(508)],dtype=h5py.string_dtype()))
        h.attrs['axis_kind']='polar';h.attrs['feature_basis']='{"nrad":3,"lmax":3}'
    with h5py.File(source/'collective.h5','w') as h:
        h['rho_centers']=rng.normal(size=(n,3))+1j*rng.normal(size=(n,3))
        h['k_vectors']=np.eye(3)[None];h['integer_modes']=np.eye(3,dtype=int)
    report={'frames_analyzed':n,'species':'H2O','temperature_K':300.,'input':'synthetic.dump',
        'input_sha256':'synthetic-source-hash','center_definition':'heavy_skeleton_com'}
    (source/'report.json').write_text(json.dumps(report))
    np.savetxt(energy/(case+'.energy.csv'),np.column_stack([step,u,press]),delimiter=',',header='step,pe_eV,press_bar',comments='',fmt='%.16g')
    # Parse exactly the serialized decimal data for CSV/NPZ identity.
    serialized=np.loadtxt(energy/(case+'.energy.csv'),delimiter=',',skiprows=1)
    np.savez(energy/'npz'/(case+'.npz'),step=step,pe=serialized[:,1],press=press,
        meta=json.dumps({'species':'H2O','temperature_K':300.,'density_mol_A3':.002,'model_sha256':'synthetic-model'}))
    task={'case':case,'source':str(source),'output':str(output),'energy_dir':str(energy),'energy_commit':'synthetic'}
    assert prepare_case(task)['status']=='prepared'
    assert pilot_case(task)['candidate_count']==0
    assert generate_case(task)['status']=='source_unqualified'
