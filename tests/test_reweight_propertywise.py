import numpy as np
from waterpost.reweight_propertywise import direction_window, direction_monitor


def test_direction_window_does_not_use_unrelated_slow_feature():
    rng=np.random.default_rng(9);n=2001
    data={'energy':rng.normal(size=n),'fields':rng.normal(size=(n,15)),
          'watch':np.arange(n)[:,None]}
    window=direction_window(data,None)
    assert window['status']=='diagnostics_passed'
    assert window['blocks']>=8
    assert window['start']==0
    np.testing.assert_array_equal(direction_monitor(data,None),data['energy'][:,None])
    np.testing.assert_array_equal(direction_monitor(data,3),data['fields'][:,3:4])
    assert direction_monitor(data,3,joint=True).shape==(n,2)


def test_only_slow_direction_is_withheld():
    rng=np.random.default_rng(3);n=2001
    fields=rng.normal(size=(n,15));fields[:,0]=np.linspace(-10,10,n)
    data={'energy':rng.normal(size=n),'fields':fields}
    assert direction_window(data,None)['status']=='diagnostics_passed'
    assert direction_window(data,0)['status']=='unqualified'
    assert direction_window(data,1)['status']=='diagnostics_passed'


def test_transformed_influence_detects_slow_cancellation():
    from waterpost.reweight_collection import transform_influence
    from waterpost.reweight import statistical_inefficiency
    rng=np.random.default_rng(10);n=2000
    fast=rng.normal(size=n);slow=np.linspace(0,1,n)*.001
    x=np.column_stack([fast+slow,fast])
    influence=transform_influence(x,np.ones(n)/n,lambda m:np.array([m[0]-m[1]]))
    assert statistical_inefficiency(influence[:,0])>100
    assert statistical_inefficiency(x[:,0])<5


def test_sparse_support_is_local_to_affected_feature():
    from waterpost.reweight_collection import supported_local_columns
    denominator=np.ones((800,3));denominator[:,1]=0
    denominator[0,1]=1;denominator[400,1]=1
    np.testing.assert_array_equal(supported_local_columns(denominator,100),[True,False,True])


def test_cached_source_with_failed_old_global_gate_is_retained(tmp_path,monkeypatch):
    import json,h5py
    import waterpost.reweight_collection as legacy
    from waterpost.reweight_propertywise import prepare,pilot,generate
    rng=np.random.default_rng(21);n=800;case='synthetic';cache=tmp_path/'cache'/case;cache.mkdir(parents=True)
    energy_dir=tmp_path/'energy';energy_dir.mkdir();(energy_dir/'npz').mkdir()
    source=tmp_path/'source';report=source/'analysis/H2O'/case;report.mkdir(parents=True)
    root=tmp_path/'v2';root.mkdir()
    for sub in ['source_admission','ranges','generated','failures']:(root/sub).mkdir()
    step=np.arange(n)*1000;u=-100+rng.normal(size=n)
    csv=energy_dir/(case+'.energy.csv')
    np.savetxt(csv,np.column_stack([step,u,np.zeros(n)]),delimiter=',',comments='',header='step,pe_eV,press_bar',fmt='%.16g')
    u=np.loadtxt(csv,delimiter=',',skiprows=1)[:,1]
    meta={'species':'H2O','model_sha256':'model','temperature_K':300.}
    np.savez(energy_dir/'npz'/(case+'.npz'),step=step,pe=u,press=np.zeros(n),meta=json.dumps(meta))
    fields=rng.normal(size=(n,15));fields[:,0]=np.linspace(-1,1,n)
    counts=np.array([[rng.multinomial(100,np.ones(24)/24) for axis in range(3)] for frame in range(n)])
    numerator=np.column_stack([np.linspace(0,1,n),rng.normal(size=(n,2))])
    np.savez_compressed(cache/'frames.npz',step=step,energy=u,fields=fields,watch=np.linspace(0,1,n)[:,None],
        numerator=numerator,denominator=np.ones_like(numerator),counts=counts,
        rdf=1+rng.normal(0,.001,(n,16)),rdf_edges=np.linspace(0,8,17),
        rho_modes=(rng.normal(size=(n,3))+1j*rng.normal(size=(n,3)))*np.sqrt(50),k_vectors=np.eye(3)*.5)
    old_window={'status':'unqualified','reason':'insufficient_complete_time_blocks','statistical_inefficiency_max':500.}
    p={'case':case,'species':'H2O','source_temperature_K':300.,'number_of_centers':100,'volume_A3':10000.,
       'number_density_A_minus3':.01,'source_trajectory_sha256':'source','source_model_sha256':'model',
       'energy_csv_sha256':legacy.sha256(csv),'feature_names':['slow','fast0','fast1'],'split_group':'source','window':old_window}
    (cache/'provenance.json').write_text(json.dumps(p));(report/'report.json').write_text(json.dumps({'input_sha256':'source'}))
    task={'root':str(root),'case':case,'cache_root':str(cache.parent),'source_root':str(source),'energy_dir':str(energy_dir)}
    prepared=prepare(task)
    assert prepared['source_admitted']
    assert prepared['direction_windows']['temperature']['status']=='diagnostics_passed'
    assert prepared['direction_windows']['field_00']['status']=='unqualified'
    monkeypatch.setattr(legacy,'SCALES',(.025,))
    assert pilot(task)['candidate_count']>0
    # Limit synthetic generation to one temperature and one field point.
    for f in root.glob('groups/*/*/perturbation_ranges.json'):
        d=json.loads(f.read_text());d['candidates_for_label_qualification']=d['candidates_for_label_qualification'][:1] if f.parent.parent.name in ('temperature','field_01') else []
        f.write_text(json.dumps(d))
    result=generate(task);assert result['source_admitted'] and result['states']==3
    with h5py.File(root/'groups/direct'/case/'states.h5') as h:
        mask=h['0000/local_observables/training_eligible_element_mask'][:]
        assert not mask[0] and mask[1:].any()
    manifests=[json.loads(f.read_text()) for f in root.glob('groups/*/*/manifest.json')]
    ids=[s['id'] for m in manifests for s in m['states']]
    assert len(ids)==len(set(ids))
