import subprocess
import sys
import numpy as np
import h5py
from waterpost.collective import cumulants


def write_fixture(path):
    # Six rigid water molecules; one O and its Hs cross the periodic boundary.
    oxy=np.array([[9.8,1,1],[2.6,1,1],[1,4,1],[1,1,4],[6,6,6],[7,2,7.]])
    lines=[]
    for frame in range(4):
        lines+=['ITEM: TIMESTEP',str(frame*500),'ITEM: NUMBER OF ATOMS','18',
                'ITEM: BOX BOUNDS pp pp pp','0 12','0 12','0 12',
                'ITEM: ATOMS id mol type x y z fx fy fz']
        for i,o in enumerate(oxy+np.array([frame*.1,0,0])):
            for h,(offset,typ) in enumerate([(np.zeros(3),4),(np.array([.96,0,0]),2),(np.array([-.24,.93,0]),2)]):
                x=(o+offset)%12
                lines.append(f'{i*3+h+1} {i+1} {typ} '+' '.join(map(str,x))+' 0 0 0')
    path.write_text('\n'.join(lines)+'\n')


def test_cumulants_known_distribution():
    c=cumulants(np.array([[0.,1,2]]))
    np.testing.assert_allclose(c['mean'],[1])
    np.testing.assert_allclose(c['variance'],[2/3])
    np.testing.assert_allclose(c['third_cumulant'],[0],atol=1e-14)
    np.testing.assert_allclose(c['fourth_cumulant'],[-2/3])


def test_end_to_end_outputs_and_force_units(tmp_path):
    import json
    p=tmp_path/'water.dump';write_fixture(p);out=tmp_path/'results'
    r=subprocess.run([sys.executable,'-m','waterpost','analyze','--input',str(p),'--output',str(out),
                      '--frame-dt-ps','.5','--env-cut','5','--rdf-max','5','--blocks','2',
                      '--expensive-stride','1','--max-lag','2'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    with h5py.File(out/'environment.h5') as f:
        assert f['o'].shape==(4,6,3)
        np.testing.assert_allclose(np.linalg.norm(f['oh'][0],axis=-1)[:,0],.96,atol=1e-5)
        assert f['forces'].attrs['units']=='native_unknown'
        assert f['water_features'].shape==(4,6,176)
        assert f['water_orientation_weighted_means'].shape==(4,6,7,4)
    with h5py.File(out/'distributions.h5') as f:
        assert f['rdf/OO/counts'][...].sum()>0
        assert np.isclose(f['local/q/probability'][...].sum(),1)
    report=json.loads((out/'report.json').read_text())
    assert report['frames_analyzed']==4
    assert report['status']=='completed'
    assert len(report['coverage'])==86
    assert report['periodicity']['all_frames_periodic']
