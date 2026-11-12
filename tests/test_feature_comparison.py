import json
import subprocess
import sys
import numpy as np
import h5py


def test_same_dimension_different_radial_basis_is_not_silently_compared(tmp_path):
    paths=[]
    for i,cutoff in enumerate([6.,8.]):
        p=tmp_path/f'state{i}';p.mkdir();paths.append(p)
        (p/'report.json').write_text(json.dumps({'frames_analyzed':1,'density_g_cm3':.3,'input':'input.dump','configuration':{},'species':'CH4'}))
        with h5py.File(p/'collective.h5','w') as f:f['feature_statistics/mean']=np.array([1.,2.])
        with h5py.File(p/'environment.h5','w') as f:
            f.attrs['feature_basis']=json.dumps({'cutoff_A':cutoff,'nrad':3,'lmax':3})
            f['feature_names']=np.array(['a','b'],dtype=h5py.string_dtype())
    out=tmp_path/'comparison.json'
    r=subprocess.run([sys.executable,'-m','waterpost','compare','--results',*[str(p) for p in paths],'--output',str(out)],capture_output=True,text=True)
    assert r.returncode!=0
    assert 'basis' in r.stderr.lower()
    assert not out.exists()
