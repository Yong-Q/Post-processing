import json
import subprocess
import sys
import numpy as np
import h5py
import pytest
from test_molecules import configurations,frame
from waterpost.molecules import MolecularTopology,SPECS
from waterpost.molecular_io import CenterTracker


@pytest.mark.parametrize('species',list(SPECS))
def test_fast_center_tracker_matches_complete_topology(species):
    elements,p=configurations()[species];p=np.asarray(p)+[11.7,3,4]
    cell=np.array([[12.,0,0],[4.,11,0],[2.,1,10]])
    f=frame(elements,p,cell=cell)
    a=MolecularTopology(f,species).extract(f);centers,valid=CenterTracker(f,species).extract(f)
    np.testing.assert_allclose(centers,a['centers'],atol=1e-12)
    np.testing.assert_array_equal(valid,a['heavy_valid'])


def write_methane(path,native=False):
    elements,relative=configurations()['CH4'];relative=np.asarray(relative);lines=[]
    for t in range(4):
        lines+=['ITEM: TIMESTEP',str(t*1000),'ITEM: NUMBER OF ATOMS','10','ITEM: BOX BOUNDS pp pp pp','0 20','0 20','0 20',
                'ITEM: ATOMS id mol type '+('xu yu zu' if native else 'x y z')+' fx fy fz']
        for mol,center in enumerate([[19.7+t*.1,1,1],[2.8+t*.1,1,1]]):
            for a,(e,v) in enumerate(zip(elements,relative)):
                p=np.asarray(center)+v
                p=p+np.array([20*t,0,0]) if native else p%20
                typ=1 if e=='C' else 2
                lines.append(f'{mol*5+a+1} {mol+1} {typ} '+' '.join(map(str,p))+' 0 0 0')
    path.write_text('\n'.join(lines)+'\n')


def test_methane_generic_end_to_end(tmp_path):
    source=tmp_path/'CH4-T290.5-example.dump';write_methane(source);out=tmp_path/'out'
    result=subprocess.run([sys.executable,'-m','waterpost','analyze','--species','CH4','--input',str(source),'--output',str(out),
                           '--nrad','3','--rdf-max','5','--env-cut','5','--max-lag','2','--block-frames','2','--no-voronoi','--expensive-stride','1'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    r=json.loads((out/'report.json').read_text());assert r['species']=='CH4'
    assert r['molar_mass_g_mol']==pytest.approx(16.043)
    assert r['center_definition']=='heavy_skeleton_com'
    assert not r['applicability']['hydrogen_bonds']
    with h5py.File(out/'environment.h5') as f:
        assert f['features'].shape==(4,2,508)
        assert np.isnan(f['axis'][...]).all()
        np.testing.assert_allclose(f['centers'][0,0],[19.7,1,1],atol=1e-5)
    with h5py.File(out/'direct_correlations.h5') as f:assert 'CH4' in f['c2'].attrs['scope']


def test_ethane_rdf_counts_one_center_per_molecule(tmp_path):
    elements,coords=configurations()['C2H6'];coords=np.asarray(coords)+[5,5,5]
    source=tmp_path/'C2H6-T465.4-example.dump';lines=[]
    for t in range(2):
        lines+=['ITEM: TIMESTEP',str(t*1000),'ITEM: NUMBER OF ATOMS','16','ITEM: BOX BOUNDS pp pp pp','0 30','0 30','0 30',
                'ITEM: ATOMS id mol type x y z']
        for mol,shift in enumerate([[0,0,0],[4,0,0]]):
            for a,(e,p) in enumerate(zip(elements,coords+shift)):
                lines.append(f'{mol*8+a+1} {mol+1} {1 if e=="C" else 2} '+' '.join(map(str,p)))
    source.write_text('\n'.join(lines)+'\n');out=tmp_path/'rdf'
    result=subprocess.run([sys.executable,'-m','waterpost','rdf-only','--species','C2H6','--input',str(source),'--output',str(out),'--rdf-max','8','--bins','80'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((out/'report.json').read_text())
    assert report['number_of_centers']==2
    assert report['number_density_A_minus3']==pytest.approx(2/30**3)
    assert report['molar_mass_g_mol']==pytest.approx(30.07)


def test_native_unwrapped_centers_are_not_discarded(tmp_path):
    source=tmp_path/'CH4-T290.5-native.dump';write_methane(source,native=True);out=tmp_path/'out'
    result=subprocess.run([sys.executable,'-m','waterpost','analyze','--species','CH4','--input',str(source),'--output',str(out),
                           '--nrad','1','--rdf-max','5','--env-cut','5','--max-lag','1','--no-voronoi','--no-equivariant'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with h5py.File(out/'environment.h5') as f:assert 'native_unwrapped_centers' in f
    with h5py.File(out/'dynamics.h5') as f:
        assert f['translation/msd'][1]==pytest.approx(20.1**2,rel=1e-5)


def test_zero_lag_setting_really_has_no_positive_lags():
    from waterpost.dynamics import lag_grid
    np.testing.assert_array_equal(lag_grid(10,0),[0])
