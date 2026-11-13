import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from waterpost.collection import discover,choose_window
from test_pipeline import write_fixture


def test_collection_discovery_ignores_incomplete_downloads_and_keeps_states(tmp_path):
    for name in ['H2O-T986.4-r00-rho0.6700.dump.gz','NH3-T618.2-r01-rho0.5978.dump.gz','CH4-T290.5-r02-rho0.5256.dump.gz.part']:
        (tmp_path/name).write_bytes(b'')
    rows=discover(tmp_path)
    assert len(rows)==2
    assert {r['species'] for r in rows}=={'H2O','NH3'}
    assert rows[0]['temperature_K']==pytest.approx(986.4)


def test_weakly_structured_state_uses_explicit_species_reference_window(tmp_path):
    edges=np.linspace(0,12,241);r=(edges[:-1]+edges[1:])/2
    local=tmp_path/'local';ref=tmp_path/'reference';local.mkdir();ref.mkdir()
    for directory,g in [(local,np.ones(240)),(ref,1+.8*np.exp(-((r-3)/.2)**2)-.2*np.exp(-((r-4)/.2)**2))]:
        with h5py.File(directory/'distributions.h5','w') as f:f['rdf/centers/edges_0']=edges;f['rdf/centers/g']=g
    result=choose_window(local,ref,8.5)
    assert result['source']=='species_temperature_reference_RDF'
    assert 3.9<result['first_cut_A']<4.1
    assert 'No resolved' in result['local_reason']


def test_collection_rdf_stage_and_safe_resume(tmp_path):
    import gzip
    data=tmp_path/'data';data.mkdir();raw=tmp_path/'tiny.dump';write_fixture(raw)
    source=data/'H2O-T986.4-r03-rho0.4533.dump.gz'
    with gzip.open(source,'wb') as f:f.write(raw.read_bytes())
    out=tmp_path/'result';script=Path(__file__).resolve().parents[1]/'tools/run_collection.py'
    command=[sys.executable,str(script),'--input-dir',str(data),'--output-root',str(out),'--workers','2','--phase','rdf']
    # Fixture box12 needs an explicit larger box because production RDF radius12.
    # Scale centers apart by replacing only the box while retaining the molecules.
    text=raw.read_text().replace('0 12','0 30')
    with gzip.open(source,'wb') as f:f.write(text.encode())
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads((out/'collection_summary.json').read_text())
    assert report['summary']=={'rdf:completed':1}
    case=out/'rdf/H2O/H2O-T986.4-r03-rho0.4533'
    initial=(case/'report.json').stat().st_mtime_ns
    again=subprocess.run(command+['--resume'],capture_output=True,text=True)
    assert again.returncode==0,again.stdout+again.stderr
    assert (case/'report.json').stat().st_mtime_ns==initial
    assert json.loads((out/'progress.json').read_text())['stages']['rdf:H2O-T986.4-r03-rho0.4533']['reused']
