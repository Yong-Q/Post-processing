"""Metamorphic periodicity checks on an actual selected input frame."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from waterpost.io import read_dump,Frame,WaterTopology
from waterpost.geometry import mic,force_torque
from waterpost.static import local_metrics,hydrogen_bonds,periodic_voronoi
from waterpost.features import encode


def audit(path,index=200):
    rng=np.random.default_rng(812)
    for i,frame in enumerate(read_dump(path)):
        if i==index:break
    else:raise ValueError('Requested frame missing')
    top=WaterTopology(frame,4,2,mode='nearest');a=top.extract(frame)
    data=frame.data.copy()
    shifts=rng.integers(-2,3,(len(data),3))@frame.cell
    for axis,name in enumerate(['x','y','z']):data[:,frame.columns.index(name)]+=shifts[:,axis]
    translated=Frame(frame.step,frame.cell,frame.origin,frame.columns,data)
    b=top.extract(translated)
    checks={}
    for key in ['water_valid','assigned_h_count','h_owner','atom_ids']:
        checks[key+'_unchanged']=bool(np.array_equal(a[key],b[key]))
    for key in ['oh']:
        checks[key+'_max_abs_error']=float(np.nanmax(abs(a[key]-b[key])))
    ma=local_metrics(a['o'],a['oh'],frame.cell);mb=local_metrics(b['o'],b['oh'],frame.cell)
    for key in ['q','nearest','distance','lsi']:
        finite=np.isfinite(ma[key])&np.isfinite(mb[key])
        checks[key+'_max_abs_error']=float(np.max(abs(ma[key][finite]-mb[key][finite]))) if finite.any() else 0.
    checks['hbond_triplets_unchanged']=bool(np.array_equal(hydrogen_bonds(a['o'],a['oh'],frame.cell),hydrogen_bonds(b['o'],b['oh'],frame.cell)))
    fa=encode(a['o'],a['oh'],frame.cell,raw_h=a['raw_h_positions'],h_owner=a['h_owner'])['invariants']
    fb=encode(b['o'],b['oh'],frame.cell,raw_h=b['raw_h_positions'],h_owner=b['h_owner'])['invariants']
    checks['feature_max_abs_error']=float(np.nanmax(abs(fa-fb)))
    vora=periodic_voronoi(a['o'],frame.cell);vorb=periodic_voronoi(b['o'],frame.cell)
    checks['voronoi_volume_sum_error_A3']=float(abs(vora['volume'].sum()-np.linalg.det(frame.cell)))
    checks['voronoi_relative_image_error']=float(np.max(abs(vora['volume']-vorb['volume'])/vora['volume']))
    ia=rng.integers(0,len(a['o']),30);ib=rng.integers(0,len(a['o']),30)
    vectors=a['o'][ib]-a['o'][ia]
    grid=np.array(list(itertools.product(range(-2,3),repeat=3)))@frame.cell
    brute=np.min(np.linalg.norm(vectors[:,None,:]-grid[None,:,:],axis=-1),axis=1)
    checks['mic_vs_brute_max_error_A']=float(np.max(abs(np.linalg.norm(mic(vectors,frame.cell),axis=-1)-brute)))
    for key,value in checks.items():
        if key.endswith('unchanged') and not value:raise AssertionError(key)
        if key.endswith('max_abs_error') and value>1e-7:raise AssertionError((key,value))
    if checks['voronoi_relative_image_error']>2e-4:raise AssertionError('Voronoi translation discrepancy')
    return {'input':str(Path(path).resolve()),'frame':index,'step':frame.step,'status':'pass','checks':checks,
            'scope':'Actual-frame whole-lattice shifts per atom; brute lattice distance check; not a proof for every input or unresolved temporal jumps'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise SystemExit('Refusing existing output')
    result=[audit(path) for path in a.inputs];out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
