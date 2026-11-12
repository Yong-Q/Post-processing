"""Read every frame; audit original mol IDs independently of feature analysis."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from waterpost.io import read_dump,WaterTopology
from waterpost.geometry import mic,unique_radius


def audit(path):
    result={'input':str(Path(path).resolve()),'frames':0,'bad_original_mol_frames':0,'first_original_anomalies':[],
            'nonwater_O_frames':0,'unassigned_H_frames':0,'max_original_OH_A':0.,'all_periodic':True}
    top=None;previous=None;intervals=set();start=time.monotonic();counts=[];cells=[]
    for index,f in enumerate(read_dump(path)):
        if top is None:top=WaterTopology(f,4,2)
        if not np.array_equal(f.field('id'),top.ids):raise ValueError('Changing IDs')
        positions=f.positions;original=positions[top.indices]
        lengths=np.linalg.norm(mic(original[:,1:]-original[:,:1],f.cell),axis=-1)
        bad=int(np.sum((lengths<.5)|(lengths>1.6)))
        result['max_original_OH_A']=max(result['max_original_OH_A'],float(lengths.max()))
        if bad:
            result['bad_original_mol_frames']+=1
            if len(result['first_original_anomalies'])<8:result['first_original_anomalies'].append({'frame':index,'step':f.step,'bad_bonds':bad,'max_A':float(lengths.max())})
        o=positions[top.indices[:,0]];hp=positions[top.h_indices]
        if np.allclose(f.cell,np.diag(np.diag(f.cell))):
            box=np.diag(f.cell);tree=cKDTree(o%box,boxsize=box);distance,owner=tree.query(hp%box,k=1)
        else:
            distances=np.linalg.norm(mic(hp[None,:,:]-o[:,None,:],f.cell),axis=-1)
            owner=np.argmin(distances,axis=0);distance=distances[owner,np.arange(len(hp))]
        owner[distance>1.35]=-1
        nh=np.bincount(owner[owner>=0],minlength=len(o))
        result['nonwater_O_frames']+=int((nh!=2).sum());result['unassigned_H_frames']+=int((owner<0).sum())
        counts.append(np.bincount(np.minimum(nh,5),minlength=6))
        cells.append(f.cell)
        if previous is not None:intervals.add(f.step-previous)
        previous=f.step;result['frames']+=1
        if (index+1)%500==0:print(Path(path).name,'audited frames',index+1,flush=True)
    result['molecules_by_original_mol']=len(top.mol_ids);result['step_intervals']=sorted(intervals)
    result['assigned_H_count_histogram_0_1_2_3_4_5plus']=np.sum(counts,axis=0).tolist()
    result['cell_constant']=bool(np.allclose(cells,cells[0]));result['minimum_unique_radius_A']=min(unique_radius(c) for c in cells)
    result['elapsed_s']=time.monotonic()-start
    result['topology_definition']='Nearest O within 1.35 Angstrom; geometric audit only, not an electronic reaction assignment'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise SystemExit('Refusing existing output')
    data=[audit(path) for path in a.inputs]
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
