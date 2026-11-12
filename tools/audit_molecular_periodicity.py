"""Actual-frame invariance under atom images and equivalent-atom relabeling."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from waterpost.io import read_dump,Frame
from waterpost.molecules import MolecularTopology
from waterpost.molecular_io import infer_species
from waterpost.molecular_features import encode_molecules
from waterpost.geometry import mic
from waterpost.static import periodic_voronoi


def audit(path,index):
    rng=np.random.default_rng(20261112)
    for i,frame in enumerate(read_dump(path)):
        if i==index:break
    else:raise ValueError('Requested frame unavailable')
    species=infer_species(path);top=MolecularTopology(frame,species);base=top.extract(frame)
    xyz=[frame.columns.index(x) for x in ['x','y','z']]
    shifted=frame.data.copy();shifted[:,xyz]+=rng.integers(-2,3,(len(shifted),3))@frame.cell
    images=Frame(frame.step,frame.cell,frame.origin,frame.columns,shifted)
    moved=top.extract(images)
    a=encode_molecules(base['centers'],base['positions'],base['elements'],base['owner'],frame.cell,base['molecule_valid'],cutoff=8.)
    b=encode_molecules(moved['centers'],moved['positions'],moved['elements'],moved['owner'],frame.cell,moved['molecule_valid'],cutoff=8.)
    checks={'validity_images_equal':bool(np.array_equal(base['molecule_valid'],moved['molecule_valid'])),
            'owner_images_equal':bool(np.array_equal(base['owner'],moved['owner'])),
            'center_image_error_A':float(abs(mic(base['centers']-moved['centers'],frame.cell)).max()),
            'feature_image_error':float(np.nanmax(abs(a['features']-b['features'])))}
    permutation=np.arange(len(frame.data))
    # H labels can permute globally in a reactive representation. Heavy equal
    # atoms permute only inside a skeleton so molecular centers remain defined.
    h=np.flatnonzero(top.element_codes==1);permutation[h]=rng.permutation(h)
    for owner in range(len(top.molecule_ids)):
        for element in [0,2,3]:
            rows=np.flatnonzero((top.initial_owner==owner)&(top.element_codes==element));permutation[rows]=rng.permutation(rows)
    data=frame.data.copy();data[:,xyz]=frame.data[permutation][:,xyz]
    force=[frame.columns.index(k) for k in ['fx','fy','fz']] if all(k in frame.columns for k in ['fx','fy','fz']) else []
    if force:data[:,force]=frame.data[permutation][:,force]
    permuted=top.extract(Frame(frame.step,frame.cell,frame.origin,frame.columns,data))
    c=encode_molecules(permuted['centers'],permuted['positions'],permuted['elements'],permuted['owner'],frame.cell,permuted['molecule_valid'],cutoff=8.)
    checks.update(validity_permutation_equal=bool(np.array_equal(base['molecule_valid'],permuted['molecule_valid'])),
                  feature_permutation_error=float(np.nanmax(abs(a['features']-c['features']))),
                  center_permutation_error_A=float(abs(mic(base['centers']-permuted['centers'],frame.cell)).max()))
    if force:checks['molecular_force_permutation_error_native']=float(np.nanmax(abs(base['molecular_force']-permuted['molecular_force'])))
    vor=periodic_voronoi(base['centers'],frame.cell);checks['voronoi_relative_volume_sum_error']=float(abs(vor['volume'].sum()/np.linalg.det(frame.cell)-1))
    for key,value in checks.items():
        if key.endswith('_equal') and not value:raise AssertionError(key)
        if ('feature_' in key or 'center_' in key) and 'error' in key and value>1e-7:raise AssertionError((key,value))
    if checks['voronoi_relative_volume_sum_error']>3e-5:raise AssertionError('Periodic volumes do not tile box')
    return {'species':species,'input_file':Path(path).name,'source_frame':index,'status':'pass','molecules':len(base['centers']),
            'invalid_target_molecules':int((~base['molecule_valid']).sum()),'feature_dimension':len(a['names']),'checks':checks,
            'scope':'Actual-frame atom image and equivalent-element relabeling, not a physical model validation or all-frame proof'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--frame',type=int,default=200);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise SystemExit('Output exists; refusing overwrite')
    results=[audit(path,a.frame) for path in a.inputs]
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
