"""Fast full-trajectory OO RDF without recomputing all local features."""
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import h5py
from scipy.spatial import cKDTree
from .io import read_dump
from .geometry import mic,unique_radius
from .histograms import Histograms
from .correlations import write_correlations


def periodic_oo_counts(o,cell,edges):
    if np.allclose(cell,np.diag(np.diag(cell)),atol=1e-12):
        lengths=np.diag(cell);tree=cKDTree(o%lengths,boxsize=lengths)
        # Includes both directed pairs. Constant self count cancels in diff.
        return np.diff(tree.count_neighbors(tree,edges,cumulative=True)).astype(float)
    d=np.linalg.norm(mic(o[None,:,:]-o[:,None,:],cell),axis=-1);np.fill_diagonal(d,np.inf)
    return np.histogram(d[np.isfinite(d)],bins=edges)[0].astype(float)


def rdf_only(args):
    source=Path(args.input);output=Path(args.output)
    if output.exists():raise ValueError('Output already exists; refusing overwrite')
    if not source.is_file():raise ValueError('Missing input dump')
    if args.start<0 or args.stride<1 or args.block_frames<1 or args.bins<2 or args.rdf_max<=0 or args.frame_dt_ps<=0:
        raise ValueError('Invalid RDF selection/grid arguments')
    if args.max_frames is not None and args.max_frames<1:raise ValueError('max-frames must be positive')
    output.mkdir(parents=True);clock=time.monotonic();hist=Histograms(args.block_frames)
    edges=np.linspace(0,args.rdf_max,args.bins+1);shell=4*np.pi/3*np.diff(edges**3)
    selected=0;volumes=[];steps=[];source_frames=[];ids=None;cells=[];rho_modes=[];k_reference=None
    sha=hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):sha.update(chunk)
    report={'status':'running','mode':'rdf_only','input':str(source.resolve()),'input_sha256':sha.hexdigest(),'configuration':vars(args),
            'scope':'All oxygen centers, not a filtered fluctuating population of instantaneous H2O; no additional MD',
            'warnings':['Bulk/isotropy assumption required. No automatic equilibration removal. Finite-N/cutoff effects in OZ remain.']}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    integer=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]])
    previous_step=None;interval=None
    try:
        for index,f in enumerate(read_dump(source)):
            if previous_step is not None:
                difference=f.step-previous_step
                if difference<=0 or (interval is not None and difference!=interval):raise ValueError('Irregular dump timesteps')
                interval=difference
            previous_step=f.step
            if index<args.start or (index-args.start)%args.stride:continue
            if args.max_frames is not None and selected>=args.max_frames:break
            if args.rdf_max>=unique_radius(f.cell):raise ValueError('RDF radius exceeds conservative unique-image radius')
            mask=f.field('type')==args.oxygen_type;o=f.positions[mask];current=f.field('id')[mask].astype(int)
            if len(o)<2:raise ValueError('Need at least two configured oxygen centers')
            if ids is None:ids=current.copy();columns=f.columns
            elif not np.array_equal(ids,current):raise ValueError('Oxygen population/identity changes across frames')
            n=len(o);volume=float(np.linalg.det(f.cell));counts=periodic_oo_counts(o,f.cell,edges)
            hist.add_binned('rdf/OO',counts,[edges],selected,n*(n-1)/volume*shell,total_samples=n*(n-1))
            k=2*np.pi*integer@np.linalg.inv(f.cell).T
            if k_reference is None:k_reference=k
            elif not np.allclose(k_reference,k,rtol=1e-6,atol=1e-8):raise ValueError('rdf-only currently requires a fixed cell for k-space cross-check')
            rho_modes.append(np.exp(-1j*(o@k.T)).sum(axis=0));volumes.append(volume);cells.append(f.cell);steps.append(f.step);source_frames.append(index)
            selected+=1
            if selected%500==0:print(f'RDF frames={selected} elapsed_s={time.monotonic()-clock:.1f}',flush=True)
        if not selected:raise ValueError('No selected frames')
        with h5py.File(output/'distributions.h5','x') as h5:
            hist.write(h5);h5.attrs['pair_convention']='Ordered distinct O pairs, ideal exposure N(N-1)/V; not an exact thermodynamic-limit RDF correction'
        with h5py.File(output/'collective.h5','x') as h5:
            h5.create_dataset('integer_modes',data=integer);h5.create_dataset('k_vectors',data=np.asarray([k_reference]))
            h5.create_dataset('rho_O',data=np.asarray(rho_modes))
            h5.create_dataset('structure_factors/OO_per_O',data=np.mean(abs(np.asarray(rho_modes))**2,axis=0)/len(ids))
            h5.create_dataset('source_frame',data=source_frames);h5.create_dataset('step',data=steps)
        rho=len(ids)/float(np.mean(volumes))
        temperature=args.temperature_K
        if temperature is None:
            match=re.search(r'-T([0-9.]+)-',source.name)
            if match:temperature=float(match.group(1))
        report.update(frames_analyzed=selected,oxygen_centers=len(ids),water_molecules=len(ids),source_columns=columns,
                      number_density_A_minus3=rho,density_g_cm3_from_18_01528_amu=rho*18.01528*1.66053906660,
                      temperature_K=temperature,temperature_source='explicit option' if args.temperature_K is not None else 'filename label, not measured temperature log',
                      frame_dt_ps=args.frame_dt_ps*args.stride,block_frames=args.block_frames,status='rdf_completed')
        (output/'report.json').write_text(json.dumps(report,indent=2))
        report['direct_correlations']=write_correlations(output,output/'direct_correlations.h5',args.oz_window,args.oz_tail_start,args.oz_min_s,
                                                       args.beta_mu,args.beta_mu_ex,args.reference_volume)
        report.update(status='completed',elapsed_s=time.monotonic()-clock)
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':'completed','frames':selected,'output':str(output),'c2':report['direct_correlations']['status'],'c1':report['direct_correlations']['c1_status']}),flush=True)
        return report
    except Exception as error:
        report.update(status='failed',failure_type=type(error).__name__,failure=str(error))
        (output/'report.json').write_text(json.dumps(report,indent=2));raise
