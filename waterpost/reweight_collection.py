"""Two-stage, immutable static augmentation: test ranges, then generate labels.

Heavy frame preparation runs in Slurm. No target-state MD is implied by
sampling diagnostics; no trajectories with different boxes are pooled.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import h5py
import numpy as np
from .geometry import mic,unique_radius
from .correlations import oz_from_rdf
from .reweight import (KB,weights,log_weights,periodic_shape,block_estimate,
    paired_noise,overlap_diagnostics,statistical_inefficiency,tangent_basis,
    count_moments,response_from_moments,window_cumulants)

SHAPES=('cos1','sin1','cos2','sin2','mixed')
SCALES=(.025,.05,.075,.10,.15,.20,.25,.30,.40)


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.generic):return x.item()
    return x


def write_json(path,data):
    with Path(path).open('x') as f:json.dump(clean(data),f,indent=2,ensure_ascii=False,allow_nan=False)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read_energy(csv_path,npz_path,steps):
    with Path(csv_path).open() as f:
        if f.readline().strip()!='step,pe_eV,press_bar':raise ValueError('Energy CSV requires explicit units/header')
    d=np.loadtxt(csv_path,delimiter=',',skiprows=1)
    if d.shape!=(len(steps),3) or not np.isfinite(d).all() or not np.array_equal(d[:,0],steps):
        raise ValueError('Energy/source frame mismatch')
    if len(np.unique(steps))!=len(steps):raise ValueError('Duplicate source steps')
    with np.load(npz_path,allow_pickle=False) as z:
        if not all(np.array_equal(z[k],d[:,j]) for j,k in enumerate(('step','pe','press'))):
            raise ValueError('CSV/NPZ disagree')
        meta=json.loads(str(z['meta']))
    if not meta.get('model_sha256') or not meta.get('temperature_K'):raise ValueError('Missing model or temperature provenance')
    return {'energy':d[:,1],'virial_pressure':d[:,2],'meta':meta,'csv_sha256':sha256(csv_path)}


def profile_transform(bins,volume,beta_v):
    beta_v=np.asarray(beta_v,float)
    def transform(mean_count):
        count=np.asarray(mean_count,float)
        if np.any(count<=0) or not np.isfinite(count).all():raise ValueError('Empty density bins')
        c=np.log(count/(volume/bins))+beta_v
        return c-c.mean()
    return transform


def rdf_frames(centers,cell,edges):
    centers=np.asarray(centers,float);n=centers.shape[1];i,j=np.triu_indices(n,1)
    volume=abs(np.linalg.det(cell));shell=4*np.pi/3*np.diff(np.asarray(edges)**3)
    if n<2 or edges[-1]>unique_radius(cell)+1e-10:raise ValueError('RDF beyond unique periodic sphere')
    expected=n*(n-1)/2*shell/volume
    return np.asarray([np.histogram(np.linalg.norm(mic(r[j]-r[i],cell),axis=1),bins=edges)[0]/expected for r in centers])


def classify_observable(base,target):
    result,added=paired_noise(base,target)
    result['status']='diagnostics_passed' if result.pop('pass') else 'unqualified'
    return result


def select_window(energy,monitors,burn_fraction=.2):
    n=len(energy);start=int(np.ceil(n*burn_fraction));x=np.asarray(monitors,float)
    g=max(statistical_inefficiency(v[start:]) for v in np.column_stack([energy,x]).T)
    block=max(50,int(np.ceil(5*g)))
    # Multiple of four blocks: calibration/holdout and a second block scale.
    used=((n-start)//(4*block))*4*block;nb=used//block
    result={'status':'unqualified','start':start,'stop':start+used,'block_frames':block,
            'blocks':nb,'statistical_inefficiency_max':g,'reason':'insufficient_complete_time_blocks',
            'burn_fraction':burn_fraction,'burn_in_equilibrium_proven':False}
    if nb<8:return result
    y=np.column_stack([energy,x])[start:start+used];half=used//2
    one=block_estimate(y[:half],np.ones(half),block);two=block_estimate(y[half:],np.ones(half),block)
    denominator=np.sqrt(one[1]**2+two[1]**2)
    floor=1e-10*np.maximum(1.,abs(y.mean(axis=0)))
    z=abs(one[0]-two[0])/np.maximum(denominator,floor)
    fine=block_estimate(y,np.ones(used),block);coarse=block_estimate(y,np.ones(used),2*block)
    ratio=coarse[1]/np.maximum(fine[1],floor)
    result.update(half_window_max_z=float(z.max()),max_SE_ratio_coarse_to_fine=float(ratio.max()))
    if np.any(z>4.):result['reason']='source_window_nonstationary'
    elif np.any(ratio>1.5):result['reason']='block_error_not_stable'
    else:result.update(status='diagnostics_passed',reason=None)
    return result


def _conditional_transform(v):
    half=len(v)//2;den=v[half:]
    if np.any(den<=0):raise ValueError('No finite support for a saved observable')
    return v[:half]/den


def prepare_case(task):
    """Cache sufficient per-frame inputs once, without copying raw trajectories."""
    case=task['case'];directory=Path(task['output'])/case;directory.mkdir()
    source=Path(task['source'])
    report=json.loads((source/'report.json').read_text())
    with h5py.File(source/'environment.h5','r') as h:
        step=h['step'][:];cells=h['cell'][:];centers=h['centers'][:].astype(float)
        source_frame=h['source_frame'][:] if 'source_frame' in h else np.arange(len(step))
        if not np.allclose(cells,cells[0],rtol=1e-10,atol=1e-10):raise ValueError('Only fixed-cell NVT supported')
        cell=cells[0];nframes,nmol=centers.shape[:2]
        if nframes!=report['frames_analyzed']:raise ValueError('Report/frame count mismatch')
        energy=read_energy(Path(task['energy_dir'])/(case+'.energy.csv'),Path(task['energy_dir'])/'npz'/(case+'.npz'),step)
        meta=energy['meta'];temperature=float(meta['temperature_K'])
        if meta.get('species')!=report['species'] or not np.isclose(temperature,report['temperature_K'],atol=1e-8):raise ValueError('State metadata mismatch')
        volume=float(abs(np.linalg.det(cell)));rho=nmol/volume
        if meta.get('density_mol_A3') is not None and not np.isclose(meta['density_mol_A3'],rho,rtol=1e-3):raise ValueError('Metadata density/cell mismatch')
        names=list(h['feature_names'].asstr()[:]);num=[];den=[]
        for start in range(0,nframes,8):
            f=h['features'][start:start+8].astype(float);good=np.isfinite(f)
            num.extend(np.sum(np.where(good,f,0.),axis=1)/nmol)
            den.extend(np.sum(good,axis=1)/nmol)
        numerator=np.asarray(num);denominator=np.asarray(den)
        extras=[];extra_den=[];extra_names=[]
        for key in ('packing_q','coordination','second_shell','lsi','radial_cv','hbond_donors','hbond_acceptors'):
            if key not in h:continue
            values=h[key][:].astype(float);good=np.isfinite(values)
            if not good.any():continue
            extras.append(np.where(good,values,0.).sum(axis=1)/nmol)
            extra_den.append(good.sum(axis=1)/nmol);extra_names.append(key)
        axis=h['axis'][:].astype(float);kind=h.attrs['axis_kind']
        valid=h['molecule_valid'][:].astype(bool)
        if kind in ('polar','director'):
            for dimension in range(3):
                c=axis[:,:,dimension];good=valid&np.isfinite(c)
                for rank,value in [(1,c),(2,(3*c*c-1)/2)]:
                    if rank==1 and kind!='polar':continue
                    extras.append(np.where(good,value,0.).sum(axis=1)/nmol)
                    extra_den.append(good.sum(axis=1)/nmol);extra_names.append(f'axis_P{rank}_lab_{dimension}')
        if extras:
            numerator=np.column_stack([numerator,*extras]);denominator=np.column_stack([denominator,*extra_den]);names.extend(extra_names)
        basis=str(h.attrs['feature_basis'])
    with h5py.File(source/'collective.h5') as h:
        rho_modes=h['rho_centers'][:];kv=h['k_vectors'][0];integer=h['integer_modes'][:]
        window_data={}
        if 'window_counts' in h:
            wc=h['window_counts'][:].astype(float)
            window_data={'window_moments':np.concatenate([np.mean(wc**power,axis=2) for power in (1,2,3,4)]+[np.mean(wc==0,axis=2)],axis=1),
                'window_radii_A':h['window_radii_A'][:],'window_centers_fractional':h['window_centers_fractional'][:]}
    structure=abs(rho_modes)**2/nmol
    # Include a spread of feature channels and all scalar/structural monitors.
    features=numerator/np.maximum(denominator,1e-30)
    watch=np.column_stack([features[:,np.linspace(0,507,8,dtype=int)],features[:,508:],structure])
    edges=np.linspace(0,min(12.,.98*unique_radius(cell)),121)
    rdf=rdf_frames(centers,cell,edges)
    counts=np.zeros((nframes,3,24),float)
    fractions=(centers@np.linalg.inv(cell))%1
    for i in range(nframes):
        for axis_index in range(3):counts[i,axis_index]=np.histogram(fractions[i,:,axis_index],np.linspace(0,1,25))[0]
    fields=np.stack([periodic_shape(centers,cell,d,s).sum(axis=1) for d in range(3) for s in SHAPES],axis=1)
    # Density-mode phase relaxation matters for a positional field even if
    # |rho_k|^2 appears decorrelated. Include the actual field coordinates.
    watch=np.column_stack([watch,fields])
    # Base temporal scale includes every saved feature numerator/denominator,
    # RDF bin and count-response moment, not just a sparse pilot watch list.
    time_inputs=[watch,numerator,denominator,rdf,rho_modes.real,rho_modes.imag]
    if window_data:time_inputs.append(window_data['window_moments'])
    for axis_index in range(3):
        for bins in (6,12,24):
            c=counts[:,axis_index].reshape(nframes,bins,24//bins).sum(axis=2)
            time_inputs.append(count_moments(c) if bins<=12 else c)
    window=select_window(energy['energy'],np.column_stack(time_inputs))
    np.savez_compressed(directory/'frames.npz',step=step,source_frame=source_frame,energy=energy['energy'],
        numerator=numerator,denominator=denominator,watch=watch,rdf=rdf,rdf_edges=edges,
        counts=counts,fields=fields,rho_modes=rho_modes,k_vectors=kv,integer_modes=integer,**window_data)
    provenance={'case':case,'species':report['species'],'source_temperature_K':temperature,
        'number_of_centers':nmol,'volume_A3':volume,'number_density_A_minus3':rho,
        'source_trajectory':report['input'],'source_trajectory_sha256':report['input_sha256'],
        'source_hdf5':str((source/'environment.h5').resolve()),'source_hdf5_size':(source/'environment.h5').stat().st_size,
        'source_feature_basis':json.loads(basis),'feature_names':names,'source_model_sha256':meta['model_sha256'],
        'energy_csv_sha256':energy['csv_sha256'],'energy_metadata':meta,'source_energy_commit':task['energy_commit'],
        'center_definition':report['center_definition'],'window':window,
        'split_group':report['input_sha256'],'units':{'energy':'whole-box eV','position':'Angstrom','density':'centers/Angstrom^3'},
        'coordinate_precision':'Centers copied from existing float32 postprocessing; original raw dump retained.',
        'energy_configuration_identity':'declared_same_named_trajectory; energy metadata lacks input trajectory hash; not independently verified',
        'sampling_Hamiltonian_identity':'energy model SHA256 available; original source report has no independently matched model hash',
        'unresolved_assumptions':['Energy rows correspond to these exact source configurations and the original sampling Hamiltonian, as declared by data producer.'],
        'independent_target_validation':False,'scope':'static fixed-NVT configurational observables only'}
    write_json(directory/'provenance.json',provenance)
    return {'case':case,'status':'prepared','window':window}


def _screen_logq(logq,watch,block,partitions='both'):
    n=len(logq);half=n//2;records=[]
    for label,sl in [('calibration',slice(0,half)),('holdout',slice(half,n))]:
        if partitions=='calibration' and label!='calibration':continue
        q=logq[sl];x=watch[sl];w=weights(q)
        for multiplier in (1,2):
            size=block*multiplier
            overlap=overlap_diagnostics(w,size)
            baseline=block_estimate(x,np.ones(len(w)),size)
            target=block_estimate(x,w,size)
            noise=classify_observable(baseline,target)
            records.append({'partition':label,'block_frames':size,'overlap':overlap,'noise':noise})
    return {'pass':all(r['overlap']['pass'] and r['noise']['status']=='diagnostics_passed' for r in records),
            'checks':records}


def pilot_case(task):
    directory=Path(task['output'])/task['case'];p=json.loads((directory/'provenance.json').read_text())
    win=p['window'];ranges=[];tests=[];t0=p['source_temperature_K'];accepted=[]
    if win['status']=='diagnostics_passed':
        with np.load(directory/'frames.npz',allow_pickle=False) as z:
            sl=slice(win['start'],win['stop']);u=z['energy'][sl];watch=np.column_stack([u,z['watch'][sl]])
            fields=z['fields'][sl];half=len(u)//2;beta0=1/(KB*t0)
            for kind in ['temperature','field']:
                indices=[None] if kind=='temperature' else list(range(15))
                for index in indices:
                    fluct=u[:half] if index is None else fields[:half,index]
                    std=float(np.std(fluct,ddof=1))
                    if std<=1e-12:continue
                    for sign in (-1,1):
                        best=None
                        for scale in SCALES:
                            if kind=='temperature':
                                beta=beta0+sign*scale/std
                                if beta<=0:break
                                temperature=1/(KB*beta);amplitude=0.
                                if abs(temperature/t0-1)>.02:break
                            else:temperature=t0;amplitude=sign*scale/std
                            field=None if index is None else fields[:,index]
                            logq=log_weights(u,t0,temperature,field,amplitude)
                            check=_screen_logq(logq,watch,win['block_frames'],partitions='calibration')
                            record={'kind':kind,'field_index':index,'sign':sign,'pilot_logq_scale':scale,
                                    'target_temperature_K':temperature,'amplitude_beta0':amplitude,**check}
                            tests.append(record)
                            if not check['pass']:break
                            best={k:v for k,v in record.items() if k not in ('checks','pass')}
                        if best:
                            field=None if index is None else fields[:,index]
                            confirm=_screen_logq(log_weights(u,t0,best['target_temperature_K'],field,best['amplitude_beta0']),watch,win['block_frames'])
                            best['holdout_confirmation']=confirm
                            ranges.append(best)
                            # Never shrink/tune the range in response to holdout.
                            if not confirm['pass']:continue
                            for fraction in (.25,.5,.75,1.):
                                beta=beta0+fraction*(1/(KB*best['target_temperature_K'])-beta0)
                                accepted.append({'kind':kind,'field_index':index,'target_temperature_K':1/(KB*beta),
                                                 'amplitude_beta0':fraction*best['amplitude_beta0']})
            tr=[r for r in ranges if r['kind']=='temperature' and r['holdout_confirmation']['pass']]
            for fr in [r for r in ranges if r['kind']=='field' and r['holdout_confirmation']['pass']]:
                for temp in tr:
                    beta=beta0+.5*(1/(KB*temp['target_temperature_K'])-beta0)
                    target=1/(KB*beta);amp=.5*fr['amplitude_beta0'];idx=fr['field_index']
                    check=_screen_logq(log_weights(u,t0,target,fields[:,idx],amp),watch,win['block_frames'])
                    tests.append({'kind':'joint','field_index':idx,'target_temperature_K':target,'amplitude_beta0':amp,**check})
                    if check['pass']:accepted.append({'kind':'joint','field_index':idx,'target_temperature_K':target,'amplitude_beta0':amp})
    result={'case':task['case'],'source_window':win,'ranges':ranges,'candidates_for_label_qualification':accepted,
            'pilot_tests':tests,'scope':'Sampling/overlap screen, not a target-bias bound; maximum tested accepted amplitude, not a theorem about all intermediate amplitudes.'}
    write_json(directory/'perturbation_ranges.json',result)
    return {'case':task['case'],'status':'pilot_completed','source_window':win,'ranges':ranges,'candidate_count':len(accepted)}


def _dataset(group,name,data):
    x=np.asarray(data)
    group.create_dataset(name,data=x,**({'compression':'gzip','compression_opts':1} if x.size>4 else {}))


def _label_estimate(group,x,w,block,transform,baseline=None):
    value,se,leave=block_estimate(x,w,block,transform)
    if not np.isfinite(value).all() or not np.isfinite(se).all():raise ValueError('nonfinite label/replicate')
    estimate=(value,se,leave)
    info={'status':'diagnostics_passed'}
    if baseline is not None:info=classify_observable(baseline,estimate)
    _dataset(group,'value',value);_dataset(group,'stderr',se)
    if baseline is not None:
        _,added=paired_noise(baseline,estimate);_dataset(group,'added_difference_stderr',added)
    coarse=block_estimate(x,w,2*block,transform)
    floor=1e-10*np.maximum(1.,abs(value))
    ratio=coarse[1]/np.maximum(se,floor)
    info['max_SE_ratio_coarse_to_fine']=float(np.max(ratio))
    if not np.isfinite(coarse[0]).all() or not np.isfinite(coarse[1]).all() or np.any(ratio>1.5):
        info.update(status='unqualified',reason='block_error_not_stable')
    half=len(w)//2
    first=block_estimate(x[:half],w[:half],block,transform)
    last=block_estimate(x[half:],w[half:],block,transform)
    temporal=abs(first[0]-last[0])/np.maximum(np.sqrt(first[1]**2+last[1]**2),floor)
    info['half_window_max_z']=float(np.max(temporal))
    if np.any(temporal>4.):info.update(status='unqualified',reason='label_temporal_window_nonstationary')
    group.attrs['qualification']=json.dumps(clean(info))
    return estimate,info


def _bin_potential(index,bins,temperature_ratio,amplitude):
    # Uniform-bin Boltzmann average, no molecular sample-dependent potential.
    _,shape=divmod(index,5)
    x=(np.arange(bins)[:,None]+(np.arange(32)+.5)[None,:]/32)/bins
    r=np.zeros((bins,32,3));r[:,:,0]=x
    f=periodic_shape(r,np.eye(3),0,SHAPES[shape])
    y=-temperature_ratio*amplitude*f
    from scipy.special import logsumexp
    effective=-(logsumexp(y,axis=1)-np.log(32))
    midpoint=temperature_ratio*amplitude*periodic_shape(np.column_stack([(np.arange(bins)+.5)/bins,np.zeros((bins,2))]),np.eye(3),0,SHAPES[shape])
    return effective,midpoint


def _oz_vector(g,edges,rho,window='none'):
    d=oz_from_rdf(edges,g,rho,tail_window=window)
    if not np.isfinite(d['c2_r']).all():raise ValueError('Invalid positive-k OZ spectrum')
    return np.r_[d['c2_r'],d['c2_hat'],d['c2_planar']]


def _bulk_c2(group,rdf,w,block,edges,rho,base_rdf,structure,rho_modes,nmol,kv):
    trans=lambda v:_oz_vector(v,edges,rho)
    base=block_estimate(base_rdf,np.ones(len(w)),block,trans)
    target,info=_label_estimate(group,rdf,w,block,trans,base)
    value,se,_=target;n=len(edges)-1
    _dataset(group,'r_edges_A',edges)
    d=oz_from_rdf(edges,w@rdf,rho)
    for key in ('r','k','z','g_input','S_k','c2_r','c2_hat','c2_planar'):_dataset(group,key,d[key])
    windowed=_oz_vector(w@rdf,edges,rho,'cosine')
    short_n=max(4,int(.9*n));short=oz_from_rdf(edges[:short_n+1],(w@rdf)[:short_n],rho)
    if not np.isfinite(short['c2_r']).all():raise ValueError('Reduced cutoff has invalid OZ spectrum')
    length=max(2,int(.8*short_n))
    for component,current,uncertainty,window_value,short_value in [
        ('radial',value[:length],se[:length],windowed[:length],short['c2_r'][:length]),
        ('planar',value[2*n:2*n+length+1],se[2*n:2*n+length+1],windowed[2*n:2*n+length+1],short['c2_planar'][:length+1])]:
        noise=max(float(np.sqrt(np.mean(uncertainty**2))),1e-10)
        for method,other in [('window',window_value),('cutoff',short_value)]:
            ratio=float(np.sqrt(np.mean((other-current)**2))/noise)
            info[f'{method}_{component}_sensitivity_RMS_in_SE']=ratio
            if ratio>1.:info.update(status='unqualified',reason=f'OZ_{method}_{component}_sensitivity_exceeds_sampling_SE')
    _dataset(group,'training_radial_mask',np.arange(n)<length)
    _dataset(group,'training_planar_mask',np.arange(n+1)<=length)
    group.attrs['c2_hat_strict_training_eligible']=False
    group.attrs['qualified_component_scope']='If diagnostics pass, radial/planar values within masks only; k-domain output is an auxiliary finite-DST estimate.'
    spectrum=block_estimate(rdf,w,block,lambda v:oz_from_rdf(edges,v,rho)['S_k'])
    _dataset(group,'S_k_stderr',spectrum[1])
    if np.any(d['S_k']<=3*spectrum[1]):info.update(status='unqualified',reason='OZ_spectrum_uncertainty_reaches_zero')
    # Direct nonzero-k covariance, with coherent finite-sample means removed.
    direct=w@structure-abs(w@rho_modes)**2/nmol
    kmag=np.linalg.norm(kv,axis=1);r=d['r'];dr=edges[1]-edges[0]
    linear=4*np.pi*rho*dr*r*r*np.sinc(kmag[:,None]*r/np.pi)
    from_rdf=1+(w@rdf-1)@linear.T
    sx=np.column_stack([structure,rho_modes.real,rho_modes.imag])
    m=structure.shape[1]
    sf_transform=lambda v:v[:m]-(v[m:2*m]**2+v[2*m:]**2)/nmol
    sd=block_estimate(sx,w,block,sf_transform)[1]
    sr=block_estimate(rdf,w,block,lambda g:1+(g-1)@linear.T)[1]
    discrepancy=abs(from_rdf-direct)/np.maximum(np.sqrt(sd**2+sr**2),1e-8)
    _dataset(group,'S_direct_connected',direct);_dataset(group,'S_from_RDF_at_native_k',from_rdf)
    info['max_nonzero_S_discrepancy_in_SE']=float(discrepancy.max())
    if np.any(discrepancy>3):info.update(status='unqualified',reason='OZ_native_nonzero_S_crosscheck_failed')
    # Zero mode remains explicitly ineligible; never manufacture absolute c1.
    group.attrs['finite_cutoff_c2_hat_zero']=float(d['c2_hat_zero'])
    group.attrs['zero_mode_training_eligible']=False
    group.attrs['zero_mode_status']='unqualified_finite_N_and_truncation_systematic_error'
    group.attrs['qualification']=json.dumps(clean(info))
    group.attrs['scope']='Scalar center finite-cutoff radial OZ; not full molecular orientation c2. value concatenates c2_r,c2_hat,c2_planar.'
    return info


def generate_case(task):
    directory=Path(task['output'])/task['case'];p=json.loads((directory/'provenance.json').read_text())
    pilot=json.loads((directory/'perturbation_ranges.json').read_text());win=p['window']
    manifest=[]
    if win['status']!='diagnostics_passed':
        write_json(directory/'manifest.json',{'case':task['case'],'status':'source_unqualified','window':win,'states':[]})
        return {'case':task['case'],'status':'source_unqualified','states':0}
    with np.load(directory/'frames.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    sl=slice(win['start'],win['stop']);block=win['block_frames'];n=win['stop']-win['start']
    u=data['energy'][sl];t0=p['source_temperature_K'];nmol=p['number_of_centers'];volume=p['volume_A3']
    numerators=data['numerator'][sl];denominators=data['denominator'][sl]
    # A missing metric does not erase every other feature. Require support in
    # both temporal halves; unsupported indices remain explicit in the schema.
    support=(denominators[:n//2].sum(axis=0)>0)&(denominators[n//2:].sum(axis=0)>0)
    observable_indices=np.flatnonzero(support)
    primitives=np.column_stack([numerators[:,support],denominators[:,support]])
    try:basefeatures=block_estimate(primitives,np.ones(n),block,_conditional_transform)
    except ValueError:basefeatures=None
    modes=data['rho_modes'][sl];mode_count=modes.shape[1]
    structure_inputs=np.column_stack([abs(modes)**2/nmol,modes.real,modes.imag])
    structure_transform=lambda v:v[:mode_count]-(v[mode_count:2*mode_count]**2+v[2*mode_count:]**2)/nmol
    base_structure=block_estimate(structure_inputs,np.ones(n),block,structure_transform)
    candidates=[{'kind':'direct','field_index':None,'target_temperature_K':t0,'amplitude_beta0':0.}]+pilot['candidates_for_label_qualification']
    with h5py.File(directory/'states.h5','x') as h:
        h.attrs['provenance']=json.dumps(clean(p));h.attrs['schema_version']='1.0'
        h.attrs['validation_limit']='Sampling diagnostics only; target-state bias not independently verified.'
        _dataset(h,'source_step',data['step']);_dataset(h,'source_frame',data.get('source_frame',np.arange(len(data['step']))))
        h.create_dataset('observable_names',data=np.asarray(p['feature_names'],dtype=h5py.string_dtype()))
        _dataset(h,'saved_observable_indices',observable_indices)
        h.attrs['unsupported_observable_indices']=json.dumps(np.flatnonzero(~support).tolist())
        for number,candidate in enumerate(candidates):
            identifier=f'{number:04d}';g=h.create_group(identifier);index=candidate['field_index']
            temp=candidate['target_temperature_K'];a=candidate['amplitude_beta0']
            field=None if index is None else data['fields'][sl,index]
            logq=log_weights(u,t0,temp,field,a);w=weights(logq)
            diagnostics=_screen_logq(logq,np.column_stack([u,data['watch'][sl]]),block)
            allweights=np.zeros(len(data['step']));allweights[sl]=w
            _dataset(g,'frame_weights',allweights)
            g.attrs['parameters']=json.dumps(candidate)
            if index is not None:
                axis,shape=divmod(index,5)
                g.attrs['external_potential']=json.dumps({'kind':'periodic_scalar_center_potential','axis':axis,'shape':SHAPES[shape],
                    'amplitude_beta0':a,'amplitude_eV_per_center':KB*t0*a,
                    'argument':'fractional cell coordinate; not an electric field','formula':'Uext=kBT0*a*sum_centers f(s)'})
            labels={};errors={}
            try:
                sg=g.create_group('connected_structure_factor')
                _,labels['connected_structure_factor']=_label_estimate(sg,structure_inputs,w,block,structure_transform,base_structure)
                _dataset(sg,'k_vectors_A_inverse',data['k_vectors'])
                sg.attrs['scope']='Connected equal-time center density covariance per center at saved nonzero k; not dynamics or thermodynamic zero mode'
            except (ValueError,np.linalg.LinAlgError) as error:errors['connected_structure_factor']=str(error)
            if 'window_moments' in data:
                try:
                    wm=data['window_moments'][sl];basewm=block_estimate(wm,np.ones(n),block,window_cumulants)
                    wg=g.create_group('pooled_window_count_cumulants')
                    _,labels['pooled_window_count_cumulants']=_label_estimate(wg,wm,w,block,window_cumulants,basewm)
                    _dataset(wg,'radii_A',data['window_radii_A']);_dataset(wg,'centers_fractional',data['window_centers_fractional'])
                    wg.attrs['order']='mean,variance,kappa3,kappa4,empty_probability; each in saved radius order'
                    wg.attrs['scope']='Pool fixed spatial windows with shared frame weights; in fields this is a spatial mixture, not local c3/c4 or thermodynamic S(0)'
                except (ValueError,np.linalg.LinAlgError) as error:errors['pooled_window_count_cumulants']=str(error)
            try:
                if basefeatures is None or len(observable_indices)==0:raise ValueError('No supported source observables')
                _,labels['local_observables']=_label_estimate(g.create_group('local_observables'),primitives,w,block,_conditional_transform,basefeatures)
            except (ValueError,np.linalg.LinAlgError) as error:errors['local_observables']=str(error)
            if index is None:
                rg=g.create_group('rdf');baseg=block_estimate(data['rdf'][sl],np.ones(n),block)
                _,labels['rdf']=_label_estimate(rg,data['rdf'][sl],w,block,None,baseg)
                _dataset(rg,'edges_A',data['rdf_edges'])
                try:
                    labels['bulk_c2']=_bulk_c2(g.create_group('bulk_c2'),data['rdf'][sl],w,block,data['rdf_edges'],
                        p['number_density_A_minus3'],data['rdf'][sl],abs(data['rho_modes'][sl])**2/nmol,data['rho_modes'][sl],nmol,data['k_vectors'])
                except (ValueError,np.linalg.LinAlgError) as error:errors['bulk_c2']=str(error)
                g.attrs['bulk_c1_plus_beta_mu']=float(np.log(p['number_density_A_minus3']))
                g.attrs['bulk_c1_reference_volume_A3']=1.
                g.attrs['bulk_c1_status']='absolute_c1_unavailable_unknown_mu; zero_mode_density_integration_not_qualified'
            else:
                axis=index//5
                raw=data['counts'][sl,axis]
                for bins in (6,12,24):
                    counts=raw.reshape(n,bins,24//bins).sum(axis=2)
                    beta_v,midpoint=_bin_potential(index,bins,t0/temp,a)
                    transform=profile_transform(bins,volume,beta_v)
                    # Use the identical target potential for the paired baseline:
                    # its deterministic contribution cancels in sampling variance.
                    cg=g.create_group(f'profile_{bins}')
                    try:
                        basec1=block_estimate(counts,np.ones(n),block,transform)
                        _,labels[f'c1_centered_{bins}']=_label_estimate(cg.create_group('c1_centered'),counts,w,block,transform,basec1)
                        mean=w@counts
                        _dataset(cg,'density_A_minus3',mean/(volume/bins));_dataset(cg,'beta_V_bin_boltzmann_average',beta_v)
                        _dataset(cg,'beta_V_midpoint',midpoint);_dataset(cg,'fractional_edges',np.linspace(0,1,bins+1))
                        cg.attrs['c1_scope']='ln(bin density)+bin effective beta V, arithmetic-bin-mean centered; finite-bin scalar-center approximation, unknown absolute mu'
                        discretization=float(np.max(abs(beta_v-midpoint)))
                        info=labels[f'c1_centered_{bins}'];info['max_potential_bin_discretization']=discretization
                        se=cg['c1_centered/stderr'][:]
                        if discretization>max(float(np.sqrt(np.mean(se*se))),1e-10):
                            info.update(status='unqualified',reason='potential_bin_discretization_exceeds_SE')
                        info['continuum_spatial_resolution']='not_independently_qualified; label is the specified finite-bin approximation only'
                        info['continuum_c1_training_eligible']=False
                        cg['c1_centered'].attrs['qualification']=json.dumps(clean(info))
                    except (ValueError,np.linalg.LinAlgError) as error:errors[f'c1_centered_{bins}']=str(error)
                    if bins in (6,12):
                        try:
                            moments=count_moments(counts);fn=lambda v:response_from_moments(v,bins)
                            reference=block_estimate(moments,np.ones(n),block,fn)
                            group=cg.create_group('canonical_projected_c2')
                            _,info=_label_estimate(group,moments,w,block,fn,reference)
                            effective=overlap_diagnostics(w,block)['frame_ess']/win['statistical_inefficiency_max']
                            if effective<10*(bins-1):info.update(status='unqualified',reason='insufficient_effective_samples_per_mode')
                            info['effective_frames_per_mode']=effective/(bins-1)
                            labels[f'canonical_projected_c2_{bins}']=info
                            _dataset(group,'Q',tangent_basis(bins));group.attrs['qualification']=json.dumps(clean(info))
                            group.attrs['scope']='Dimensionless finite-N bin-count Legendre-response excess Hessian on sum-zero subspace; not radial or continuum thermodynamic c2; no regularization'
                        except (ValueError,np.linalg.LinAlgError) as error:errors[f'canonical_projected_c2_{bins}']=str(error)
            eligible={name:(info['status']=='diagnostics_passed' and diagnostics['pass']) for name,info in labels.items()}
            status='direct_estimate' if candidate['kind']=='direct' else ('reweighted_diagnostics_passed' if any(eligible.values()) else 'reweighted_unqualified')
            entry={'id':task['case']+'/'+identifier,'origin':status,'is_perturbed':candidate['kind']!='direct',
                'split_group':p['split_group'],'parameters':candidate,'screen_passed':diagnostics['pass'],
                'labels':labels,'training_eligible_by_label':eligible,'errors':errors,
                'qualification_scope':'Conditional on declared source energy/configuration/Hamiltonian identity; finite-sampling diagnostics only.',
                'unresolved_assumptions':p.get('unresolved_assumptions',[]),
                'strict_independently_verified_training_eligible':False,
                'independent_target_validation':False,'weights_dataset':identifier+'/frame_weights','file':str(directory/'states.h5')}
            g.attrs['manifest']=json.dumps(clean(entry));manifest.append(entry)
    result={'case':task['case'],'status':'generated','source_window':win,'states':manifest}
    write_json(directory/'manifest.json',result)
    return {'case':task['case'],'status':'generated','states':len(manifest),
            'perturbed_states_with_qualified_label':sum(x['origin']=='reweighted_diagnostics_passed' for x in manifest)}


def _task_run(stage,task):
    try:return {'prepare':prepare_case,'pilot':pilot_case,'generate':generate_case}[stage](task)
    except Exception as error:
        root=Path(task['output']);root.mkdir(exist_ok=True,parents=True)
        result={'case':task['case'],'status':'failed','stage':stage,'error':str(error),'traceback':traceback.format_exc()}
        write_json(root/(task['case']+'.'+stage+'.failure.json'),result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True);parser.add_argument('--energy-dir',required=True)
    parser.add_argument('--output-root',required=True);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--energy-commit',required=True);parser.add_argument('--cases',nargs='*')
    args=parser.parse_args(argv);output=Path(args.output_root).resolve()
    output.mkdir(exist_ok=False,parents=True)
    sources=sorted(Path(args.source_root).glob('analysis/*/*/report.json'))
    tasks=[{'case':r.parent.name,'source':str(r.parent.resolve()),'energy_dir':str(Path(args.energy_dir).resolve()),
            'energy_commit':args.energy_commit,'output':str(output)} for r in sources if not args.cases or r.parent.name in args.cases]
    if not tasks:raise ValueError('No source cases')
    energy_cases={p.name.removesuffix('.energy.csv') for p in Path(args.energy_dir).glob('*.energy.csv')}
    if any(t['case'] not in energy_cases for t in tasks):raise ValueError('Missing energy cases')
    ncpu=int(os.environ.get('SLURM_CPUS_PER_TASK',args.workers));workers=min(args.workers,ncpu,len(tasks))
    if workers<1:raise ValueError('workers must be positive')
    write_json(output/'run_configuration.json',{'arguments':vars(args),'tasks':tasks,'workers':workers,
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'hostname':os.uname().nodename,'start_unix':time.time(),
        'order':['prepare','pilot_all_cases','publish_ranges','generate'],
        'status_definitions':'diagnostics_passed is a finite-sampling screen, not independently validated target bias'})
    summary={};active=tasks
    for stage in ('prepare','pilot','generate'):
        answers=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_task_run,stage,t) for t in active]
            for future in as_completed(futures):
                answer=future.result();answers.append(answer)
                print(stage,answer['case'],answer['status'],flush=True)
        summary[stage]=answers
        write_json(output/(stage+'_summary.json'),answers)
        if stage=='pilot':
            write_json(output/'perturbation_ranges_all.json',{'cases':answers,
                'units':'temperature K; field amplitude_beta0 dimensionless, multiply by kBT0 for eV per center',
                'heldout_role':'amplitude selection on first half; sampling diagnostics confirmed on second half',
                'note':'No data generation begins before this report is published.'})
        successful={a['case'] for a in answers if a['status']!='failed'}
        active=[t for t in active if t['case'] in successful]
    # Build a flat JSONL catalog without copying state data or frame features.
    with (output/'catalog.jsonl').open('x') as f:
        for task in tasks:
            path=output/task['case']/'manifest.json'
            if path.exists():
                for state in json.loads(path.read_text()).get('states',[]):f.write(json.dumps(clean(state),ensure_ascii=False)+'\n')
    summary['status']='completed_with_failures' if any(a['status']=='failed' for stage in summary.values() if isinstance(stage,list) for a in stage) else 'completed'
    write_json(output/'collection_summary.json',summary)
    return 1 if summary['status']=='completed_with_failures' else 0
