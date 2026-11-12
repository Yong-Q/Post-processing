"""Bulk scalar OZ and c1 offsets, grounded in PRL 134,056103 / PNAS 2026.

This is an oxygen-position projection, not the full molecular orientation DCF.
No closure is used to manufacture c1 from g at a single density.
"""
import json
from pathlib import Path
import numpy as np
import h5py
from scipy.fft import dst
from scipy.integrate import cumulative_trapezoid


def planar_projection(edges,c2,z):
    """2 pi int_|z|^R r c(r) dr; piecewise-constant radial bins."""
    edges=np.asarray(edges,float);c2=np.asarray(c2,float);z=np.asarray(z,float)
    lower=np.maximum(edges[:-1][None,:],abs(z).reshape(-1,1))
    integral=np.maximum(0,edges[1:][None,:]**2-lower**2)
    return (np.pi*(integral@c2)).reshape(z.shape)


def oz_from_rdf(edges,g,rho,tail_window='none',tail_start=.8,min_s=1e-8):
    """Midpoint radial DST-II/III discretization with finite h support.

    r_j=(j+.5)dr; k_m=(m+1)pi/R. At rho=0 the discrete inverse
    returns h exactly. k=0 is estimated separately, never read from fixed N.
    """
    edges=np.asarray(edges,float);g=np.asarray(g,float);rho=float(rho)
    if edges.ndim!=1 or g.ndim!=1 or len(edges)!=len(g)+1 or len(g)<2:
        raise ValueError('Require 1D bin edges and matching g, with at least two bins')
    dr=np.diff(edges)
    if not np.all(np.isfinite(edges)) or abs(edges[0])>1e-12 or np.any(dr<=0) or not np.allclose(dr,dr[0],rtol=1e-9,atol=1e-12):
        raise ValueError('OZ DST requires uniform radial bins starting at zero')
    if not np.all(np.isfinite(g)) or np.any(g<0) or not np.isfinite(rho) or rho<0:
        raise ValueError('g must be finite and nonnegative; rho must be finite and nonnegative')
    if not np.isfinite(min_s) or min_s<=0 or not 0<tail_start<1:
        raise ValueError('Positive min_s and 0<tail_start<1 required')
    r=(edges[:-1]+edges[1:])/2;R=edges[-1];dr=float(dr[0]);h=g-1
    window=np.ones_like(r)
    if tail_window=='cosine':
        tail=r>tail_start*R
        window[tail]=.5*(1+np.cos(np.pi*(r[tail]-tail_start*R)/(R*(1-tail_start))))
    elif tail_window!='none':raise ValueError('tail_window must be none or cosine')
    hw=h*window;k=np.arange(1,len(g)+1)*np.pi/R
    hhat=2*np.pi*dr*dst(r*hw,type=2)/k
    S=1+rho*hhat;valid=np.isfinite(S)&(S>min_s)
    chat=np.divide(hhat,S,out=np.full_like(hhat,np.nan),where=valid)
    c2=dst(k*chat,type=3)/(4*np.pi*R*r) if np.all(valid) else np.full_like(r,np.nan)
    hzero=float(4*np.pi*dr*np.sum(r*r*hw));s0=1+rho*hzero
    czero=hzero/s0 if np.isfinite(s0) and s0>min_s else np.nan
    integral_real=float(4*np.pi*dr*np.sum(r*r*c2)) if np.all(np.isfinite(c2)) else np.nan
    status='computed_with_finite_cutoff_assumption'
    if not np.all(valid):status='invalid_positive_k_spectrum'
    elif not np.isfinite(czero):status='computed_positive_k_invalid_zero_estimate'
    z=np.r_[0.,r]
    cbar=planar_projection(edges,c2,z) if np.all(np.isfinite(c2)) else np.full_like(z,np.nan)
    tail=r>=.8*R
    return {'status':status,'r_edges':edges,'r':r,'g_input':g,'h_raw':h,'tail_window':window,
            'h_used':hw,'k':k,'h_hat':hhat,'S_k':S,'valid_k':valid,'c2_hat':chat,'c2_r':c2,
            'rho_A_minus3':rho,'h_hat_zero':hzero,'S_zero_estimate':s0,'c2_hat_zero':czero,
            'c2_real_integral_midpoint':integral_real,'zero_mode_discretization_difference':integral_real-czero,
            'z':z,'c2_planar':cbar,'beta_over_area_functional_hessian':-cbar,
            'raw_tail_mean_g':float(g[tail].mean()),'raw_tail_max_abs_h':float(abs(h[tail]).max()),
            'valid_positive_k_count':int(valid.sum()),'total_positive_k_count':len(k),
            'tail_window_kind':tail_window,'tail_start_fraction':tail_start,'minimum_allowed_S':min_s,
            'scope':'Scalar oxygen-position effective direct correlation at one bulk state; finite-support/DST estimate; not full position-orientation molecular c2',
            'zero_mode_warning':'Estimated by truncating the RDF integral; fixed-N global number variance is not S(0); finite-N/cutoff effects not automatically corrected',
            'units':'r,z Angstrom; k Angstrom^-1; h,c2,S dimensionless; h_hat,c2_hat Angstrom^3; c2_planar Angstrom^2'}


def c1_equilibrium(rho,beta_vext=0.,beta_mu=None,reference_volume=1.):
    """PNAS Eq.3, c1=ln(reference_volume*rho)+beta Vext-beta mu."""
    rho=np.asarray(rho,float);v=np.broadcast_to(np.asarray(beta_vext,float),rho.shape)
    if np.any(rho<0) or np.any(~np.isfinite(rho)) or np.any(~np.isfinite(v)) or not np.isfinite(reference_volume) or reference_volume<=0:
        raise ValueError('Nonnegative finite rho, finite beta Vext, positive reference volume required')
    mask=rho>0;combo=np.full_like(rho,np.nan)
    combo[mask]=np.log(rho[mask]*reference_volume)+v[mask]
    result={'c1_plus_beta_mu':combo,'rho_positive_mask':mask,'reference_volume_A3':reference_volume,
            'status':'requires_beta_mu','formula':'c1 = ln(rho * Lambda^3/zeta) + beta Vext - beta mu'}
    if beta_mu is not None:
        if not np.isfinite(beta_mu):raise ValueError('beta_mu must be finite')
        result.update(c1=combo-float(beta_mu),beta_mu=float(beta_mu),status='computed_from_supplied_beta_mu')
    return result


def integrate_bulk_c1(rhos,c2_hat_zero,anchor_index=0,anchor_c1=None):
    """dc1_b/drho=c2_hat(0), same T/Hamiltonian/scalar component assumed.

    Without an explicit anchor only delta_c1 is returned. No unseen dilute
    states or zero-density reference are extrapolated.
    """
    rho=np.asarray(rhos,float);cz=np.asarray(c2_hat_zero,float)
    if rho.ndim!=1 or cz.shape!=rho.shape or len(rho)<2 or np.any(~np.isfinite(rho)) or np.any(rho<0) or np.any(np.diff(rho)<=0) or np.any(~np.isfinite(cz)):
        raise ValueError('At least two finite, strictly increasing nonnegative densities and finite c2_hat_zero values required')
    if not 0<=anchor_index<len(rho):raise ValueError('anchor_index out of range')
    integrated=cumulative_trapezoid(cz,rho,initial=0.);delta=integrated-integrated[anchor_index]
    result={'rho_A_minus3':rho,'c2_hat_zero':cz,'delta_c1':delta,'anchor_index':anchor_index,
            'reference_rho_A_minus3':rho[anchor_index],'status':'relative_only_missing_c1_anchor',
            'assumption':'Same T and Hamiltonian, scalar one-component thermodynamic mapping; no unmeasured low-density extrapolation'}
    if anchor_c1 is not None:
        if not np.isfinite(anchor_c1):raise ValueError('anchor_c1 must be finite')
        result.update(c1=delta+anchor_c1,anchor_c1=float(anchor_c1),status='anchored_bulk_c1')
    return result


def _write(group,data):
    for key,value in data.items():
        if isinstance(value,str):group.attrs[key]=value
        else:group.create_dataset(key,data=value)


def _json(data):
    if isinstance(data,dict):return {k:_json(v) for k,v in data.items()}
    if isinstance(data,(list,tuple)):return [_json(v) for v in data]
    if isinstance(data,np.ndarray):return _json(data.tolist())
    if isinstance(data,(float,np.floating)) and not np.isfinite(data):return None
    if isinstance(data,np.generic):return data.item()
    return data


def _sem(values):
    a=np.asarray(values,float);n=np.sum(np.isfinite(a),axis=0)
    mean=np.divide(np.nansum(a,axis=0),n,out=np.full_like(np.asarray(n,dtype=float),np.nan),where=n>0)
    ss=np.nansum((a-mean)**2,axis=0)
    return np.sqrt(np.divide(ss,n*(n-1),out=np.full_like(mean,np.nan),where=n>1)),n


def write_correlations(results,output_file,tail_window='none',tail_start=.8,min_s=1e-8,beta_mu=None,beta_mu_ex=None,reference_volume=1.,report_override=None):
    """Read existing OO RDF + state density, write a NEW HDF5 file."""
    results=Path(results);target=Path(output_file)
    if target.exists():raise ValueError('Output already exists; refusing overwrite: '+str(target))
    if beta_mu is not None and beta_mu_ex is not None:raise ValueError('Supply beta_mu OR beta_mu_ex, not both')
    report=report_override if report_override is not None else json.loads((results/'report.json').read_text())
    if 'number_density_A_minus3' in report:rho=float(report['number_density_A_minus3'])
    else:rho=float(report['density_g_cm3_from_18_01528_amu'])/(18.01528*1.66053906660)
    with h5py.File(results/'distributions.h5') as h5:
        edges=h5['rdf/OO/edges_0'][...];g=h5['rdf/OO/g'][...]
        blocks=h5['rdf/OO/block_g'][...] if 'rdf/OO/block_g' in h5 else None
        block_ids=h5['rdf/OO/block_ids'][...] if 'rdf/OO/block_ids' in h5 else (np.arange(len(blocks)) if blocks is not None else None)
    data=oz_from_rdf(edges,g,rho,tail_window,tail_start,min_s)
    one=c1_equilibrium(np.asarray(rho),beta_mu=beta_mu,reference_volume=reference_volume)
    one['scope']='Bulk scalar oxygen-center convention, beta Vext=0 assumed; no local per-water c1 labels inferred'
    if beta_mu_ex is not None:
        if not np.isfinite(beta_mu_ex):raise ValueError('beta_mu_ex must be finite')
        one.update(c1=np.asarray(-beta_mu_ex),status='computed_from_supplied_beta_mu_ex',beta_mu_ex=float(beta_mu_ex))
    target.parent.mkdir(parents=True,exist_ok=True)
    block_results=[];processed_ids=[];block_samples=[]
    if blocks is not None:
        for block_id,block in zip(block_ids,blocks):
            if np.all(np.isfinite(block)) and np.all(block>=0):
                block_results.append(oz_from_rdf(edges,block,rho,tail_window,tail_start,min_s));processed_ids.append(block_id)
                if 'frames_analyzed' in report and 'block_frames' in report:
                    length=int(report['block_frames']);start=int(block_id)*length;end=min(start+length,int(report['frames_analyzed']))
                    stride=int(report.get('configuration',{}).get('pair_stride',1))
                    block_samples.append(max(0,(end-1)//stride-(start-1)//stride) if end>start else 0)
                else:block_samples.append(1)
    eligible=np.asarray(block_samples)>=.5*max(block_samples) if block_samples else np.array([],bool)
    with h5py.File(target,'x') as h5:
        _write(h5.create_group('c2'),data);_write(h5.create_group('c1'),one)
        h5.attrs['source_results']=str(results.resolve())
        h5.attrs['radial_reference']='Input RDF uses N(N-1)/V ideal reference; no exact canonical-to-grand-canonical correction claimed'
        h5.attrs['paper_sources']='PRL 134,056103 (2025), Eqs A1-A8; PNAS 123,e2610049123 (2026), Eqs 2-3 and p9'
        collective=results/'collective.h5'
        if collective.exists():
            with h5py.File(collective) as coll:
                if 'structure_factors/OO_per_O' in coll and 'k_vectors' in coll:
                    kv=coll['k_vectors'][...]
                    if np.allclose(kv,kv[0],rtol=1e-6,atol=1e-8):
                        lengths=np.linalg.norm(kv[0],axis=-1);dr=edges[1]-edges[0]
                        forward=1+rho*4*np.pi*dr*np.sum((data['r']**2*data['h_used'])[None,:]*np.sinc(lengths[:,None]*data['r'][None,:]/np.pi),axis=1)
                        direct=coll['structure_factors/OO_per_O'][...]
                        _write(h5.create_group('nonzero_k_crosscheck'),{'k':lengths,'S_from_truncated_RDF':forward,'S_direct_modes':direct,
                                                                     'difference':forward-direct,'warning':'Nonzero-k cross-check only; finite cutoff/normalization/sampling can differ; no automatic calibration of S(0)'})
        if block_results:
            bg=h5.create_group('blocks')
            bg.create_dataset('block_ids',data=processed_ids);bg.create_dataset('rdf_frame_samples',data=block_samples);bg.create_dataset('stderr_eligible',data=eligible)
            for key in ['c2_r','c2_hat','c2_planar','c2_hat_zero','S_zero_estimate']:
                value=np.asarray([b[key] for b in block_results]);stderr,valid=_sem(value[eligible])
                bg.create_dataset(key,data=value);bg.create_dataset(key+'_stderr',data=stderr);bg.create_dataset(key+'_valid_blocks',data=valid)
            bg.attrs['status_by_block']=json.dumps([b['status'] for b in block_results])
            bg.attrs['warning']='Aggregate g gives the estimate; SEM uses finite blocks with >=50% of maximum RDF-frame count; shorter terminal blocks retained in aggregate but excluded from SEM. No bias/ensemble correction; inspect valid counts.'
    temperature=report.get('temperature_K')
    if temperature is None:
        import re
        match=re.search(r'-T([0-9.]+)-',Path(report.get('input') or '').name)
        if match:temperature=float(match.group(1))
    summary={'status':data['status'],'c1_status':one['status'],'rho_A_minus3':rho,'number_density_A_minus3':rho,'c2_hat_zero':data['c2_hat_zero'],
             'S_zero_estimate':data['S_zero_estimate'],'raw_tail_mean_g':data['raw_tail_mean_g'],
             'raw_tail_max_abs_h':data['raw_tail_max_abs_h'],'valid_positive_k_count':data['valid_positive_k_count'],
             'total_positive_k_count':data['total_positive_k_count'],'tail_window':tail_window,
             'output':str(target),'blocks_processed':len(block_results),'scope':data['scope'],
             'zero_mode_warning':data['zero_mode_warning'],'source_results':str(results.resolve()),
             'input':report.get('input'),'temperature_K':temperature,
             'temperature_source':report.get('temperature_source','source report or filename label; not measured here')}
    return _json(summary)


def run_correlations(args):
    output=Path(args.output)
    if output.exists():raise ValueError('Output already exists; refusing overwrite')
    output.mkdir(parents=True)
    summary=write_correlations(args.results,output/'direct_correlations.h5',args.oz_window,args.oz_tail_start,args.oz_min_s,
                               args.beta_mu,args.beta_mu_ex,args.reference_volume)
    (output/'report.json').write_text(json.dumps(summary,indent=2))
    return summary


def c1_density_scan(args):
    """Relative bulk c1 from same-T density scan, with explicit model scope."""
    if not args.assume_single_component:
        raise ValueError('Density-integration sum rule needs --assume-single-component; an OO projection in reacting water is not automatically a full molecular c1')
    output=Path(args.output)
    if output.exists():raise ValueError('Output already exists; refusing overwrite')
    rows=[];temperatures=[];grids=[]
    for directory in args.results:
        directory=Path(directory);report=json.loads((directory/'report.json').read_text())
        temperature=report.get('temperature_K')
        if temperature is None:
            import re
            match=re.search(r'-T([0-9.]+)-',Path(report.get('input') or '').name)
            if match:temperature=float(match.group(1))
        if temperature is None:temperature=args.temperature_K
        if temperature is None:raise ValueError('State temperature unknown; declare --temperature-K consistently for all states')
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be finite and positive')
        if args.temperature_K is not None and not np.isclose(temperature,args.temperature_K):raise ValueError('State temperature differs from supplied temperature')
        temperatures.append(temperature)
        with h5py.File(directory/'direct_correlations.h5') as h5:
            c=h5['c2'];rho=float(c['rho_A_minus3'][()]);zero=float(c['c2_hat_zero'][()])
            if not np.isfinite(zero):raise ValueError('Invalid estimated c2(k=0) at '+str(directory))
            if np.any(~c['valid_k'][...]):raise ValueError('Invalid positive-k OZ spectrum at '+str(directory))
            grids.append((float(c['r_edges'][-1]),len(c['r']),str(c.attrs['tail_window_kind']),float(c['tail_start_fraction'][()])))
            rows.append({'results':str(directory.resolve()),'rho':rho,'c2_hat_zero':zero,
                         'c2_hat_zero_block_stderr':float(h5['blocks/c2_hat_zero_stderr'][()]) if 'blocks/c2_hat_zero_stderr' in h5 else np.nan})
    if not np.allclose(temperatures,temperatures[0]):raise ValueError('Cannot integrate across different temperatures')
    if any(g!=grids[0] for g in grids):raise ValueError('Density scan must use identical RDF grids and tail treatments')
    rows.sort(key=lambda x:x['rho'])
    data=integrate_bulk_c1([r['rho'] for r in rows],[r['c2_hat_zero'] for r in rows],args.anchor_index,args.anchor_c1)
    data['temperature_K']=float(temperatures[0]);data['model_id_assumption']=args.model_id
    data['scope']='Conditional scalar one-component mapping of oxygen centers; not established as full reactive molecular thermodynamics'
    data['zero_mode_warning']='c2(k=0) is a finite-RDF estimate, sensitive to fixed-N effects and tails; derivative-sum-rule integration does not remove these errors'
    output.mkdir(parents=True)
    with h5py.File(output/'bulk_c1.h5','x') as h5:_write(h5,data)
    report=_json({**data,'states':rows,'quadrature':'Trapezoidal along supplied densities only; no extrapolation to rho=0; finite-state-spacing bias not quantified',
                  'number_of_states':len(rows),'output':str(output/'bulk_c1.h5')})
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report
