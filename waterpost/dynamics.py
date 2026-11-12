import numpy as np
from .geometry import mic


def lag_grid(n,max_lag=None):
    end=min(n-1,max_lag if max_lag is not None else max(1,n//4))
    if end<0:return np.array([],dtype=int)
    return np.unique(np.r_[np.arange(min(end+1,11)),np.geomspace(1,max(1,end),60).astype(int)])


def motion(positions,orientations,times,max_lag=None):
    p=np.asarray(positions);u=np.asarray(orientations);t=np.asarray(times)
    if len(p)<2:raise ValueError('Dynamics requires at least two frames')
    if np.any(np.diff(t)<=0) or not np.allclose(np.diff(t),np.diff(t)[0]):
        raise ValueError('Time correlations require uniformly spaced increasing times')
    lags=lag_grid(len(p),max_lag);result={k:[] for k in ['msd','fourth_displacement','alpha2','c1','c2','time_origins']}
    for lag in lags:
        stop=len(p)-lag
        displacement=p[lag:]-p[:stop];r2=np.sum(displacement**2,axis=-1)
        dot=np.clip(np.sum(u[lag:]*u[:stop],axis=-1),-1,1)
        m2=float(np.mean(r2));m4=float(np.mean(r2**2))
        result['msd'].append(m2);result['fourth_displacement'].append(m4)
        result['alpha2'].append(3*m4/(5*m2*m2)-1 if m2>0 else np.nan)
        result['c1'].append(np.nanmean(dot) if np.any(np.isfinite(dot)) else np.nan)
        result['c2'].append(np.nanmean((3*dot**2-1)/2) if np.any(np.isfinite(dot)) else np.nan)
        result['time_origins'].append(stop)
    result={k:np.array(v) for k,v in result.items()};result['lags_frames']=lags;result['lag_ps']=lags*(t[1]-t[0])
    return result


def inferred_nojump(positions,cells):
    p=np.asarray(positions);cells=np.asarray(cells)
    if not np.allclose(cells,cells[0],rtol=1e-6,atol=1e-6):
        raise ValueError('No-image unwrapping with a variable cell is unsupported; provide xu or image flags')
    delta=mic(np.diff(p,axis=0),cells[0])
    return np.concatenate([p[:1],p[:1]+np.cumsum(delta,axis=0)],axis=0)


def scalar_correlations(values,lags):
    x=np.asarray(values,dtype=float);mean=np.nanmean(x);dx=x-mean
    variance=np.nanmean(dx**2)
    raw=[]
    for lag in lags:
        raw.append(np.nanmean(dx[lag:]*dx[:len(dx)-lag]))
    raw=np.asarray(raw)
    return {'covariance':raw,'normalized':raw/variance if variance>0 else np.full_like(raw,np.nan),
            'global_mean':mean,'variance':variance}


def sampled_bond_correlations(edge_sets,lags):
    """Intermittent and uninterrupted-at-saved-frames survival of directed pairs."""
    intermittent=[];continuous=[];denominators=[]
    for lag in lags:
        den=num=surv=0
        for start in range(len(edge_sets)-lag):
            initial=edge_sets[start];den+=len(initial)
            num+=len(initial & edge_sets[start+lag])
            keep=initial.copy()
            for step in range(1,lag+1):
                keep.intersection_update(edge_sets[start+step])
                if not keep:break
            surv+=len(keep)
        intermittent.append(num/den if den else np.nan);continuous.append(surv/den if den else np.nan);denominators.append(den)
    return {'intermittent':np.array(intermittent),'continuous_at_saved_frames':np.array(continuous),'initial_bonds':np.array(denominators)}


def mean_autocorrelation(x):
    """FFT autocorrelation of the frame-averaged observable; finite time estimate."""
    x=np.asarray(x,dtype=float)
    if len(x)<4 or not np.all(np.isfinite(x)) or np.var(x)==0:
        return {'tau_int_frames':float('nan'),'effective_frames':float('nan')}
    y=x-x.mean();n=len(y);f=np.fft.rfft(y,n=2*n)
    ac=np.fft.irfft(f*f.conj(),n=2*n)[:n]/np.arange(n,0,-1);ac/=ac[0]
    nonpositive=np.flatnonzero(ac[1:]<=0);end=int(nonpositive[0]+1) if len(nonpositive) else max(1,n//4)
    end=min(end,max(1,n//4));tau=.5+float(np.sum(ac[1:end]))
    return {'autocorrelation':ac[:max(2,n//2)],'tau_int_frames':tau,'effective_frames':min(n,n/(2*tau)),
            'warning':'Finite sample initial-positive estimate; not proof of equilibration or spatial independence'}


def bond_orientation(oh,atom_ids,lags):
    """OH orientational correlation conditional on same O-H IDs at endpoints.

    H slots may exchange; a different H is never treated as the same bond.
    Transient breaks between saved endpoints are not reconstructed.
    """
    oh=np.asarray(oh);ids=np.asarray(atom_ids)
    norm=np.linalg.norm(oh,axis=-1,keepdims=True)
    u=np.divide(oh,norm,out=np.full_like(oh,np.nan,dtype=float),where=norm>0)
    c1=[];c2=[];count=[]
    for lag in lags:
        end=len(oh)-lag
        match=(ids[:end,:,:,None]==ids[lag:,:,None,:])&(ids[:end,:,:,None]>=0)
        dot=np.einsum('tihd,tikd->tihk',u[:end],u[lag:])
        valid=match&np.isfinite(dot);values=np.clip(dot[valid],-1,1)
        count.append(len(values));c1.append(np.mean(values) if len(values) else np.nan)
        c2.append(np.mean((3*values**2-1)/2) if len(values) else np.nan)
    return {'c1':np.array(c1),'c2':np.array(c2),'matched_bonds':np.array(count),
            'definition':'Same O-H atom IDs at endpoints, conditional on valid H2O endpoints; missed intermediate breaks unresolved'}


def persistent_water_orientation(u,h_ids,lags):
    """Conditional molecular rotation: identical H set at all saved frames."""
    u=np.asarray(u);ids=np.sort(np.asarray(h_ids),axis=-1)
    changed=np.any(ids[1:]!=ids[:-1],axis=-1)
    prefix=np.concatenate([np.zeros_like(changed[:1],dtype=int),np.cumsum(changed,axis=0)],axis=0)
    valid=np.all(ids>=0,axis=-1)&np.all(np.isfinite(u),axis=-1)
    c1=[];c2=[];counts=[]
    for lag in lags:
        end=len(u)-lag
        keep=valid[:end]&valid[lag:]&((prefix[lag:]-prefix[:end])==0)
        dot=np.clip(np.sum(u[:end]*u[lag:],axis=-1),-1,1)[keep]
        counts.append(len(dot));c1.append(np.mean(dot) if len(dot) else np.nan)
        c2.append(np.mean((3*dot**2-1)/2) if len(dot) else np.nan)
    return {'c1':np.array(c1),'c2':np.array(c2),'surviving_water_pairs':np.array(counts),
            'definition':'Conditional on unchanged O-H2 composition at EVERY saved frame; unobserved sub-frame transfers unresolved'}


def block_scan(x,seed=20261112):
    """Block-length sensitivity and block bootstrap of frame-level means."""
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)];sizes=[];errors=[];numbers=[]
    size=1
    while len(x)//size>=4:
        nb=len(x)//size;means=x[:nb*size].reshape(nb,size).mean(axis=1)
        sizes.append(size);numbers.append(nb);errors.append(np.std(means,ddof=1)/np.sqrt(nb));size*=2
    if not sizes:return {'status':'insufficient_frames_for_4_blocks'}
    rng=np.random.default_rng(seed);nb=numbers[-1];size=sizes[-1]
    means=x[:nb*size].reshape(nb,size).mean(axis=1)
    boot=np.mean(means[rng.integers(0,nb,(400,nb))],axis=1)
    return {'block_length_frames':np.array(sizes),'blocks':np.array(numbers),'mean_stderr':np.array(errors),
            'bootstrap_95_percentile':np.quantile(boot,[.025,.975]),'bootstrap_block_frames':size,
            'warning':'Block-length sensitivity must plateau; longest available block is not automatically independent'}


def displacement_distributions(p,lags,edges):
    counts=[]
    for lag in lags:
        radius=np.linalg.norm(p[lag:]-p[:len(p)-lag],axis=-1)
        counts.append(np.histogram(radius,bins=edges)[0])
    counts=np.array(counts);total=counts.sum(axis=1,keepdims=True)
    shell=4*np.pi/3*np.diff(edges**3)
    return {'edges':edges,'counts':counts,'radial_probability_density':np.divide(counts,total*np.diff(edges),out=np.full_like(counts,np.nan,dtype=float),where=total>0),
            'spatial_self_van_hove':np.divide(counts,total*shell,out=np.full_like(counts,np.nan,dtype=float),where=total>0),
            'in_range_normalization':True}
