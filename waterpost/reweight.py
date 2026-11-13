"""Static NVT importance sampling with explicit finite-population diagnostics.

No target-state dynamics, absolute chemical potentials, or bias guarantee.
All observations in a configuration share its single weight.
"""
import numpy as np
from scipy.linalg import helmert
from scipy.special import logsumexp

KB=8.617333262145e-5  # eV/K


def weights(logq):
    a=np.asarray(logq,float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('finite nonempty one-dimensional log weights required')
    return np.exp(a-logsumexp(a))


def log_weights(energy,source_T,target_T,field=None,a=0.):
    u=np.asarray(energy,float)
    if source_T<=0 or target_T<=0 or not np.isfinite([source_T,target_T,a]).all() or not np.isfinite(u).all():
        raise ValueError('finite positive temperatures and finite energies/amplitude required')
    logq=(1/(KB*source_T)-1/(KB*target_T))*(u-u.mean())
    if field is not None:
        f=np.asarray(field,float)
        if f.shape!=u.shape or not np.isfinite(f).all():raise ValueError('field must match energy')
        logq=logq-(source_T/target_T)*a*f
    return logq


def periodic_shape(positions,cell,axis,shape):
    if axis not in (0,1,2):raise ValueError('axis must be 0,1,2')
    s=(np.asarray(positions,float)@np.linalg.inv(cell))[...,axis]%1.
    phase=2*np.pi*s
    if shape=='cos1':return np.cos(phase)
    if shape=='sin1':return np.sin(phase)
    if shape=='cos2':return np.cos(2*phase)
    if shape=='sin2':return np.sin(2*phase)
    if shape=='mixed':return np.cos(phase)+.5*np.sin(2*phase)
    raise ValueError('unknown periodic field shape')


def block_estimate(x,w,block_frames,transform=None):
    """Equal-size contiguous delete-one-block ratio jackknife.

    Input must already be trimmed to complete blocks; never silently discard.
    Nonlinear transforms are evaluated after averaging, in every replicate.
    """
    x=np.asarray(x,float);w=np.asarray(w,float);n=len(w);b=int(block_frames)
    if b<1 or n%b or n//b<2 or len(x)!=n or not np.isfinite(x).all():
        raise ValueError('finite aligned observations, >=2 complete blocks required')
    if w.ndim!=1 or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError('nonnegative finite weights required')
    x=x.reshape(n,-1);w=w/w.sum();nb=n//b
    totals=np.sum(w[:,None]*x,axis=0)
    partial=(w[:,None]*x).reshape(nb,b,-1).sum(axis=1)
    mass=w.reshape(nb,b).sum(axis=1)
    if np.any(mass>=1-1e-14):raise ValueError('single block contains all probability mass')
    leave=(totals[None,:]-partial)/(1-mass[:,None])
    if transform is not None:
        totals=np.asarray(transform(totals));leave=np.asarray([transform(a) for a in leave])
    center=np.mean(leave,axis=0)
    se=np.sqrt((nb-1)/nb*np.sum((leave-center)**2,axis=0))
    return totals,se,leave


def paired_noise(base,target):
    """SE(target-source)/SE(source), from paired jackknife replicates."""
    bv,bs,bl=base;tv,ts,tl=target
    diff=tl-bl;nb=len(diff)
    ds=np.sqrt((nb-1)/nb*np.sum((diff-diff.mean(axis=0))**2,axis=0))
    tolerance=1e-11*np.maximum(1.,np.maximum(abs(bv),abs(tv)))
    ratio=np.divide(ds,bs,out=np.full_like(ds,np.inf),where=bs>tolerance)
    ratio=np.where((bs<=tolerance)&(ds<=tolerance),0.,ratio)
    passed=bool(np.isfinite(tv).all() and np.isfinite(ts).all() and np.all(ratio<=1.))
    return {'pass':passed,'max_added_noise_ratio':float(np.max(ratio)),
            'rms_added_noise_ratio':float(np.sqrt(np.mean(ratio**2)))},ds


def overlap_diagnostics(w,block_frames):
    w=np.asarray(w,float);w=w/w.sum();n=len(w);b=int(block_frames)
    if n%b:raise ValueError('trim weights to complete blocks')
    masses=w.reshape(-1,b).sum(axis=1);nb=len(masses)
    ess=1/np.sum(w*w);bess=1/np.sum(masses*masses)
    peak=float(w.max()*n)
    return {'frame_ess':float(ess),'frame_ess_fraction':float(ess/n),
            'block_ess':float(bess),'block_ess_fraction':float(bess/nb),
            'max_frame_weight':float(w.max()),'max_frame_mass_relative_to_uniform':peak,
            'max_block_weight':float(masses.max()),
            'pass':bool(ess/n>=.9 and bess/nb>=.9 and peak<=2.)}


def statistical_inefficiency(x):
    """Initial positive paired autocorrelation sum; diagnostic, not a proof."""
    x=np.asarray(x,float);x=x-x.mean();n=len(x);var=float(x@x/n)
    if var<1e-24:return 1.
    size=1<<(2*n-1).bit_length()
    f=np.fft.rfft(x,n=size);ac=np.fft.irfft(f*f.conj(),n=size)[:n]
    ac=ac/(np.arange(n,0,-1)*var);summed=0.
    for k in range(1,n//2-1,2):
        pair=ac[k]+ac[k+1]
        if pair<=0:break
        summed+=pair
    return max(1.,float(1+2*summed))


def tangent_basis(bins):
    if bins<2:raise ValueError('at least two bins required')
    return helmert(bins,full=False).T


def response_from_moments(moments,bins):
    mean=np.asarray(moments[:bins]);second=np.asarray(moments[bins:]).reshape(bins,bins)
    if np.any(mean<=0):raise ValueError('unsampled bins')
    cov=second-np.outer(mean,mean);q=tangent_basis(bins)
    projected=q.T@cov@q;eig=np.linalg.eigvalsh(projected)
    if eig[0]<=0 or eig[-1]/eig[0]>1e8:raise ValueError('singular/ill-conditioned canonical response')
    return q.T@np.diag(1/mean)@q-np.linalg.inv(projected)


def count_moments(counts):
    x=np.asarray(counts,float)
    if x.ndim!=2 or not np.allclose(x.sum(axis=1),x[0].sum()) or np.any(x<0):
        raise ValueError('bins must partition the complete fixed population')
    return np.concatenate([x,np.einsum('ni,nj->nij',x,x).reshape(len(x),-1)],axis=1)


def response_c2(counts,w):
    x=np.asarray(counts,float);w=np.asarray(w,float);w=w/w.sum()
    return response_from_moments(w@count_moments(x),x.shape[1])


def window_cumulants(moments):
    """Pooled window count moments -> mean,var,kappa3,kappa4,P(empty).

    This is not a direct correlation c3/c4 or a thermodynamic S(0).
    """
    m1,m2,m3,m4,p0=np.asarray(moments,float).reshape(5,-1)
    var=m2-m1*m1
    third=m3-3*m2*m1+2*m1**3
    fourth=m4-4*m3*m1+6*m2*m1*m1-3*m1**4-3*var*var
    return np.r_[m1,var,third,fourth,p0]
