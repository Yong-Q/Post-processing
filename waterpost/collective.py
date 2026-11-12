import itertools
import numpy as np
from .geometry import mic


def modes(o,oh,cell,nmax=2,raw_h=None):
    integers=np.array([v for v in itertools.product(range(-nmax,nmax+1),repeat=3) if v!=(0,0,0)],dtype=int)
    # Keep one member of +/- pairs, preserving a unique deterministic mode list.
    integers=integers[[next(x for x in v if x)!=0 and next(x for x in v if x)>0 for v in integers]]
    k=2*np.pi*integers@np.linalg.inv(cell).T
    oxygen=np.exp(-1j*(o@k.T)).sum(axis=0)
    hp=(o[:,None,:]+oh).reshape(-1,3) if raw_h is None else raw_h
    hydrogen=np.exp(-1j*(hp@k.T)).sum(axis=0)
    return integers,k,oxygen,hydrogen


def window_counts(o,cell,centers_fractional,radii):
    centers=centers_fractional@cell
    dist=np.linalg.norm(mic(o[None,:,:]-centers[:,None,:],cell),axis=-1)
    return np.array([(dist<r).sum(axis=1) for r in radii])


def cumulants(counts):
    x=np.asarray(counts,dtype=float);mean=x.mean(axis=-1);dx=x-mean[...,None]
    var=np.mean(dx**2,axis=-1)
    return {'mean':mean,'variance':var,'third_cumulant':np.mean(dx**3,axis=-1),
            'fourth_cumulant':np.mean(dx**4,axis=-1)-3*var**2,
            'empty_probability':np.mean(x==0,axis=-1)}


def intermediate_scattering(positions,cells,rho_modes,k,lags):
    """Directional self and connected collective correlations, not spherical fits."""
    p=np.asarray(positions);rho=np.asarray(rho_modes);n=p.shape[1]
    connected=rho-rho.mean(axis=0);fs=[];fc=[]
    for lag in lags:
        phase=np.einsum('tnd,kd->tnk',p[lag:]-p[:len(p)-lag],k)
        fs.append(np.mean(np.cos(phase),axis=(0,1)))
        fc.append(np.real(np.mean(connected[lag:]*connected[:len(p)-lag].conj(),axis=0))/n)
    return {'self':np.array(fs),'collective_connected':np.array(fc),'k_vectors':k,
            'finite_trajectory_mean_subtracted':True}


def rdf_diagnostics(edges,g,rho,coordination_rho=None):
    from scipy.signal import find_peaks
    centers=(edges[:-1]+edges[1:])/2
    if not np.all(np.isfinite(g)):return {}
    peaks,_=find_peaks(g,prominence=.05)
    minima,_=find_peaks(-g,prominence=.03)
    mask=g>0
    integrand=np.ones_like(g);integrand[mask]=g[mask]*np.log(g[mask])-g[mask]+1
    return {'r_centers':centers,'peak_r':centers[peaks],'minimum_r':centers[minima],
            'integrated_coordination':np.cumsum(4*np.pi*(rho if coordination_rho is None else coordination_rho)*g*(edges[1:]**3-edges[:-1]**3)/3),
            'OO_position_pair_entropy_over_kB':-2*np.pi*rho*np.sum(integrand*(edges[1:]**3-edges[:-1]**3)/3),
            'entropy_warning':'OO point-position approximation; finite cutoff; NOT full molecular water entropy'}
