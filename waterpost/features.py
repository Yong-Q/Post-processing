"""Finite, explicit local feature basis; not a learned inverse or free energy."""
import numpy as np
from scipy.special import sph_harm
from .geometry import mic,unit,water_frame


def encode(o,oh,cell,cutoff=6.,nrad=4,lmax=3,raw_h=None,h_owner=None,neighbor_water_only=False):
    n=len(o);frame=water_frame(oh);u=frame[:,0];normal=frame[:,2]
    valid_water=np.all(np.isfinite(oh),axis=(1,2))
    delta=mic(o[None,:,:]-o[:,None,:],cell);dist=np.linalg.norm(delta,axis=-1)
    np.fill_diagonal(dist,np.inf)
    neighbor_mask=dist<cutoff
    water_counts=np.sum(neighbor_mask&valid_water[None,:],axis=1)
    other_counts=np.sum(neighbor_mask&~valid_water[None,:],axis=1)
    if neighbor_water_only:dist[:,~valid_water]=np.inf
    i,j=np.where(dist<cutoff);r=dist[i,j];rh=delta[i,j]/r[:,None]
    centers=np.linspace(cutoff/(nrad+1),cutoff*nrad/(nrad+1),nrad)
    width=cutoff/(nrad+1)
    radial=np.exp(-.5*((r[:,None]-centers)/width)**2)*(.5+.5*np.cos(np.pi*r[:,None]/cutoff))
    ci=np.einsum('ij,ij->i',u[i],rh);cj=np.einsum('ij,ij->i',u[j],-rh)
    uu=np.einsum('ij,ij->i',u[i],u[j]);nn=np.einsum('ij,ij->i',normal[i],normal[j])**2
    scalars=[];names=[]
    for label,factor in [('count',np.ones(len(i))),('center_cos',ci),('neighbor_cos',cj),
                         ('bisector_cos',uu),('plane_cos2',nn),('center_cos2',ci**2),
                         ('neighbor_cos2',cj**2),('center_neighbor_product',ci*cj)]:
        out=np.zeros((n,nrad));np.add.at(out,i,radial*np.nan_to_num(factor)[:,None]);scalars.append(out)
        names.extend(f'{label}_radial{b}' for b in range(nrad))
    # Two site channels, excluding all sites belonging to the center molecule.
    coefficients=np.zeros((n,2,nrad,(lmax+1)**2),dtype=np.complex128)
    for species in range(2):
        if species==0:
            a,b=i,j;v=delta[i,j]
        else:
            if raw_h is None:
                hv=mic(delta[:,:,None,:]+oh[None,:,:,:],cell)
                hd=np.linalg.norm(hv,axis=-1);hd[np.arange(n),np.arange(n),:]=np.inf
                if neighbor_water_only:hd[:,~valid_water,:]=np.inf
                a,b,h=np.where(hd<cutoff);v=hv[a,b,h]
            else:
                hv=mic(raw_h[None,:,:]-o[:,None,:],cell);hd=np.linalg.norm(hv,axis=-1)
                hd[np.arange(n)[:,None]==h_owner[None,:]]=np.inf
                if neighbor_water_only:
                    keep=(h_owner>=0)&valid_water[np.maximum(h_owner,0)]
                    hd[:,~keep]=np.inf
                a,b=np.where(hd<cutoff);v=hv[a,b]
        length=np.linalg.norm(v,axis=-1);direction=v/length[:,None]
        rad=np.exp(-.5*((length[:,None]-centers)/width)**2)*(.5+.5*np.cos(np.pi*length[:,None]/cutoff))
        theta=np.arccos(np.clip(direction[:,2],-1,1));phi=np.arctan2(direction[:,1],direction[:,0])
        index=0
        for l in range(lmax+1):
            for m in range(-l,l+1):
                values=rad*sph_harm(m,l,phi,theta)[:,None]
                for radial_index in range(nrad):np.add.at(coefficients[:,species,radial_index,index],a,values[:,radial_index])
                index+=1
    # Same-channel and O/H cross-channel rotational invariants. Contains self
    # terms as well as contributions from two distinct neighbor sites.
    for l in range(lmax+1):
        sl=slice(l*l,(l+1)**2)
        for sa,sb in [(0,0),(0,1),(1,1)]:
            for a in range(nrad):
                for b in range(nrad):
                    if sa==sb and b<a:continue
                    value=np.real(np.sum(coefficients[:,sa,a,sl]*coefficients[:,sb,b,sl].conj(),axis=-1))
                    scalars.append(value[:,None]);names.append(f'power_s{sa}{sb}_r{a}{b}_l{l}')
    invariant=np.concatenate(scalars,axis=1)
    invariant[~np.all(np.isfinite(oh),axis=(1,2))]=np.nan
    return {'invariants':invariant,'names':names,'equivariant':coefficients,
            'radial_centers':centers,'radial_width':width,'lmax':lmax,
            'valid_water_neighbors':water_counts,'other_oxygen_neighbors':other_counts,
            'neighbor_water_only':neighbor_water_only}
