"""Element-resolved finite molecular shape/environment expansion.

All lengths use the input coordinate units. Elements are fixed C=0,H=1,N=2,O=3.
Complex coefficients transform as spherical harmonics under a common rotation;
scalar products over m give rotational invariants. Equivalent atoms are summed,
never used to select axes. This finite power basis is not a complete inverse.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm

from .geometry import mic


_ELEMENTS = ('C', 'H', 'N', 'O')


def _accumulate(vectors, groups, weights, n_groups, lmax, solid=False):
    """Sum Y_lm times weights by group; regular harmonics if solid=True."""
    result = np.zeros((n_groups, weights.shape[1], (lmax+1)**2), complex)
    if not len(vectors):
        return result
    radii = np.linalg.norm(vectors, axis=1)
    nonzero = radii > 0
    theta = np.arccos(np.clip(np.divide(vectors[:,2], radii,
                                      out=np.zeros_like(radii), where=nonzero), -1, 1))
    phi = np.arctan2(vectors[:,1], vectors[:,0])
    nrad = weights.shape[1]
    indices = (groups[:,None]*nrad + np.arange(nrad)).ravel()
    for ell in range(lmax+1):
        # A point at the origin has only an l=0 contribution. This also defines
        # the isotropic limit for coincident external sites without a fake axis.
        factor = radii**ell if solid else (np.ones_like(radii) if ell==0 else nonzero)
        for emm in range(-ell,ell+1):
            values = (weights * (factor*sph_harm(emm,ell,phi,theta))[:,None]).ravel()
            reduced = np.bincount(indices, weights=values.real, minlength=n_groups*nrad)
            reduced = reduced + 1j*np.bincount(indices, weights=values.imag, minlength=n_groups*nrad)
            result[:,:,ell*ell+ell+emm] = reduced.reshape(n_groups,nrad)
    return result


def _neighbors(centers, positions, cell, cutoff):
    """Exact site cutoff after periodic candidate selection, without an NxA grid."""
    if not len(positions) or not len(centers):
        return np.empty(0,int), np.empty(0,int), np.empty((0,3)), np.empty(0)
    diagonal = np.diag(cell)
    if np.allclose(cell, np.diag(diagonal), atol=1e-12) and np.all(diagonal>0):
        tree = cKDTree(np.mod(positions,diagonal), boxsize=diagonal)
        lists = tree.query_ball_point(np.mod(centers,diagonal), cutoff)
        counts = np.fromiter((len(x) for x in lists),int,count=len(centers))
        i = np.repeat(np.arange(len(centers)),counts)
        j = np.concatenate(lists).astype(int) if counts.sum() else np.empty(0,int)
        vectors = mic(positions[j]-centers[i],cell)
        distance = np.linalg.norm(vectors,axis=1)
        keep=distance<cutoff
        return i[keep],j[keep],vectors[keep],distance[keep]
    # ASE's general MIC handles skew/rotated cells, including non-reduced cells.
    # One center at a time bounds temporary storage instead of allocating NxA.
    owners=[]; sites=[]; vectors=[]; radii=[]
    for index,center in enumerate(centers):
        delta=mic(positions-center,cell)
        distance=np.linalg.norm(delta,axis=1)
        keep=np.flatnonzero(distance<cutoff)
        owners.append(np.full(len(keep),index,int)); sites.append(keep)
        vectors.append(delta[keep]); radii.append(distance[keep])
    return tuple(np.concatenate(x) for x in (owners,sites,vectors,radii))


def encode_molecules(centers, positions, elements, owner, cell, valid=None,
                     cutoff=6., nrad=3, lmax=3, neighbor_valid_only=False):
    """Return the common molecular basis and its equivariant components.

    ``coefficients`` sum Gaussian radial weights times complex Y_lm over
    external atomic sites. The Gaussian centers are b*cutoff/(nrad+1), b=1..nrad,
    width=cutoff/(nrad+1), with a cosine envelope vanishing at cutoff.
    ``shape_moments`` sum regular solid harmonics r**l Y_lm over owned sites,
    using MIC displacements from the molecular center (no radial truncation).
    Harmonic index is l*l+l+m, m=-l..l. ``shape_strength`` is sum_m |moment|²,
    with no normalization (units length**(2*l)).

    Feature order: radial sums (element,radial); environment powers
    (l,upper triangle of flattened element/radial channels); internal powers
    (l,upper triangle of elements); shape/environment couplings
    (l=1..lmax,shape element,environment element,radial). Each power/coupling
    is Re(sum_m a_lm conjugate(b_lm)), with no normalization. For defaults,
    12+312+40+144=508 scalar features. l=0 coupling is omitted as redundant
    with elemental shape counts and environment radial sums.

    ``neighbor_counts`` and ``neighbor_element_counts`` count external sites
    after the requested validity filter. Invalid center feature rows are NaN;
    their raw shape/environment arrays and counts remain available for audits.
    """
    centers=np.asarray(centers,float); positions=np.asarray(positions,float)
    elements_raw=np.asarray(elements); owner_raw=np.asarray(owner)
    elements=np.asarray(elements,int); owner=np.asarray(owner,int); cell=np.asarray(cell,float)
    n=len(centers); atoms=len(positions)
    if not np.isfinite(cutoff) or cutoff<=0 or int(nrad)!=nrad or nrad<1 or int(lmax)!=lmax or lmax<0:
        raise ValueError('cutoff must be positive, nrad positive integer, lmax nonnegative integer')
    nrad=int(nrad); lmax=int(lmax)
    if centers.shape!=(n,3) or positions.shape!=(atoms,3) or cell.shape!=(3,3):
        raise ValueError('centers/positions must have shape (N,3); cell (3,3)')
    if elements.shape!=(atoms,) or owner.shape!=(atoms,) or np.any(elements!=elements_raw) or np.any(owner!=owner_raw):
        raise ValueError('elements and owner must be integer arrays of length Natoms')
    if np.any((elements<0)|(elements>3)) or np.any((owner< -1)|(owner>=n)):
        raise ValueError('elements must be 0..3 and owner -1..N-1')
    if not all(np.all(np.isfinite(x)) for x in (centers,positions,cell)) or abs(np.linalg.det(cell))<1e-12:
        raise ValueError('finite coordinates and nonsingular cell required')
    valid=np.ones(n,bool) if valid is None else np.asarray(valid,bool)
    if valid.shape!=(n,): raise ValueError('valid must have shape (N,)')
    radial_centers=np.arange(1,nrad+1)*cutoff/(nrad+1)
    width=cutoff/(nrad+1)
    i,j,v,r=_neighbors(centers,positions,cell,cutoff)
    keep=owner[j]!=i
    if neighbor_valid_only:
        eligible=np.zeros(atoms,bool)
        eligible[owner>=0]=valid[owner[owner>=0]]
        keep &= eligible[j]
    i=i[keep]; j=j[keep]; v=v[keep]; r=r[keep]
    radial=np.exp(-.5*((r[:,None]-radial_centers)/width)**2)*(.5+.5*np.cos(np.pi*r[:,None]/cutoff))
    groups=i*4+elements[j]
    coefficients=_accumulate(v,groups,radial,n*4,lmax).reshape(n,4,nrad,(lmax+1)**2)
    radial_sums=np.zeros((n*4,nrad))
    for b in range(nrad): radial_sums[:,b]=np.bincount(groups,weights=radial[:,b],minlength=n*4)
    own=np.flatnonzero(owner>=0)
    internal=mic(positions[own]-centers[owner[own]],cell) if len(own) else np.empty((0,3))
    shape=_accumulate(internal,owner[own]*4+elements[own],np.ones((len(own),1)),n*4,lmax,solid=True).reshape(n,4,(lmax+1)**2)
    strength=np.stack([np.sum(np.abs(shape[:,:,l*l:(l+1)**2])**2,axis=-1) for l in range(lmax+1)],axis=-1)
    columns=[radial_sums.reshape(n,4*nrad)]
    names=[f'radial_{e}_r{b}' for e in _ELEMENTS for b in range(nrad)]
    env=coefficients.reshape(n,4*nrad,(lmax+1)**2)
    a,b=np.triu_indices(4*nrad); sa,sb=np.triu_indices(4)
    for l in range(lmax+1):
        sl=slice(l*l,(l+1)**2)
        columns.append(np.real(np.sum(env[:,a,sl]*env[:,b,sl].conj(),axis=-1)))
        names.extend(f'env_power_{_ELEMENTS[x//nrad]}r{x%nrad}_{_ELEMENTS[y//nrad]}r{y%nrad}_l{l}' for x,y in zip(a,b))
    for l in range(lmax+1):
        sl=slice(l*l,(l+1)**2)
        columns.append(np.real(np.sum(shape[:,sa,sl]*shape[:,sb,sl].conj(),axis=-1)))
        names.extend(f'shape_power_{_ELEMENTS[x]}_{_ELEMENTS[y]}_l{l}' for x,y in zip(sa,sb))
    for l in range(1,lmax+1):
        sl=slice(l*l,(l+1)**2)
        coupling=np.einsum('nsm,nerm->nser',shape[:,:,sl],coefficients[:,:,:,sl].conj()).real
        columns.append(coupling.reshape(n,16*nrad))
        names.extend(f'coupling_{s}_{e}_r{r}_l{l}' for s in _ELEMENTS for e in _ELEMENTS for r in range(nrad))
    features=np.concatenate(columns,axis=1)
    features[~valid]=np.nan
    counts=np.bincount(groups,minlength=n*4).reshape(n,4)
    return dict(features=features,names=names,coefficients=coefficients,
                shape_moments=shape,shape_strength=strength,radial_centers=radial_centers,
                radial_width=width,cutoff=float(cutoff),nrad=nrad,lmax=lmax,
                neighbor_counts=counts.sum(axis=1),neighbor_element_counts=counts,
                neighbor_valid_only=bool(neighbor_valid_only))
