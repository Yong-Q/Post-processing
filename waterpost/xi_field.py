"""Finite kernel-weighted environment fields; coordinates use row-vector cells."""
import numpy as np


def kernel_1d(fractions,bins=32,sigma=.12):
    """Periodic Gaussian deposition weights; each particle sums to one on grid."""
    if bins<4 or not 0<sigma<.3:raise ValueError('bins>=4 and 0<sigma<.3 required')
    s=np.asarray(fractions,float)%1;grid=(np.arange(bins)+.5)/bins
    delta=grid[:,None]-s.reshape(1,-1)
    value=sum(np.exp(-.5*((delta+image)/sigma)**2) for image in range(-2,3))
    return value/value.sum(axis=0,keepdims=True)


def field_3d(positions,z,cell,bins=8,sigma=.12):
    """xi_a(r)=sum_i Z_ia K(r-ri), density units inverse cell volume.

    Product kernel is defined in fractional coordinates; not isotropic in a
    general triclinic metric. This pilot declares that choice explicitly.
    """
    positions=np.asarray(positions,float);z=np.asarray(z,float);cell=np.asarray(cell,float)
    if positions.shape!=(len(z),3) or z.ndim!=2 or not np.isfinite(z).all():raise ValueError('finite aligned positions/Z required')
    s=(positions@np.linalg.inv(cell))%1
    k=[kernel_1d(s[:,d],bins,sigma) for d in range(3)]
    counts=np.einsum('xi,yi,zi,ia->xyza',*k,z,optimize=True)
    return counts*bins**3/abs(np.linalg.det(cell))


def probe_features(coefficients,shape):
    """Three raw, invariant Zi channels before train-only standardization.

    O radial r1; H environment l2 power; center-H/environment-H l1 coupling.
    All raw centers are included, not only centers currently classified H2O.
    """
    a=np.asarray(coefficients);b=np.asarray(shape)
    radial=a[...,3,1,0].real*np.sqrt(4*np.pi)
    angular=np.sum(abs(a[...,1,1,4:9])**2,axis=-1)
    coupled=np.sum(b[...,1,1:4]*a[...,1,1,1:4].conj(),axis=-1).real
    return np.stack([radial,angular,coupled],axis=-1)


def moment_observables(positions,z,cell,sigma=.12):
    """Planar Fourier projection of xi: density cos/sin, environment globals,
    then each environment cos/sin. Constant particle number is omitted.
    """
    z=np.asarray(z,float);s=(np.asarray(positions,float)@np.linalg.inv(cell))[:,0]%1
    lam=np.exp(-2*np.pi**2*sigma*sigma)
    trig=lam*np.stack([np.cos(2*np.pi*s),np.sin(2*np.pi*s)],axis=-1)
    return np.r_[z[:,0]@trig,z[:,1:].sum(axis=0),(z[:,1:].T@trig).ravel()]
