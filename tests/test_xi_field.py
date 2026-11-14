import numpy as np
import pytest
from waterpost.xi_field import kernel_1d,field_3d,probe_features,moment_observables


def test_periodic_kernel_mass_and_images():
    s=np.array([.01,.5,.99]);k=kernel_1d(s,16,.12)
    np.testing.assert_allclose(k.sum(axis=0),1.)
    np.testing.assert_allclose(kernel_1d(s+2,16,.12),k,atol=1e-14)
    cell=np.diag([8.,9.,10.]);r=np.column_stack([s*8,s*9,s*10])
    z=np.column_stack([np.ones(3),[1.,2.,-1.]])
    f=field_3d(r,z,cell,8,.12)
    np.testing.assert_allclose(f.sum(axis=(0,1,2))*np.linalg.det(cell)/8**3,z.sum(axis=0))
    angle=.4;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    np.testing.assert_allclose(field_3d(r@rotation.T,z,cell@rotation.T,8,.12),f,atol=1e-14)
    np.testing.assert_allclose(field_3d(r+cell[0]/8,z,cell,8,.12),np.roll(f,1,axis=0),atol=1e-14)


def test_same_density_different_environment_field():
    cell=np.eye(3)*10;r=np.array([[1.,2.,3.],[7.,6.,5.]])
    z=np.array([[1.,2.],[1.,-1.]]);changed=z.copy();changed[0,1]+=1
    one=field_3d(r,z,cell,8,.12);two=field_3d(r,changed,cell,8,.12)
    np.testing.assert_array_equal(one[...,0],two[...,0])
    assert np.linalg.norm(one[...,1]-two[...,1])>0


def test_probe_features_match_formula():
    a=np.zeros((2,4,3,16),complex);b=np.zeros((2,4,16),complex)
    a[:,3,1,0]=2;a[:,1,1,4:9]=3;a[:,1,1,1:4]=2;b[:,1,1:4]=4
    z=probe_features(a,b)
    np.testing.assert_allclose(z[:,0],2*np.sqrt(4*np.pi))
    np.testing.assert_allclose(z[:,1],45.)
    np.testing.assert_allclose(z[:,2],24.)


def test_moment_number_mode_removed_and_kernel_attenuation():
    r=np.array([[0.,0,0],[2.5,0,0]]);z=np.ones((2,2));cell=np.eye(3)*10
    a=moment_observables(r,z,cell,.1)
    assert a.shape==(5,)
    np.testing.assert_allclose(a[:2],np.exp(-2*np.pi**2*.1**2)*np.ones(2),atol=1e-14)


def test_actual_h_rotation_changes_probes_at_fixed_oxygen_density():
    from waterpost.molecular_features import encode_molecules
    cell=np.eye(3)*15;centers=np.array([[3.,3.,3.],[5.8,3.6,3.]])
    offsets=np.array([[0,0,0],[.7,.6,0],[.7,-.6,0]])
    pos=np.concatenate([c+offsets for c in centers]);owner=np.repeat([0,1],3);elements=np.tile([3,1,1],2)
    rotated=pos.copy();rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    for i in range(2):rotated[3*i+1:3*i+3]=centers[i]+offsets[1:]@rotation.T
    def z(p):
        e=encode_molecules(centers,p,elements,owner,cell,cutoff=6.,nrad=3,lmax=3)
        return np.column_stack([np.ones(2),probe_features(e['coefficients'],e['shape_moments'])])
    before=field_3d(centers,z(pos),cell);after=field_3d(centers,z(rotated),cell)
    np.testing.assert_array_equal(before[...,0],after[...,0]);assert np.linalg.norm(before[...,1:]-after[...,1:])>1e-6
