import h5py
import numpy as np
import pytest
from waterpost.molecular_dynamics import write_molecular_dynamics
from waterpost.molecular_features import encode_molecules


def source(tmp_path, species='CO2', centers=None, owner=None, axis=None, moments=None, variable=False):
    centers=np.zeros((3,2,3)) if centers is None else np.asarray(centers,float)
    t,n=centers.shape[:2]
    path=tmp_path/'environment.h5'
    with h5py.File(path,'w') as f:
        f.attrs.update(species=species,axis_kind='none' if species=='CH4' else 'director',shape_lmax=3,
                       first_shell_cutoff_A=3.,second_shell_cutoff_A=6.)
        f['centers']=centers; f['cell']=np.tile(np.eye(3)*10,(t,1,1)); f['time_ps']=np.arange(t)*.1
        if variable: f['cell'][-1]=np.eye(3)*11
        f['molecule_valid']=np.ones((t,n),bool)
        f['axis']=np.tile([1.,0,0],(t,n,1)) if axis is None else axis
        f['plane_normal']=np.tile([0.,0,1.],(t,n,1))
        f['owner']=np.tile(np.repeat(np.arange(n),2),(t,1)) if owner is None else owner
        f['atom_ids']=np.arange(n*2); f['elements']=np.tile([0,1],n); f['molecule_ids']=np.arange(n)
        f['internal_shape_moments']=np.ones((t,n,4,16),complex) if moments is None else moments
        neighbors=f.create_group('neighbors')
        for i in range(t):
            g=neighbors.create_group(f'{i:06d}'); g['center_index']=[0]; g['neighbor_index']=[min(1,n-1)]
            g['distance']=[2.]; g['displacement']=[[2.,0,0]]; g['hbond_triplets']=np.empty((0,3),int)
    return path


def run(tmp_path, **kwargs):
    env=source(tmp_path,**kwargs); out=tmp_path/'dynamics.h5'
    result=write_molecular_dynamics(env,None,out,max_lag=2)
    return result,out


def test_director_reversal_and_no_overwrite(tmp_path):
    axis=np.tile([1.,0,0],(3,2,1)); axis[1]*=-1
    summary,path=run(tmp_path,axis=axis)
    with h5py.File(path) as f:
        assert 'c1' not in f['orientation/instantaneous']
        assert np.allclose(f['orientation/instantaneous/c2'],1)
        assert np.allclose(f['orientation/instantaneous/absolute_angle_rad'],0)
    with pytest.raises(ValueError,match='exists'):
        write_molecular_dynamics(tmp_path/'environment.h5',None,path)


def test_transfer_and_return_excludes_persistent_shape(tmp_path):
    owner=np.array([[0,0,1,1],[0,1,1,1],[0,0,1,1]])
    _,path=run(tmp_path,owner=owner)
    with h5py.File(path) as f:
        assert f['composition/same_composition_pairs'][-1]==0
        assert f['shape/persistent/pair_count'][-1]==0
        assert f['shape/instantaneous/pair_count'][-1]==2
        assert np.isnan(f['shape/persistent/correlation'][-1])


def test_nojump_boundary_crossing_is_explicitly_inferred(tmp_path):
    centers=np.tile(np.array([[9.9,2,2],[.1,2,2],[.3,2,2]])[:,None,:],(1,2,1))
    summary,path=run(tmp_path,centers=centers)
    with h5py.File(path) as f:
        assert np.isclose(f['translation/msd'][1],.04)
        assert 'inferred' in f['translation'].attrs['unwrapping']
        assert 'sparse' in f['translation'].attrs['warning'].lower()
        assert 'diffusion' not in f


def test_methane_rank_three_permutation_and_rotation(tmp_path):
    vertices=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*.63
    rotated=vertices@np.array([[0,1,0],[-1,0,0],[0,0,1.]])
    moments=[]
    for h in [vertices,vertices[[2,0,3,1]],rotated]:
        p=np.vstack([[0,0,0],h])+4
        shape=encode_molecules(np.array([[4.,4,4]]),p,np.array([0,1,1,1,1]),np.zeros(5,int),np.eye(3)*10)['shape_moments']
        moments.append(shape)
    _,path=run(tmp_path,species='CH4',centers=np.ones((3,1,3))*4,moments=np.array(moments))
    with h5py.File(path) as f:
        assert 'orientation' not in f
        assert f['shape'].attrs['rank']==3
        assert np.isclose(f['shape/instantaneous/correlation'][0],1)
        assert np.isclose(f['shape/instantaneous/correlation'][-1],-1)
        assert f['shape/instantaneous/pair_count'][-1]==1


def test_variable_cell_skips_translation(tmp_path):
    summary,path=run(tmp_path,variable=True)
    assert any('variable cell' in s for s in summary['skipped'])
    with h5py.File(path) as f: assert 'translation' not in f


def test_shell_break_return_and_hbond_pair_identity(tmp_path):
    env=source(tmp_path,species='H2O')
    with h5py.File(env,'a') as f:
        f.attrs['axis_kind']='polar'
        for t in range(3):
            f[f'neighbors/{t:06d}/distance'][...]=4. if t==1 else 2.
            del f[f'neighbors/{t:06d}/hbond_triplets']
            f[f'neighbors/{t:06d}/hbond_triplets']=np.array([[0,1 if t==0 else 3,1]]) if t!=1 else np.empty((0,3),int)
    out=tmp_path/'dynamics.h5';write_molecular_dynamics(env,None,out,max_lag=2,bond_max_lag=2)
    with h5py.File(out) as f:
        assert f['neighbors/first_shell/intermittent'][-1]==1
        assert f['neighbors/first_shell/continuous_at_saved_frames'][-1]==0
        assert f['hydrogen_bonds/intermittent'][-1]==1
        assert f['hydrogen_bonds/continuous_at_saved_frames'][-1]==0
        assert 'c1' in f['orientation/instantaneous']


def test_scalar_sampling_scattering_and_zero_shape(tmp_path):
    env=source(tmp_path)
    with h5py.File(env,'a') as f:
        f['coordination']=np.arange(6).reshape(3,2)
        f['internal_shape_moments'][...]=0
    coll=tmp_path/'collective.h5'
    with h5py.File(coll,'w') as f:
        f['k_vectors']=np.array([[[2*np.pi/10,0,0]]])
        f['rho_centers']=np.ones((3,1),complex)*2
    out=tmp_path/'dynamics.h5';summary=write_molecular_dynamics(env,coll,out,max_lag=2)
    with h5py.File(out) as f:
        assert np.allclose(f['scattering/self'],1)
        assert np.allclose(f['scattering/collective_connected'],0)
        assert f['scalar/coordination/correlation/pair_count'][0]==6
        assert f['shape/instantaneous/pair_count'][0]==0
        assert f['scalar/coordination/block_scan'].attrs['status']=='insufficient_frames_for_4_blocks'


def test_invalid_intermediate_excludes_persistent_identity(tmp_path):
    env=source(tmp_path)
    with h5py.File(env,'a') as f:f['molecule_valid'][1,0]=False
    out=tmp_path/'dynamics.h5';write_molecular_dynamics(env,None,out,max_lag=2)
    with h5py.File(out) as f:
        assert f['shape/instantaneous/pair_count'][-1]==2
        assert f['shape/persistent/pair_count'][-1]==1
        assert f['composition/same_composition_pairs'][-1]==2


def test_single_frame_and_nonuniform_times(tmp_path):
    env=source(tmp_path,centers=np.ones((1,2,3)))
    summary=write_molecular_dynamics(env,None,tmp_path/'short.h5')
    assert summary['status']=='insufficient_frames'
    with h5py.File(env,'w') as f:
        f['centers']=np.ones((3,2,3));f['cell']=np.tile(np.eye(3),(3,1,1));f['time_ps']=[0,.1,.3]
    with pytest.raises(ValueError,match='uniformly'):
        write_molecular_dynamics(env,None,tmp_path/'bad.h5')
    assert not (tmp_path/'bad.h5').exists()


def test_missing_hbond_data_is_explicitly_skipped(tmp_path):
    env=source(tmp_path,species='NH3')
    with h5py.File(env,'a') as f:
        for t in range(3):del f[f'neighbors/{t:06d}/hbond_triplets']
    out=tmp_path/'dynamics.h5';summary=write_molecular_dynamics(env,None,out,max_lag=2)
    assert any('hydrogen bond' in x and 'unavailable' in x for x in summary['skipped'])
    with h5py.File(out) as f:assert 'hydrogen_bonds' not in f
