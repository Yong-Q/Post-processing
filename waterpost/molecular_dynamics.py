"""Saved-frame molecular dynamics, with explicit identity and sampling limits."""
from pathlib import Path
import numpy as np
import h5py
from .dynamics import (lag_grid, motion, inferred_nojump, displacement_distributions,
                       mean_autocorrelation, block_scan)


def _write(group, values):
    for key,value in values.items():
        if isinstance(value,dict): _write(group.create_group(key),value)
        elif isinstance(value,str) or key == 'rank': group.attrs[key]=value
        else: group.create_dataset(key,data=value)


def _mean(values, mask):
    keep=mask & np.isfinite(values)
    count=int(np.count_nonzero(keep))
    return (float(np.mean(values[keep])) if count else np.nan),count


def _membership_prefix(owner, n):
    """An ownership change affects both the old and the new molecule."""
    changes=np.zeros((len(owner),n),np.int32)
    t,a=np.nonzero(owner[1:]!=owner[:-1])
    for values in (owner[t,a],owner[t+1,a]):
        good=(values>=0)&(values<n)
        changes[t[good]+1,values[good]]=1
    return np.cumsum(changes,axis=0)


def _orientation(axis,lags,masks,kind):
    out={'pair_count':[],'c2':[],'absolute_angle_rad':[]}
    if kind=='polar':out['c1']=[]
    for lag,mask in zip(lags,masks):
        stop=len(axis)-lag
        dot=np.clip(np.einsum('tni,tni->tn',axis[lag:],axis[:stop]),-1,1)
        c,count=_mean((3*dot*dot-1)/2,mask)
        out['c2'].append(c);out['pair_count'].append(count)
        out['absolute_angle_rad'].append(_mean(np.arccos(abs(dot)),mask)[0])
        if kind=='polar':out['c1'].append(_mean(dot,mask)[0])
    return {k:np.array(v) for k,v in out.items()}


def _shape(q,lags,masks):
    norm=np.sqrt(np.sum(abs(q)**2,axis=-1))
    usable=np.isfinite(norm)&(norm>1e-12)
    normalized=np.divide(q,norm[...,None],out=np.zeros_like(q),where=usable[...,None])
    correlation=[];counts=[]
    for lag,mask in zip(lags,masks):
        stop=len(q)-lag
        overlap=np.einsum('tnm,tnm->tn',normalized[lag:].conj(),normalized[:stop]).real
        value,count=_mean(np.clip(overlap,-1,1),mask&usable[lag:]&usable[:stop])
        correlation.append(value);counts.append(count)
    return {'correlation':np.array(correlation),'pair_count':np.array(counts)}


def _pair_survival(edge_sets,lags):
    """Bound time-origin work, retaining uninterrupted sampled-frame meaning."""
    intermittent=[];continuous=[];denominators=[];strides=[];origins=[]
    for lag in lags:
        stop=len(edge_sets)-lag;stride=max(1,int(np.ceil(stop/200)))
        denominator=endpoint=surviving=0
        starts=range(0,stop,stride)
        for start in starts:
            initial=edge_sets[start];denominator+=len(initial)
            endpoint+=len(np.intersect1d(initial,edge_sets[start+lag],assume_unique=True))
            keep=initial.copy()
            for time in range(start+1,start+lag+1):
                keep=np.intersect1d(keep,edge_sets[time],assume_unique=True)
                if not len(keep):break
            surviving+=len(keep)
        intermittent.append(endpoint/denominator if denominator else np.nan)
        continuous.append(surviving/denominator if denominator else np.nan)
        denominators.append(denominator);strides.append(stride);origins.append(len(starts))
    return {'intermittent':np.array(intermittent),'continuous_at_saved_frames':np.array(continuous),
            'initial_pairs':np.array(denominators),'origin_stride':np.array(strides),'sampled_origins':np.array(origins),
            'definition':'Directed molecular pairs; at most 200 uniformly spaced saved time origins per lag; continuous means no absent SAVED frame; sub-frame breaks unresolved'}


def write_molecular_dynamics(environment_path,collective_path,output_path,max_lag=100,bond_max_lag=20):
    """Write a new HDF5 result and return a JSON-compatible capability summary.

    No diffusion coefficient is fitted. Position unwrapping is an inference from
    adjacent saved centers in a fixed cell; sparse sampling can hide box jumps.
    Internal shape correlations refer to the lab frame. Persistent correlations
    require unchanged atom ownership and validity at every saved intermediate
    frame; instantaneous correlations require only valid endpoints.
    """
    output=Path(output_path)
    if output.exists():raise ValueError('Output already exists; refusing overwrite: '+str(output))
    for name,value in [('max_lag',max_lag),('bond_max_lag',bond_max_lag)]:
        if int(value)!=value or value<0:raise ValueError(name+' must be a nonnegative integer')
    summary={'output':str(output),'skipped':[],'computed':[],'diffusion_fit':'not_performed'}
    data={};attrs={}
    with h5py.File(environment_path,'r') as env:
        p=env['centers'][...];cells=env['cell'][...];times=env['time_ps'][...]
        if p.ndim!=3 or p.shape[-1]!=3 or len(times)!=len(p) or cells.shape!=(len(p),3,3):
            raise ValueError('Inconsistent center, cell, or time shapes')
        t,n=p.shape[:2]
        species=str(env.attrs.get('species','unknown'));kind=str(env.attrs.get('axis_kind','none'))
        attrs.update(species=species,axis_kind=kind,source_environment=str(Path(environment_path).resolve()),
                     center_definition=str(env.attrs.get('center_definition','heavy_skeleton_com')),
                     warning='Saved-frame analysis cannot resolve sub-frame transfers, breaks, rotations, or hidden box crossings',
                     diffusion_fit='not_performed')
        summary.update(species=species,frames=t,molecules=n)
        if t<2:
            summary['status']='insufficient_frames';summary['skipped'].append('all time analyses: fewer than two frames')
        else:
            if np.any(~np.isfinite(times)) or np.any(np.diff(times)<=0) or not np.allclose(np.diff(times),np.diff(times)[0]):
                raise ValueError('Dynamics requires uniformly spaced increasing finite times')
            lags=lag_grid(t,int(max_lag));dt=times[1]-times[0]
            data.update(lag_frames=lags,lag_ps=lags*dt)
            valid=env['molecule_valid'][...].astype(bool)
            owner=env['owner'][...]
            if valid.shape!=(t,n) or owner.ndim!=2 or owner.shape[0]!=t:
                raise ValueError('Invalid validity or owner shape')
            if np.any((owner < -1)|(owner >= n)):raise ValueError('Invalid atom owner index')
            prefix=_membership_prefix(owner,n)
            invalid_prefix=np.vstack([np.zeros((1,n),int),np.cumsum(~valid,axis=0)])
            instant=[];persistent=[];same=[]
            for lag in lags:
                stop=t-lag;endpoints=valid[:stop]&valid[lag:]
                unchanged=prefix[lag:]==prefix[:stop]
                allvalid=(invalid_prefix[lag+1:]-invalid_prefix[:stop])==0
                instant.append(endpoints);persistent.append(endpoints&unchanged&allvalid)
                same.append(np.count_nonzero(unchanged))
            data['composition']={'same_composition_pairs':np.array(same),
                'instantaneously_valid_endpoint_pairs':np.array([x.sum() for x in instant]),
                'persistent_valid_pairs':np.array([x.sum() for x in persistent]),
                'definition':'same_composition_pairs: unchanged atom membership at EVERY saved frame, including invalid states; persistent_valid_pairs also requires valid target composition at every saved frame; transfer-and-return excluded; unseen sub-frame exchange unresolved'}
            if kind in ('polar','director') and 'axis' in env:
                axis=env['axis'][...]
                data['orientation']={
                    'instantaneous':_orientation(axis,lags,instant,kind),
                    'persistent':_orientation(axis,lags,persistent,kind),
                    'axis_kind':kind,
                    'definition':'Lab-frame axis correlations; instantaneous valid endpoints vs persistent valid atom membership; director sign arbitrary; absolute_angle_rad=acos(abs(dot))'}
                summary['computed'].append('axis_orientation_'+kind)
            else:summary['skipped'].append('axis orientation: no unique molecular axis')
            if 'plane_normal' in env and species in ('H2O','C2H4'):
                normal=env['plane_normal'][...]
                data['plane_orientation']={'instantaneous':_orientation(normal,lags,instant,'director'),
                    'persistent':_orientation(normal,lags,persistent,'director'),'axis_kind':'director'}
                summary['computed'].append('plane_orientation')
            rank={'H2O':1,'NH3':1,'CO2':2,'CH4':3,'C2H4':2,'C2H6':2}.get(species,1)
            if 'internal_shape_moments' in env and env['internal_shape_moments'].shape[-1]>=(rank+1)**2:
                # Read only the selected harmonic rank, once. Element summation
                # commutes with rotation and equivalent-atom permutation.
                q=env['internal_shape_moments'][...,rank*rank:(rank+1)**2].sum(axis=2)
                data['shape']={'instantaneous':_shape(q,lags,instant),'persistent':_shape(q,lags,persistent),
                    'rank':rank,'definition':'Real normalized complex overlap of sum over elemental solid-harmonic moments; zero/undefined moments excluded; lab-frame rotation correlation; not an injective orientation representation'}
                summary['computed'].append('shape_rank_'+str(rank))
            else:summary['skipped'].append('shape orientation: selected rank unavailable')
            fixed=np.allclose(cells,cells[0],rtol=1e-6,atol=1e-6)
            native='native_unwrapped_centers' in env
            if native or fixed:
                unwrapped=env['native_unwrapped_centers'][...] if native else inferred_nojump(p,cells)
                if unwrapped.shape!=p.shape or not np.all(np.isfinite(unwrapped)):raise ValueError('Invalid native unwrapped center data')
                translated=motion(unwrapped,np.full_like(p,np.nan),times,int(max_lag))
                for key in ('c1','c2'):translated.pop(key)
                translated.update(unwrapping='native_unwrapped_centers' if native else 'inferred_nojump_fixed_cell',
                    warning='Native positions retain supplied image information; variable-cell motion includes box deformation' if native else 'Sparse saved frames can conceal crossings or displacements beyond the MIC region; no native image information was supplied',
                    population='All tracked heavy skeleton centers, including composition-invalid groups; no persistent molecular identity conditioning',
                    length_units='Angstrom',time_units='ps')
                data['translation']=translated
                largest=max(float(np.max(np.linalg.norm(unwrapped[lag:]-unwrapped[:t-lag],axis=-1))) for lag in lags)
                edges=np.linspace(0,max(largest*1.000001,1e-6),101)
                data['van_hove']=displacement_distributions(unwrapped,lags,edges)
                summary['computed'] += ['native_center_msd_alpha2' if native else 'inferred_center_msd_alpha2','self_van_hove']
            else:summary['skipped'].append('translation/van Hove: variable cell without native image information')
            scalar={}
            for name in ('coordination','second_shell','packing_q','lsi','hbond_donors','hbond_acceptors'):
                if name not in env:continue
                x=env[name][...].astype(float);x[~valid]=np.nan
                finite=np.isfinite(x);count=finite.sum(axis=1)
                if not finite.any():
                    summary['skipped'].append(name+': no finite valid samples');continue
                mean=float(np.nanmean(x));dx=x-mean;variance=float(np.nanmean(dx*dx))
                cov=[];pairs=[]
                for lag,mask in zip(lags,instant):
                    value,num=_mean(dx[lag:]*dx[:t-lag],mask)
                    cov.append(value);pairs.append(num)
                cov=np.array(cov)
                frame_mean=np.divide(np.nansum(x,axis=1),count,out=np.full(t,np.nan),where=count>0)
                scalar[name]={'correlation':{'covariance':cov,'normalized':cov/variance if variance>0 else np.full_like(cov,np.nan),
                    'pair_count':np.array(pairs),'global_mean':mean,'variance':variance},
                    'autocorrelation':mean_autocorrelation(frame_mean),'block_scan':block_scan(frame_mean),
                    'definition':'Global-mean-centered observable; valid endpoints; no persistent membership condition; uncertainty uses frame means'}
            if scalar:data['scalar']=scalar;summary['computed'].append('scalar_correlations_and_sampling')
            if 'neighbors' in env:
                group=env['neighbors'];first=group.attrs.get('first_shell_cutoff_A',env.attrs.get('first_shell_cutoff_A'))
                second=group.attrs.get('second_shell_cutoff_A',env.attrs.get('second_shell_cutoff_A'))
                first_sets=[];second_sets=[];hb_sets=[];complete=True;hb_complete=True
                for frame in range(t):
                    key=f'{frame:06d}'
                    if key not in group:complete=False;break
                    g=group[key]
                    a=g['center_index'][...].astype(int);b=g['neighbor_index'][...].astype(int);r=g['distance'][...]
                    okay=valid[frame,a]&valid[frame,b]&(a!=b)
                    codes=a*n+b
                    first_sets.append(np.unique(codes[okay&(r<float(first))]) if first is not None else np.empty(0,int))
                    second_sets.append(np.unique(codes[okay&(r>=float(first))&(r<float(second))]) if first is not None and second is not None else np.empty(0,int))
                    if species in ('H2O','NH3'):
                        if 'hbond_triplets' not in g:hb_complete=False
                        hb=g['hbond_triplets'][...] if 'hbond_triplets' in g else np.empty((0,3),int)
                        keep=valid[frame,hb[:,0]]&valid[frame,hb[:,2]]
                        hb_sets.append(np.unique(hb[keep,0]*n+hb[keep,2]))
                blags=lag_grid(t,int(bond_max_lag))
                if complete:
                    if first is not None and second is not None:
                        data['neighbors']={'lag_frames':blags,'lag_ps':blags*dt,
                            'first_shell':_pair_survival(first_sets,blags),'second_shell':_pair_survival(second_sets,blags),
                            'first_shell_cutoff_A':float(first),'second_shell_cutoff_A':float(second),
                            'shell_definition':'First: r < first cutoff; second: first cutoff <= r < second cutoff; valid endpoint molecules'}
                        summary['computed'].append('sampled_first_second_shell_pair_survival')
                    else:summary['skipped'].append('shell pair survival: cutoff attributes unavailable')
                    if species in ('H2O','NH3') and hb_complete:
                        data['hydrogen_bonds']=dict(_pair_survival(hb_sets,blags),lag_frames=blags,lag_ps=blags*dt,
                            pair_definition='Directed donor-acceptor molecule pair; any equivalent donor H can support the pair; not a labeled-H bond lifetime')
                        summary['computed'].append('sampled_hydrogen_bond_pair_correlations')
                    elif species in ('H2O','NH3'):
                        summary['skipped'].append('hydrogen bonds: saved triplets unavailable')
                else:summary['skipped'].append('neighbor/HB survival: incomplete saved-frame neighbor groups')
            else:summary['skipped'].append('neighbor/HB survival: neighbor groups unavailable')
            if species not in ('H2O','NH3'):summary['skipped'].append('hydrogen bonds: unsupported species')
            if collective_path is not None and Path(collective_path).exists() and fixed:
                with h5py.File(collective_path,'r') as coll:
                    if 'rho_centers' in coll and 'k_vectors' in coll:
                        kv=coll['k_vectors'][...]
                        if kv.ndim==2:kv=kv[None]
                        if kv.shape[-1]!=3 or not np.allclose(kv,kv[0]):
                            summary['skipped'].append('scattering: variable reciprocal vectors')
                        else:
                            selection=np.unique(np.linspace(0,len(kv[0])-1,min(6,len(kv[0])),dtype=int))
                            k=kv[0,selection];rho=coll['rho_centers'][:,selection]
                            if rho.shape[0]!=t:raise ValueError('Collective and environment frame counts differ')
                            connected=rho-rho.mean(axis=0);fs=np.empty((len(lags),len(k)));fc=np.empty_like(fs)
                            # One mode at a time avoids a T*N*K phase cube.
                            for j,vector in enumerate(k):
                                phase=np.exp(-1j*(p@vector))
                                for ii,lag in enumerate(lags):
                                    stop=t-lag
                                    fs[ii,j]=np.einsum('tn,tn->',phase[lag:],phase[:stop].conj()).real/(stop*n)
                                    fc[ii,j]=np.mean(connected[lag:,j]*connected[:stop,j].conj()).real/n
                            data['scattering']={'self':fs,'collective_connected':fc,'k_vectors':k,'mode_indices':selection,
                                'definition':'At most six deterministic nonzero reciprocal modes; all tracked skeleton centers; connected collective density uses finite-trajectory mode mean; no isotropic average or diffusion fit'}
                            summary['computed'].append('subset_mode_self_collective_scattering')
                    else:summary['skipped'].append('scattering: collective modes unavailable')
            else:summary['skipped'].append('scattering: missing collective file or variable cell')
            summary['status']='computed_with_saved_frame_limits'
    output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(output,'x') as h5:
        for key,value in attrs.items():h5.attrs[key]=value
        _write(h5,data)
    return summary
