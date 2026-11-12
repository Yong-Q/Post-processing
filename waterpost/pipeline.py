import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
from .io import read_dump,WaterTopology
from .geometry import unique_radius,force_torque
from .static import local_metrics,hydrogen_bonds,steinhardt,periodic_voronoi
from .histograms import Histograms
from .analysis import Statistics
from .features import encode
from .collective import modes,window_counts,cumulants,intermediate_scattering,rdf_diagnostics
from .dynamics import motion,inferred_nojump,scalar_correlations,sampled_bond_correlations,mean_autocorrelation,displacement_distributions,bond_orientation,block_scan,persistent_water_orientation


def append(group,name,value,attrs=None):
    value=np.asarray(value)
    if name not in group:
        kw={'compression':'gzip','compression_opts':1} if value.size>1 else {}
        ds=group.create_dataset(name,shape=(0,)+value.shape,maxshape=(None,)+value.shape,dtype=value.dtype,**kw)
        if attrs:
            for k,v in attrs.items():ds.attrs[k]=v
    ds=group[name];ds.resize(ds.shape[0]+1,axis=0);ds[-1]=value


def store(group,values):
    for key,value in values.items():
        if isinstance(value,dict):store(group.require_group(key),value)
        elif isinstance(value,str):group.attrs[key]=value
        elif value is not None:group.create_dataset(key,data=value)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def inspect(path,ot=4,ht=2,scan=False):
    frames=read_dump(path);first=next(frames);top=WaterTopology(first,ot,ht,mode='nearest');top.extract(first)
    result={'path':str(Path(path).resolve()),'first_step':first.step,'atoms':len(first.data),
            'water_molecules':len(top.mol_ids),'columns':first.columns,'cell_A':first.cell.tolist(),
            'force_columns':all(k in first.columns for k in ['fx','fy','fz']),
            'unique_radius_A':unique_radius(first.cell),'type_mapping':{'O':ot,'H':ht}}
    if scan:
        count=1;last=first.step;intervals=set();nonwater=0;orphan=0;bad_original=[]
        for frame in frames:
            w=top.extract(frame);nonwater+=int((~w['water_valid']).sum());orphan+=len(w['unassigned_h_ids'])
            raw=frame.positions[top.indices];length=np.linalg.norm(mic_local(raw[:,1:]-raw[:,:1],frame.cell),axis=-1)
            if np.any(length>1.6) and len(bad_original)<8:bad_original.append({'frame':count,'step':frame.step,'max_original_OH_A':float(length.max())})
            intervals.add(frame.step-last);last=frame.step;count+=1
        result.update(frames=count,last_step=last,step_intervals=sorted(intervals),nonwater_O_frame_count=nonwater,
                      unassigned_H_frame_count=orphan,first_original_mol_anomalies=bad_original)
    return result


def analyze(args):
    output=Path(args.output)
    if output.exists():raise ValueError('Output already exists; refusing overwrite: '+str(output))
    source=Path(args.input)
    if not source.is_file():raise ValueError('Input does not exist: '+str(source))
    if args.frame_dt_ps<=0 or args.stride<1 or args.block_frames<1:raise ValueError('Positive time interval, stride and block size required')
    if not 0<args.first_cut<args.env_cut:raise ValueError('Require 0 < first-cut < env-cut')
    if args.rdf_max<=0 or args.expensive_stride<1 or args.pair_stride<1:raise ValueError('Cutoff and analysis strides must be positive')
    if not 0<args.hbond_cut<=args.env_cut or not 0<args.bond_cut<=args.env_cut:
        raise ValueError('Bond and hydrogen-bond cutoffs must be positive and no larger than env-cut')
    if not 0<=args.hbond_angle<=180 or args.split_gap_frames<0:raise ValueError('Invalid angle or split gap')
    if args.nrad<1 or args.lmax<0 or args.k_nmax<1 or args.window_centers<1 or args.bins<2:
        raise ValueError('Invalid basis, wavevector, sampling or histogram parameters')
    if args.max_frames is not None and args.max_frames<1:raise ValueError('max-frames must be positive')
    if args.max_lag is not None and args.max_lag<0:raise ValueError('max-lag must be nonnegative')
    if args.blocks is not None and args.blocks<1:raise ValueError('blocks must be positive')
    output.mkdir(parents=True)
    report={'status':'running','input':str(source.resolve()),'input_sha256':digest(source),
            'configuration':vars(args).copy(),'warnings':[],
            'periodicity':{'all_frames_periodic':True,'minimum_image':'orthorhombic exact or ASE lattice minimum image',
                           'bond_vectors':'H-O minimum image','ring_winding':'separately counted, excluded from local ring count'},
            'units':{'length':'Angstrom (configured convention)','time':'ps','forces':args.force_unit,'torque':args.force_unit+' * Angstrom'},
            'not_computed':['free energy','chemical potential','c3','c4','FEM','generative model training']}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    if args.blocks:
        count=sum(1 for idx,_ in enumerate(read_dump(source)) if idx>=args.start and (idx-args.start)%args.stride==0)
        count=min(count,args.max_frames) if args.max_frames else count
        block_frames=max(1,int(np.ceil(count/args.blocks)))
    else:block_frames=args.block_frames
    hist=Histograms(block_frames);stats=Statistics(hist,args.first_cut,args.env_cut,args.rdf_max,args.bins)
    selected=0;top=None;first_step=None;previous_step=None;step_interval=None;columns=None
    centers=np.random.default_rng(args.seed).random((args.window_centers,3))
    limits=[];vor_frames=[];feature_names=None;all_have_images=True
    starttime=time.monotonic()
    try:
        with h5py.File(output/'environment.h5','w') as env,h5py.File(output/'collective.h5','w') as coll:
            for source_index,frame in enumerate(read_dump(source)):
                if previous_step is not None:
                    diff=frame.step-previous_step
                    if diff<=0:raise ValueError('Non-increasing timesteps')
                    if step_interval is None:step_interval=diff
                    elif diff!=step_interval:raise ValueError('Irregular dump sampling; explicit per-frame times required')
                previous_step=frame.step
                if first_step is None:first_step=frame.step
                if source_index<args.start or (source_index-args.start)%args.stride:continue
                if args.max_frames is not None and selected>=args.max_frames:break
                if top is None:
                    top=WaterTopology(frame,args.oxygen_type,args.hydrogen_type,mode=args.topology_mode,bond_cut=args.bond_cut);columns=frame.columns
                    env.create_dataset('molecule_ids',data=top.mol_ids);env.create_dataset('atom_ids',data=top.ids[top.indices])
                    env.create_dataset('raw_atom_ids',data=top.ids);env.create_dataset('raw_atom_types',data=top.types)
                    env.create_dataset('raw_H_atom_ids',data=top.ids[top.h_indices]);env.create_dataset('original_mol_ids',data=top.mol)
                    env.attrs['definitions']='q: four nearest O, may lie outside first shell; LSI: gaps to first O outside lsi_cut; torque origin: central O'
                    env.attrs['force_units']=args.force_unit
                    stats.add('metadata_placeholder',np.empty(0),np.array([0,1]),0)
                    del hist.items['metadata_placeholder']
                w=top.extract(frame);o=w['o'];oh=w['oh'];n=len(o)
                append(env,'raw_positions',frame.positions.astype('f4'))
                if all(k in frame.columns for k in ['fx','fy','fz']):
                    append(env,'raw_forces',np.column_stack([frame.field(k) for k in ['fx','fy','fz']]).astype('f4'),{'units':args.force_unit})
                bound=unique_radius(frame.cell);limits.append(bound)
                if max(args.env_cut,args.rdf_max,args.lsi_cut)>=bound:
                    raise ValueError(f'Cutoff exceeds conservative unique-image radius {bound:g} Angstrom')
                if args.first_cut<=0:raise ValueError('Invalid first-shell cutoff')
                m=local_metrics(o,oh,frame.cell,args.first_cut,args.env_cut,args.lsi_cut)
                hb=hydrogen_bonds(o,oh,frame.cell,args.hbond_cut,args.hbond_angle)
                expensive=selected%args.expensive_stride==0
                vor=None
                if expensive and not args.no_voronoi:
                    try:vor=periodic_voronoi(o,frame.cell)
                    except ImportError:
                        if 'Voronoi unavailable: install optional freud-analysis' not in report['warnings']:report['warnings'].append('Voronoi unavailable: install optional freud-analysis')
                    if vor is not None:
                        if not np.isclose(vor['volume'].sum(),np.linalg.det(frame.cell),rtol=3e-5):raise ValueError('Periodic Voronoi volumes do not tile box')
                        vor_frames.append(selected)
                nd,na,network=stats.frame(o,oh,frame.cell,m,hb,selected,selected%args.pair_stride==0,expensive,vor,w['raw_h_positions'],w['h_owner'])
                for key,val in {'source_frame':source_index,'step':frame.step,'time_ps':source_index*args.frame_dt_ps,
                                'cell':frame.cell,'o':o.astype('f4'),'oh':oh.astype('f4'),'molecular_frame':m['frame'].astype('f4'),
                                'water_valid':w['water_valid'],'assigned_h_count':w['assigned_h_count'],
                                'current_atom_ids':w['atom_ids'],'raw_h_positions':w['raw_h_positions'].astype('f4'),'h_owner':w['h_owner'],
                                'q':m['q'].astype('f4'),'q_first_shell_valid':m['nearest'][:,3]<args.first_cut if n>=5 else np.zeros(n,bool),
                                'coordination':m['coordination'].astype('i4'),'second_shell':m['second_shell'].astype('i4'),
                                'lsi':m['lsi'].astype('f4'),'radial_cv':m['radial_cv'].astype('f4'),'nearest':m['nearest'].astype('f4'),
                                'hbond_donors':nd.astype('i4'),'hbond_acceptors':na.astype('i4')}.items():append(env,key,val)
                ii,jj=np.where(m['distance']<args.env_cut)
                ng=env.require_group('neighbors').create_group(f'{selected:06d}')
                for key,val in {'center_index':ii.astype('i4'),'neighbor_index':jj.astype('i4'),
                                'displacement':m['delta'][ii,jj].astype('f4'),'distance':m['distance'][ii,jj].astype('f4'),
                                'hbond_triplets':hb.astype('i4')}.items():ng.create_dataset(key,data=val,compression='gzip')
                if 'forces' in w:
                    append(env,'forces',w['forces'].astype('f4'),{'units':args.force_unit})
                    relative=np.concatenate([np.zeros((n,1,3)),oh],axis=1)
                    force,torque=force_torque(relative,w['forces'])
                    append(env,'molecule_force',force.astype('f4'));append(env,'molecule_torque_about_O',torque.astype('f4'))
                    bodyforce=np.einsum('ij,ikj->ik',force,m['frame']);bodytorque=np.einsum('ij,ikj->ik',torque,m['frame'])
                    # H-label-odd axes stored with their documented convention, not advertised as invariant scalars.
                    append(env,'body_force',bodyforce.astype('f4'));append(env,'body_torque',bodytorque.astype('f4'))
                    stats.add('force/magnitude',np.linalg.norm(force,axis=1),np.linspace(0,100,101),selected)
                    stats.add('torque/magnitude',np.linalg.norm(torque,axis=1),np.linspace(0,100,101),selected)
                    stats.add('joint/q_force',np.column_stack([m['q'],np.linalg.norm(force,axis=1)]),[stats.qedges,np.linspace(0,100,81)],selected)
                    stats.add('joint/hbond_force',np.column_stack([nd+na,np.linalg.norm(force,axis=1)]),[stats.integer,np.linspace(0,100,81)],selected)
                    stats.add('joint/hbond_torque',np.column_stack([nd+na,np.linalg.norm(torque,axis=1)]),[stats.integer,np.linspace(0,100,81)],selected)
                    stats.add('joint/q_torque',np.column_stack([m['q'],np.linalg.norm(torque,axis=1)]),[stats.qedges,np.linspace(0,100,81)],selected)
                if 'velocities' in w:append(env,'velocities',w['velocities'].astype('f4'))
                if 'charges' in w:append(env,'charges_native',w['charges'].astype('f4'))
                if 'unwrapped_o' in w:append(env,'native_unwrapped_o',w['unwrapped_o'])
                else:all_have_images=False
                if vor is not None:
                    vg=env.require_group('voronoi').create_group(f'{selected:06d}');store(vg,vor)
                if expensive:
                    eg=env.require_group('expensive').create_group(f'{selected:06d}')
                    store(eg,network)
                    for l in [4,6]:
                        ql,qbar=steinhardt(m['delta'],m['distance'],args.first_cut,l)
                        store(eg,{f'q{l}':ql,f'q{l}_bar':qbar})
                        stats.add(f'local/steinhardt_q{l}',ql,np.linspace(0,1,81),selected)
                        stats.add(f'local/steinhardt_q{l}_bar',qbar,np.linspace(0,1,81),selected)
                feature=encode(o,oh,frame.cell,args.env_cut,args.nrad,args.lmax,w['raw_h_positions'],w['h_owner'])
                append(env,'features',feature['invariants'].astype('f4'))
                if not args.no_equivariant:append(env,'equivariant_coefficients',feature['equivariant'].astype('c8'))
                water_feature=encode(o,oh,frame.cell,args.env_cut,args.nrad,args.lmax,w['raw_h_positions'],w['h_owner'],neighbor_water_only=True)
                append(env,'water_features',water_feature['invariants'].astype('f4'),{'definition':'Current valid H2O centers and H2O neighbors only; nonwater central rows are NaN'})
                append(env,'valid_water_neighbor_count',water_feature['valid_water_neighbors'].astype('i4'))
                append(env,'other_oxygen_neighbor_count',water_feature['other_oxygen_neighbors'].astype('i4'))
                weights=water_feature['invariants'][:,:args.nrad]
                terms=water_feature['invariants'][:,args.nrad:8*args.nrad].reshape(n,7,args.nrad)
                normalized=np.divide(terms,weights[:,None,:],out=np.full_like(terms,np.nan),where=weights[:,None,:]>1e-12)
                append(env,'water_orientation_weighted_means',normalized.astype('f4'),{'definition':'7 orientation moment types x nrad, divided by valid-water radial weight; empty denominator => NaN'})
                if not args.no_equivariant:append(env,'water_equivariant_coefficients',water_feature['equivariant'].astype('c8'))
                env['features'].attrs['neighbor_selection']='All neighboring O/H sites; unavailable neighbor directions contribute zero; use water_features for H2O-only analysis'
                if feature_names is None:
                    feature_names=feature['names'];env.create_dataset('feature_names',data=np.array(feature_names,dtype=h5py.string_dtype()))
                    env.attrs['feature_basis']=json.dumps({'cutoff_A':args.env_cut,'radial_centers':feature['radial_centers'].tolist(),'radial_width':feature['radial_width'],'lmax':args.lmax,'includes_neighbor_water_orientation':True})
                integer,k,rhoo,rhoh=modes(o,oh,frame.cell,args.k_nmax,w['raw_h_positions'])
                if selected==0:
                    coll.create_dataset('integer_modes',data=integer);coll.create_dataset('window_centers_fractional',data=centers)
                    radii=np.asarray([r for r in [1.,1.5,2.,2.5,3.,4.,5.] if r<bound])
                    coll.create_dataset('window_radii_A',data=radii)
                if not np.array_equal(integer,coll['integer_modes'][...]):raise ValueError('Fourier mode list changed')
                append(coll,'k_vectors',k);append(coll,'rho_O',rhoo);append(coll,'rho_H',rhoh)
                append(coll,'window_counts',window_counts(o,frame.cell,centers,radii).astype('i4'))
                u=np.nan_to_num(m['frame'][:,0])
                phase=np.exp(-1j*(o@k.T))
                polarization=np.einsum('nk,nd->kd',phase,u)
                khat=k/np.linalg.norm(k,axis=1)[:,None]
                longitudinal=abs(np.einsum('kd,kd->k',polarization,khat))**2/n
                total=np.sum(abs(polarization)**2,axis=1)/n
                append(coll,'orientation_structure_longitudinal',longitudinal)
                append(coll,'orientation_structure_transverse',total-longitudinal)
                selected+=1
                if selected%10==0:print(f'frames={selected} source_frame={source_index} elapsed_s={time.monotonic()-starttime:.1f}',flush=True)
            if not selected:raise ValueError('No selected frames')
            env.attrs['frames_analyzed']=selected;coll.attrs['frames_analyzed']=selected
            n=len(top.mol_ids)
            # Stream feature moments: a full 2000-frame run must not load all
            # million center environments merely to compute covariance/PCA.
            dim=len(feature_names);total=0;sumx=np.zeros(dim);sumxx=np.zeros((dim,dim))
            minimum=np.full(dim,np.inf);maximum=np.full(dim,-np.inf)
            for start in range(0,selected,8):
                x=env['features'][start:start+8].reshape(-1,dim).astype(float)
                x=x[np.all(np.isfinite(x),axis=1)]
                if not len(x):continue
                total+=len(x);sumx+=x.sum(axis=0);sumxx+=x.T@x
                minimum=np.minimum(minimum,x.min(axis=0));maximum=np.maximum(maximum,x.max(axis=0))
            if not total:raise ValueError('No valid H2O environments for orientation feature statistics')
            means=sumx/total;cov=(sumxx-total*np.outer(means,means))/max(1,total-1)
            std=np.sqrt(np.maximum(0,np.diag(cov)));scale=np.where(std>1e-12,std,1.)
            fg=coll.require_group('feature_statistics')
            fg.create_dataset('names',data=np.array(feature_names,dtype=h5py.string_dtype()))
            fg.create_dataset('mean',data=means);fg.create_dataset('std',data=std)
            fg.create_dataset('covariance',data=cov)
            fg.attrs['interpretation']='Empirical local-feature distribution; not F_ex or a complete high-dimensional rho(xi) reconstruction'
            target_names=['q','coordination','hbond_total']
            if 'molecule_force' in env:target_names.append('force_magnitude_native')
            sx=np.zeros((len(target_names),dim));sxx=np.zeros_like(sx);sy=np.zeros(len(target_names));syy=np.zeros_like(sy);sxy=np.zeros_like(sx);nt=np.zeros(len(target_names),int)
            for start in range(0,selected,8):
                x=env['features'][start:start+8].reshape(-1,dim).astype(float)
                ys=[env['q'][start:start+8].reshape(-1),env['coordination'][start:start+8].reshape(-1),
                    (env['hbond_donors'][start:start+8]+env['hbond_acceptors'][start:start+8]).reshape(-1)]
                if 'molecule_force' in env:ys.append(np.linalg.norm(env['molecule_force'][start:start+8],axis=-1).reshape(-1))
                for index,y in enumerate(ys):
                    valid=np.all(np.isfinite(x),axis=1)&np.isfinite(y);xx=x[valid];yy=y[valid]
                    nt[index]+=len(yy);sx[index]+=xx.sum(axis=0);sxx[index]+=(xx**2).sum(axis=0)
                    sy[index]+=yy.sum();syy[index]+=(yy**2).sum();sxy[index]+=(xx*yy[:,None]).sum(axis=0)
            nc=np.maximum(nt,1);covxy=sxy/nc[:,None]-(sx/nc[:,None])*(sy/nc)[:,None]
            denominator=np.sqrt(np.maximum(0,sxx/nc[:,None]-(sx/nc[:,None])**2)*np.maximum(0,syy/nc-(sy/nc)**2)[:,None])
            assoc=fg.require_group('target_associations')
            assoc.create_dataset('target_names',data=np.array(target_names,dtype=h5py.string_dtype()))
            store(assoc,{'pearson_correlation':np.divide(covxy,denominator,out=np.full_like(covxy,np.nan),where=denominator>1e-12),'valid_samples':nt})
            assoc.attrs['warning']='Descriptive same-data associations; no predictive reconstruction or independent validation'
            feature_edges=[];feature_counts=[]
            for idx,name in enumerate(feature_names):
                lo,hi=minimum[idx],maximum[idx]
                if lo==hi:lo-=.5;hi+=.5
                feature_edges.append(np.linspace(lo,hi,65));feature_counts.append(np.zeros(64,dtype=np.int64))
            standardized_cov=cov/(scale[:,None]*scale[None,:])
            eigenvalues,eigenvectors=np.linalg.eigh(standardized_cov)
            order=np.argsort(eigenvalues)[::-1];eigenvalues=np.maximum(0,eigenvalues[order]);components=eigenvectors[:,order].T[:3]
            fg.create_dataset('pca_components',data=components);fg.create_dataset('pca_singular_values',data=np.sqrt(eigenvalues*max(1,total-1)))
            extent=6*np.sqrt(np.maximum(eigenvalues[:2],1e-10));px=np.linspace(-extent[0],extent[0],49);py=np.linspace(-extent[1],extent[1],49)
            pc_counts=np.zeros((48,48),dtype=np.int64)
            for start in range(0,selected,8):
                x=env['features'][start:start+8].reshape(-1,dim).astype(float)
                x=x[np.all(np.isfinite(x),axis=1)]
                if not len(x):continue
                for idx in range(dim):feature_counts[idx]+=np.histogram(x[:,idx],bins=feature_edges[idx])[0]
                projected=((x-means)/scale)@components.T
                pc_counts+=np.histogram2d(projected[:,0],projected[:,1],bins=[px,py])[0].astype(np.int64)
            for idx,name in enumerate(feature_names):
                counts=feature_counts[idx];edges=feature_edges[idx]
                sg=fg.require_group('marginals').create_group(f'{idx:03d}');sg.attrs['name']=name
                store(sg,{'edges':edges,'counts':counts,'density':counts/(total*np.diff(edges))})
            store(fg.require_group('pca_joint'),{'counts':pc_counts,'edges_0':px,'edges_1':py,'total_centers':total,'in_range_centers':int(pc_counts.sum())})
            wg=coll.require_group('water_feature_statistics');wn=0;ws=np.zeros(dim);wss=np.zeros((dim,dim))
            wmin=np.full(dim,np.inf);wmax=np.full(dim,-np.inf)
            for start in range(0,selected,8):
                x=env['water_features'][start:start+8].reshape(-1,dim).astype(float);x=x[np.all(np.isfinite(x),axis=1)]
                if not len(x):continue
                wn+=len(x);ws+=x.sum(axis=0);wss+=x.T@x;wmin=np.minimum(wmin,x.min(axis=0));wmax=np.maximum(wmax,x.max(axis=0))
            wm=ws/wn;wcov=(wss-wn*np.outer(wm,wm))/max(1,wn-1)
            store(wg,{'mean':wm,'std':np.sqrt(np.maximum(0,np.diag(wcov))),'covariance':wcov,'valid_centers':wn})
            wg.create_dataset('names',data=np.array(feature_names,dtype=h5py.string_dtype()))
            constant=wmax<=wmin;wmin[constant]-=.5;wmax[constant]+=.5
            wedge=np.array([np.linspace(wmin[k],wmax[k],65) for k in range(dim)]);wcounts=np.zeros((dim,64),int)
            for start in range(0,selected,8):
                x=env['water_features'][start:start+8].reshape(-1,dim);x=x[np.all(np.isfinite(x),axis=1)]
                for k in range(dim):wcounts[k]+=np.histogram(x[:,k],bins=wedge[k])[0]
            store(wg,{'marginal_edges':wedge,'marginal_counts':wcounts,'marginal_density':wcounts/(wn*np.diff(wedge,axis=1))})
            wg.attrs['definition']='Valid instantaneous H2O centers; H2O-only neighbors; empirical marginals, not a complete high-dimensional free energy'
            rho=coll['rho_O'][...];rh=coll['rho_H'][...]
            store(coll.require_group('structure_factors'),{'OO_per_O':np.mean(abs(rho)**2,axis=0)/n,
                                                         'HH_per_H':np.mean(abs(rh)**2,axis=0)/(2*n),
                                                         'OH_cross_per_sqrt_NO_NH':np.real(np.mean(rho*rh.conj(),axis=0))/(np.sqrt(2)*n)})
            coll['structure_factors'].attrs['convention']='Includes same-molecule and self contributions; native nonzero box modes, not experimental weighted scattering'
            wc=coll['window_counts'][...].transpose(1,0,2).reshape(len(radii),-1)
            store(coll.require_group('window_statistics'),cumulants(wc))
            for ir,rad in enumerate(radii):
                counts,edges=np.histogram(wc[ir],bins=np.arange(-.5,max(1,int(wc[ir].max()))+1.5))
                store(coll.require_group(f'window_distributions/r{ir}'),{'edges':edges,'counts':counts,'radius_A':rad})
            report['frames_analyzed']=selected;report['water_molecules']=n;report['source_columns']=columns
            report['topology']={'mode':args.topology_mode,'bond_cut_A':args.bond_cut,
                                'nonwater_O_frame_count':int((~env['water_valid'][...]).sum()),
                                'unassigned_H_frame_count':int((env['h_owner'][...]<0).sum()),
                                'valid_feature_centers':total,
                                'meaning':'Nearest-O distance criterion, not electronic bond order. Invalid H2O orientations masked; raw H retained. O spatial statistics include all O. Hydrogen bonds use valid H2O donors and geometric O acceptors.'}
            report['feature_channels']=len(feature_names);report['voronoi_frames']=vor_frames
            report['periodicity']['minimum_unique_radius_A']=min(limits)
            report['periodicity']['unwrapped_source']='native xu/image flags' if all_have_images else 'minimum-image increments inferred from saved O frames; jumps > half cell unobservable'
            report['block_frames']=block_frames;report['elapsed_s']=time.monotonic()-starttime
            report['frame_dt_ps']=args.frame_dt_ps*args.stride
            # All centers in a saved frame share a role. This is a proposed
            # temporal partition, not proof that the embargo removes correlation.
            roles=np.zeros(selected,dtype=np.int8);cut1=int(selected*.6);cut2=int(selected*.8)
            roles[cut1:cut2]=1;roles[cut2:]=2
            for boundary in [cut1,cut2]:roles[max(0,boundary-args.split_gap_frames):min(selected,boundary+args.split_gap_frames)]=-1
            env.create_dataset('proposed_split_role',data=roles)
            env['proposed_split_role'].attrs['definition']='0 train, 1 validation, 2 test, -1 embargo; whole-frame groups; descriptive PCA uses ALL data and must be refit on training data for prediction'
            report['split_warning']='Proposed temporal split only; set gap based on measured correlations. Prefer whole-trajectory/state holdout. Global descriptive statistics are not training-fitted preprocessing.'
            report['density_g_cm3_from_18_01528_amu']=float(n*18.01528*1.66053906660/np.mean(np.linalg.det(env['cell'][...])))
            report['warnings']+=['No automatic equilibration removal; --start must reflect assessed equilibration.',
                                 'A short smoke run verifies execution, not 1 ns physical convergence.',
                                 'Blocks may be too short or unequal; block standard errors are estimates, not independent validation.',
                                 'Input force units not declared by dump; inspect original LAMMPS units before physical conversion.' if args.force_unit=='native_unknown' else 'Force units explicitly supplied by user configuration.',
                                 'High-temperature four-nearest q is geometric; q_first_shell_valid records whether all four lie inside first cutoff.']
            _finish_dynamics(env,coll,output,args,report,all_have_images)
        with h5py.File(output/'distributions.h5','w') as h5:
            hist.write(h5)
            h5.attrs['histogram_probability']='Conditional on finite values inside reported bin ranges; check n_finite and n_in_range'
            h5.attrs['block_frames']=block_frames
            h5.attrs['weighted_spatial_groups']='g is the pair-weighted mean product (NOT RDF) for spatial/*; inspect connected datasets'
            n=len(top.mol_ids);vol=report['density_g_cm3_from_18_01528_amu']/(18.01528*1.66053906660)
            if 'rdf/OO' in h5:
                diag=rdf_diagnostics(stats.r,h5['rdf/OO/g'][...],vol,vol*(n-1)/n);store(h5.require_group('rdf/OO/diagnostics'),diag)
            for label,key in [('q','q'),('coordination','coordination'),('hbond',None)]:
                with h5py.File(output/'environment.h5','r') as env:
                    values=env[key][...] if key else env['hbond_donors'][...]+env['hbond_acceptors'][...]
                    mean=np.nanmean(values)
                group=h5['spatial/product_'+label];group.create_dataset('connected_using_global_mean',data=group['g'][...]-mean**2)
                group.attrs['global_mean']=mean
            if 'triplet/r_r_cos' in h5 and 'rdf/OO' in h5:
                tg=h5['triplet/r_r_cos'];r1=(tg['edges_0'][:-1]+tg['edges_0'][1:])/2
                cs=(tg['edges_2'][:-1]+tg['edges_2'][1:])/2
                rjk=np.sqrt(np.maximum(0,r1[:,None,None]**2+r1[None,:,None]**2-2*r1[:,None,None]*r1[None,:,None]*cs[None,None,:]))
                rc=(stats.r[:-1]+stats.r[1:])/2;g=h5['rdf/OO/g'][...]
                g1=np.interp(r1,rc,g,left=np.nan,right=np.nan);g3=np.interp(rjk.ravel(),rc,g,left=np.nan,right=np.nan).reshape(rjk.shape)
                reference=g1[:,None,None]*g1[None,:,None]*g3
                tg.create_dataset('kirkwood_pair_product_reference',data=reference)
                tg.create_dataset('difference_from_pair_product',data=tg['g'][...]-reference)
                tg.attrs['warning']='Three-body position distribution, NOT direct c3; reference interpolated from finite-bin OO RDF'
        if not args.no_direct_correlations:
            from .correlations import write_correlations
            report['direct_correlations']=write_correlations(output,output/'direct_correlations.h5',args.oz_window,args.oz_tail_start,args.oz_min_s,
                                                           args.beta_mu,args.beta_mu_ex,args.reference_volume,report_override=report)
        from .coverage import coverage
        report['coverage']=coverage(report,output)
        report['status']='completed';report['elapsed_s']=time.monotonic()-starttime
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
        print(json.dumps({'status':'completed','output':str(output),'frames':selected,'features':len(feature_names),'seconds':round(report['elapsed_s'],1)},ensure_ascii=False),flush=True)
        return report
    except Exception as error:
        report['status']='failed';report['failure_type']=type(error).__name__;report['failure']=str(error)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
        raise


def _finish_dynamics(env,coll,output,args,report,all_have_images):
    with h5py.File(output/'dynamics.h5','w') as h5:
        if len(env['o'])<2:
            h5.attrs['status']='skipped: need at least 2 frames';return
        cells=env['cell'][...];positions=env['o'][...].astype(float);frames=env['molecular_frame'][...].astype(float)
        fixed=np.allclose(cells,cells[0],rtol=1e-6,atol=1e-6)
        if all_have_images:p=env['native_unwrapped_o'][...]
        elif fixed:p=inferred_nojump(positions,cells)
        else:
            h5.attrs['status']='skipped: variable cell without image flags';return
        result=motion(p,frames[:,:,0],env['time_ps'][...],args.max_lag)
        lags=result['lags_frames'];store(h5.require_group('oxygen_motion'),result)
        h5['oxygen_motion'].attrs['orientation_definition']='C1/C2 follow instantaneous H2O orientation anchored at the same O, permitting H exchange; not persistent-molecule rotation'
        store(h5.require_group('orientation/persistent_H2O'),persistent_water_orientation(frames[:,:,0],env['current_atom_ids'][...,1:],lags))
        h5['orientation/persistent_H2O'].create_dataset('lag_ps',data=result['lag_ps'])
        h5.attrs['unwrap_warning']=report['periodicity']['unwrapped_source']
        for key in ['q','coordination','lsi','hbond_donors','hbond_acceptors']:
            x=env[key][...]
            if np.any(np.isfinite(x)):
                store(h5.require_group('scalar/'+key),scalar_correlations(x,lags))
                ac=mean_autocorrelation(np.nanmean(x,axis=1));store(h5.require_group('sampling/'+key),ac)
                store(h5.require_group('sampling/'+key+'/block_scan'),block_scan(np.nanmean(x,axis=1),args.seed))
                half=len(x)//2
                store(h5.require_group('stationarity/'+key),{'first_half_mean':np.nanmean(x[:half]),'second_half_mean':np.nanmean(x[half:]),
                                                           'warning':'Descriptive split comparison, not a proof of equilibration'})
        store(h5.require_group('orientation/OH_identity_matched'),bond_orientation(env['oh'][...],env['current_atom_ids'][...,1:],lags))
        h5['orientation/OH_identity_matched'].create_dataset('lag_ps',data=result['lag_ps'])
        # H-exchange-odd plane normal: only even rank C2 is a physical invariant.
        normal=frames[:,:,2]
        plane=motion(p,normal,env['time_ps'][...],args.max_lag)
        store(h5.require_group('orientation/plane_normal'),{'c2':plane['c2'],'lag_ps':plane['lag_ps']})
        radii=np.linspace(0,min(30.,float(np.linalg.norm(cells[0],axis=1).min())),121)
        store(h5.require_group('self_van_hove'),displacement_distributions(p,lags,radii))
        # Sparse pair sets, never dense T*N*N arrays.
        neighbor_sets=[];second_sets=[];hb_sets=[]
        n=positions.shape[1]
        for frame in range(len(positions)):
            ng=env[f'neighbors/{frame:06d}'];ii=ng['center_index'][...];jj=ng['neighbor_index'][...];r=ng['distance'][...]
            neighbor_sets.append(set((ii[r<args.first_cut]*n+jj[r<args.first_cut]).tolist()))
            second=(r>=args.first_cut)&(r<args.env_cut)
            second_sets.append(set((ii[second]*n+jj[second]).tolist()))
            hb=ng['hbond_triplets'][...]
            hb_sets.append(set((hb[:,0]*n+hb[:,2]).tolist()))
        # Continuous-at-saved-frames estimates bounded to manageable lags;
        # the explicit cap is recorded, never advertised as a full lifetime.
        bondlags=lags[lags<=args.bond_max_lag]
        store(h5.require_group('hbond'),sampled_bond_correlations(hb_sets,bondlags))
        store(h5.require_group('first_shell'),sampled_bond_correlations(neighbor_sets,bondlags))
        store(h5.require_group('second_shell'),sampled_bond_correlations(second_sets,bondlags))
        h5['hbond'].create_dataset('lag_ps',data=bondlags*(env['time_ps'][1]-env['time_ps'][0]))
        h5['first_shell'].create_dataset('lag_ps',data=bondlags*(env['time_ps'][1]-env['time_ps'][0]))
        h5['second_shell'].create_dataset('lag_ps',data=bondlags*(env['time_ps'][1]-env['time_ps'][0]))
        if fixed:
            nm=min(12,coll['rho_O'].shape[1])
            store(h5.require_group('intermediate_scattering'),intermediate_scattering(p,cells,coll['rho_O'][...,:nm],coll['k_vectors'][0,:nm],lags))
        # Coarse D/A environment transitions; no assertion of Markovianity.
        nd=np.minimum(env['hbond_donors'][...],4);na=np.minimum(env['hbond_acceptors'][...],4);state=nd*5+na
        tg=h5.require_group('environment_transitions');tg.attrs['state_definition']='5*min(nD,4)+min(nA,4); descriptive counts, not validated Markov model'
        for lag in lags[lags>0][:8]:
            count=np.bincount((state[:-lag]*25+state[lag:]).ravel(),minlength=625).reshape(25,25)
            tg.create_dataset(f'lag_{lag}',data=count)
        # Finite-time angle distributions and translational/rotational cross statistics.
        ag=h5.require_group('lag_angle_distribution');cross=[];hbchange=[];crosscounts=[]
        for lag in lags:
            dots=np.clip(np.sum(frames[lag:,:,0]*frames[:len(p)-lag,:,0],axis=-1),-1,1)
            counts,edges=np.histogram(dots,bins=np.linspace(-1,1,81));ag.create_dataset(f'counts_lag_{lag}',data=counts)
            r2=np.sum((p[lag:]-p[:len(p)-lag])**2,axis=-1)
            angle=1-dots
            good=np.isfinite(angle)
            cross.append(np.mean(r2[good]*angle[good])-np.mean(r2[good])*np.mean(angle[good]) if good.any() else np.nan)
            crosscounts.append(int(good.sum()))
            htotal=env['hbond_donors'][...]+env['hbond_acceptors'][...]
            dh=abs(htotal[lag:]-htotal[:len(p)-lag])
            hbchange.append(np.mean(r2*dh)-np.mean(r2)*np.mean(dh))
        ag.create_dataset('cos_edges',data=edges);ag.create_dataset('lag_ps',data=result['lag_ps'])
        store(h5.require_group('cross_correlations'),{'translation_rotation_covariance':cross,'translation_rotation_valid_pairs':crosscounts,'displacement_HB_change_covariance':hbchange,'lag_ps':result['lag_ps']})
        h5.attrs['diffusion_fit']='No automatic diffusion fit; inspect linear MSD region and use explicit fit command'


def unit_local(v):
    norm=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.divide(v,norm,out=np.full_like(v,np.nan,dtype=float),where=norm>0)


def mic_local(v,cell):
    from .geometry import mic
    return mic(v,cell)
