"""Unified pure-species molecular postprocessing, retaining legacy water mode."""
import json
import re
import time
from pathlib import Path
import h5py
import numpy as np
from .io import read_dump
from .molecules import MolecularTopology,ELEMENTS
from .molecular_io import infer_species,parse_type_map
from .molecular_features import encode_molecules
from .molecular_statistics import MolecularStatistics
from .molecular_dynamics import write_molecular_dynamics
from .geometry import unique_radius,mic
from .static import periodic_voronoi
from .histograms import Histograms
from .rdf import periodic_oo_counts
from .pipeline import append,store,digest
from .correlations import write_correlations


def _feature_summary(dataset,group,names,max_samples=10000):
    T,N,D=dataset.shape;total=0;s=np.zeros(D);ss=np.zeros(D)
    low=np.full(D,np.inf);high=np.full(D,-np.inf);samples=[]
    stride=max(1,(T*N)//max_samples)
    for start in range(0,T,8):
        x=dataset[start:start+8].reshape(-1,D).astype(float);x=x[np.all(np.isfinite(x),axis=1)]
        if not len(x):continue
        pick=(np.arange(total,total+len(x))%stride)==0;samples.append(x[pick]);total+=len(x)
        s+=x.sum(axis=0);ss+=(x*x).sum(axis=0);low=np.minimum(low,x.min(axis=0));high=np.maximum(high,x.max(axis=0))
    group.create_dataset('names',data=np.asarray(names,dtype=h5py.string_dtype()))
    group.create_dataset('valid_centers',data=total)
    if not total:group.attrs['status']='no valid target molecules';return
    mean=s/total;std=np.sqrt(np.maximum(0,ss/total-mean*mean));scale=np.where(std>1e-12,std,1)
    sample=np.concatenate(samples)[:max_samples]
    store(group,{'mean':mean,'std':std,'covariance_sample_count':len(sample),'covariance_sample':np.cov(sample,rowvar=False) if len(sample)>1 else np.zeros((D,D))})
    group.attrs['covariance_sampling']='Uniform deterministic subsample of valid center rows, <=10000; means/std and marginals use all valid rows'
    const=high<=low;low[const]-=.5;high[const]+=.5
    edges=np.array([np.linspace(low[k],high[k],49) for k in range(D)]);counts=np.zeros((D,48),np.int64)
    for start in range(0,T,8):
        x=dataset[start:start+8].reshape(-1,D);x=x[np.all(np.isfinite(x),axis=1)]
        for k in range(D):counts[k]+=np.histogram(x[:,k],bins=edges[k])[0]
    store(group,{'marginal_edges':edges,'marginal_counts':counts,'marginal_density':counts/(total*np.diff(edges,axis=1))})
    if len(sample)>2:
        z=(sample-mean)/scale;_,singular,vt=np.linalg.svd(z,full_matrices=False)
        store(group,{'pca_components_sample':vt[:3],'pca_singular_values_sample':singular})
        projected=z@vt[:2].T
        if projected.shape[1]==2:
            c,a,b=np.histogram2d(projected[:,0],projected[:,1],bins=40)
            store(group.require_group('pca_joint_sample'),{'counts':c,'edges_0':a,'edges_1':b})
    group.attrs['warning']='Descriptive feature marginals/sample covariance/PCA, not a complete rho(xi) or free energy; training preprocessing must be fitted on training data only'


def analyze_molecular(args):
    source=Path(args.input);out=Path(args.output)
    if out.exists():raise ValueError('Output already exists; refusing overwrite')
    if not source.is_file():raise ValueError('Input dump missing')
    species=infer_species(source,args.species);type_map=parse_type_map(getattr(args,'type_map',None))
    if not args.no_voronoi:
        import freud
        freud.parallel.set_num_threads(1)
    if not 0<args.first_cut<=args.env_cut or min(args.rdf_max,args.lsi_cut,args.frame_dt_ps)<=0 or min(args.stride,args.pair_stride,args.expensive_stride,args.block_frames)<1:
        raise ValueError('Invalid cutoffs/time/stride')
    if args.nrad<1 or args.lmax<0 or args.bins<2:raise ValueError('Invalid descriptor basis/grid')
    if args.start<0 or (args.max_frames is not None and args.max_frames<1):raise ValueError('Invalid frame selection')
    out.mkdir(parents=True);started=time.monotonic()
    report={'status':'running','mode':'molecular','species':species,'input':str(source.resolve()),'input_sha256':digest(source),
            'model_id':getattr(args,'model_id',None),
            'configuration':vars(args).copy(),'center_definition':'heavy_skeleton_com','force_unit':args.force_unit,
            'warnings':['Pure configured species only; not an arbitrary mixture or changing heavy-skeleton reaction network.',
                        'Geometric H membership is not electronic bond order. Invalid target species are masked, raw sites retained.',
                        'Packing/LSI cutoffs are geometric analysis choices; inspect center RDF before interpreting shell labels.',
                        'No automatic equilibration removal; covariance/PCA descriptives do not establish physical convergence.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    hist=Histograms(args.block_frames)
    statistics=MolecularStatistics(hist,args.first_cut,args.env_cut,args.bins,args.hbond_cut,args.hbond_angle,args.lsi_cut)
    edges=np.linspace(0,args.rdf_max,args.bins+1);shell=4*np.pi/3*np.diff(edges**3)
    integer=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]])
    selected=0;top=None;names=None;volumes=[];invalid=0;heavy_invalid=0;orphan=0;previous_step=None;interval=None;k0=None
    rng=np.random.default_rng(args.seed);window_fractional=rng.random((args.window_centers,3))
    from .collective import window_counts,cumulants
    try:
        with h5py.File(out/'environment.h5','x') as env,h5py.File(out/'collective.h5','x') as collective:
            for index,frame in enumerate(read_dump(source)):
                if previous_step is not None:
                    diff=frame.step-previous_step
                    if diff<=0 or (interval is not None and diff!=interval):raise ValueError('Irregular dump timesteps')
                    interval=diff
                previous_step=frame.step
                if index<args.start or (index-args.start)%args.stride:continue
                if args.max_frames is not None and selected>=args.max_frames:break
                if top is None:
                    top=MolecularTopology(frame,species,type_map=type_map,mode=args.topology_mode,bond_cut=args.bond_cut)
                    for key,value in {'atom_ids':top.ids,'elements':top.element_codes,'molecule_ids':top.molecule_ids,'initial_owner':top.initial_owner}.items():env.create_dataset(key,data=value)
                    for key,value in {'species':species,'axis_kind':top.spec['orientation_kind'],'shape_lmax':args.lmax,'center_definition':'heavy_skeleton_com',
                                      'force_units':args.force_unit,'first_shell_cutoff_A':args.first_cut,'second_shell_cutoff_A':args.env_cut}.items():env.attrs[key]=value
                    env.attrs['element_channels']=json.dumps(ELEMENTS)
                    env.attrs['molar_mass_g_mol']=top.spec['molar_mass']
                    initial_columns=frame.columns
                elif frame.columns!=initial_columns:raise ValueError('Dump columns changed; native positions/forces must have consistent fields across frames')
                bound=unique_radius(frame.cell)
                if max(args.env_cut,args.rdf_max,args.lsi_cut,args.hbond_cut)>=bound:raise ValueError(f'Cutoff exceeds unique-image radius {bound:g} A')
                data=top.extract(frame);data['species']=species;N=len(data['centers'])
                expensive=selected%args.expensive_stride==0
                vor=None
                if expensive and not args.no_voronoi:
                    vor=periodic_voronoi(data['centers'],frame.cell)
                    if not np.isclose(vor['volume'].sum(),np.linalg.det(frame.cell),rtol=3e-5):raise ValueError('Voronoi volume does not tile periodic box')
                metrics=statistics.frame(data,frame.cell,selected,expensive=expensive,voronoi=vor)
                encoded=encode_molecules(data['centers'],data['positions'],data['elements'],data['owner'],frame.cell,data['molecule_valid'],args.env_cut,args.nrad,args.lmax)
                filtered=encoded if data['molecule_valid'].all() and np.all(data['owner']>=0) else encode_molecules(data['centers'],data['positions'],data['elements'],data['owner'],frame.cell,data['molecule_valid'],args.env_cut,args.nrad,args.lmax,True)
                values={'source_frame':index,'step':frame.step,'time_ps':index*args.frame_dt_ps,'cell':frame.cell,
                        'centers':data['centers'].astype('f4'),'com':data['com'].astype('f4'),'raw_com':data['raw_com'].astype('f4'),
                        'positions':data['positions'].astype('f4'),'owner':data['owner'].astype('i4'),'relative_positions':data['relative_positions'].astype('f4'),
                        'molecule_valid':data['molecule_valid'],'heavy_valid':data['heavy_valid'],'hydrogen_count':data['hydrogen_count'].astype('i4'),
                        'axis':data['axis'].astype('f4'),'plane_normal':data['plane_normal'].astype('f4'),
                        'attached_heavy_index':data['attached_heavy_index'].astype('i4'),
                        'features':encoded['features'].astype('f4'),'molecule_only_features':filtered['features'].astype('f4'),
                        'internal_shape_moments':encoded['shape_moments'].astype('c8'),'shape_strength':encoded['shape_strength'].astype('f4'),
                        'external_site_count':encoded['neighbor_counts'].astype('i4'),'valid_molecule_external_site_count':filtered['neighbor_counts'].astype('i4')}
                if frame.unwrapped is not None:
                    anchor=np.asarray([h[0] for h in top.heavy_indices])
                    values['native_unwrapped_centers']=frame.unwrapped[anchor]+mic(data['centers']-data['positions'][anchor],frame.cell)
                for key in ['coordination','second_shell','packing_q','lsi','nearest','radial_cv','hbond_donors','hbond_acceptors']:
                    if key in metrics:values[key]=np.asarray(metrics[key]).astype('f4')
                for key in ['forces','molecular_force','molecular_torque','raw_molecular_force','raw_molecular_torque']:
                    if key in data:values[key]=data[key].astype('f4')
                if not args.no_equivariant:values['equivariant_coefficients']=encoded['coefficients'].astype('c8')
                for key,value in values.items():append(env,key,value)
                if 'molecular_force' in data:
                    for label,vector in [('force',data['molecular_force']),('torque',data['molecular_torque'])]:
                        norm=np.linalg.norm(vector,axis=1)
                        hist.add(label+'/magnitude',norm[:,None],[np.linspace(0,100,args.bins+1)],selected)
                        hist.add(label+'/packing_q_joint',np.column_stack([metrics['packing_q'],norm]),[np.linspace(-3,1,args.bins+1),np.linspace(0,100,args.bins+1)],selected)
                if names is None:
                    names=encoded['names'];env.create_dataset('feature_names',data=np.asarray(names,dtype=h5py.string_dtype()))
                    env.attrs['feature_basis']=json.dumps({'nrad':args.nrad,'lmax':args.lmax,'cutoff_A':args.env_cut,'radial_centers':encoded['radial_centers'].tolist(),
                                                         'radial_width':encoded['radial_width'],'shape':'sum r^l Y_lm; raw units length^l','includes_center_orientation_coupling':True})
                neighbor_root=env.require_group('neighbors')
                neighbor_root.attrs['first_shell_cutoff_A']=args.first_cut;neighbor_root.attrs['second_shell_cutoff_A']=args.env_cut
                group=neighbor_root.create_group(f'{selected:06d}');ii,jj=np.where(metrics['distance']<args.env_cut)
                store(group,{'center_index':ii.astype('i4'),'neighbor_index':jj.astype('i4'),'distance':metrics['distance'][ii,jj].astype('f4'),
                             'displacement':metrics['delta'][ii,jj].astype('f4'),'hbond_triplets':metrics['hbond_triplets'].astype('i4')})
                geometry=env.require_group('geometry').create_group(f'{selected:06d}');store(geometry,data['geometry_samples'])
                if vor is not None:store(env.require_group('voronoi').create_group(f'{selected:06d}'),vor)
                if expensive:
                    from .static import steinhardt
                    eg=env.require_group('expensive').create_group(f'{selected:06d}')
                    for ell in [4,6]:
                        q,qbar=steinhardt(metrics['delta'],metrics['distance'],args.first_cut,ell)
                        store(eg,{f'packing_q{ell}':q,f'packing_q{ell}_bar':qbar})
                volume=float(np.linalg.det(frame.cell));volumes.append(volume)
                if selected%args.pair_stride==0:
                    hist.add_binned('rdf/centers',periodic_oo_counts(data['centers'],frame.cell,edges),[edges],selected,N*(N-1)/volume*shell,total_samples=N*(N-1))
                k=2*np.pi*integer@np.linalg.inv(frame.cell).T
                if k0 is None:
                    k0=k;collective.create_dataset('integer_modes',data=integer);collective.create_dataset('k_vectors',data=k[None])
                    window_radii=np.asarray([r for r in [1.5,2.,3.,4.,5.] if r<bound])
                    collective.create_dataset('window_radii_A',data=window_radii);collective.create_dataset('window_centers_fractional',data=window_fractional)
                elif not np.allclose(k,k0,rtol=1e-6,atol=1e-8):raise ValueError('Generic Fourier/dynamics output currently requires a fixed cell')
                mode=np.exp(-1j*(data['centers']@k.T)).sum(axis=0);append(collective,'rho_centers',mode)
                append(collective,'window_counts',window_counts(data['centers'],frame.cell,window_fractional,window_radii).astype('i4'))
                invalid+=int((~data['molecule_valid']).sum());heavy_invalid+=int((~data['heavy_valid']).sum());orphan+=int((data['owner']<0).sum())
                selected+=1
                if selected%100==0:print(f'{species} frames={selected} elapsed_s={time.monotonic()-started:.1f}',flush=True)
            if not selected:raise ValueError('No selected frames')
            _feature_summary(env['features'],collective.require_group('feature_statistics'),names)
            _feature_summary(env['molecule_only_features'],collective.require_group('molecule_feature_statistics'),names)
            rho_modes=collective['rho_centers'][...]
            collective.create_dataset('structure_factors/centers',data=np.mean(abs(rho_modes)**2,axis=0)/N)
            w=collective['window_counts'][...].transpose(1,0,2).reshape(len(window_radii),-1)
            store(collective.require_group('window_statistics'),cumulants(w))
            roles=np.zeros(selected,np.int8);a=int(.6*selected);b=int(.8*selected);roles[a:b]=1;roles[b:]=2
            for cut in [a,b]:roles[max(0,cut-args.split_gap_frames):min(selected,cut+args.split_gap_frames)]=-1
            env.create_dataset('proposed_split_role',data=roles)
            env['proposed_split_role'].attrs['meaning']='0 train,1 validation,2 test,-1 gap; whole-frame grouping only, independence not certified; descriptive statistics use all rows'
        with h5py.File(out/'distributions.h5','x') as f:
            hist.write(f);f.attrs['definitions']=json.dumps(statistics.descriptions);f.attrs['applicability']=json.dumps(statistics.applicability)
        density=N/float(np.mean(volumes));match=re.search(r'-T([0-9.]+)-',source.name)
        report.update(frames_analyzed=selected,number_of_centers=N,number_density_A_minus3=density,molar_mass_g_mol=top.spec['molar_mass'],
                      density_g_cm3=density*top.spec['molar_mass']*1.66053906660,temperature_K=float(match.group(1)) if match else None,
                      temperature_source='filename label, not a measured temperature log',source_columns=frame.columns,
                      center_rdf_group='rdf/centers',center_structure_factor_group='structure_factors/centers',
                      feature_channels=len(names),block_frames=args.block_frames,applicability=statistics.applicability,
                      invalid_target_molecule_frames=invalid,invalid_heavy_skeleton_frames=heavy_invalid,unassigned_atom_frames=orphan)
        report['dynamics']=write_molecular_dynamics(out/'environment.h5',out/'collective.h5',out/'dynamics.h5',
                                                   max_lag=args.max_lag if args.max_lag is not None else 100,bond_max_lag=args.bond_max_lag)
        if not args.no_direct_correlations:
            report['direct_correlations']=write_correlations(out,out/'direct_correlations.h5',args.oz_window,args.oz_tail_start,args.oz_min_s,
                                                             args.beta_mu,args.beta_mu_ex,args.reference_volume,report_override=report)
        report.update(status='completed',elapsed_s=time.monotonic()-started)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':'completed','species':species,'frames':selected,'features':len(names),'seconds':round(report['elapsed_s'],1),'output':str(out)}),flush=True)
        return report
    except Exception as error:
        report.update(status='failed',failure_type=type(error).__name__,failure=str(error),frames_analyzed=selected)
        (out/'report.json').write_text(json.dumps(report,indent=2));raise
