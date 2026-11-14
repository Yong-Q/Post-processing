"""One-source, fixed-T feasibility test of sum_i Zi K as a functional input.

Training is finite-moment canonical Legendre matching, not a claim that the
ordinary density c1/c2 labels are derivatives with respect to xi.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import traceback
import h5py
import numpy as np
from .xi_field import probe_features,field_3d,moment_observables
from .reweight_collection import write_json
from .reweight import weights,overlap_diagnostics,statistical_inefficiency,block_estimate,paired_noise
from .geometry import mic
from .molecular_features import encode_molecules


PROBE_NAMES=['radial_O_r1','env_power_Hr1_Hr1_l2','coupling_H_H_r1_l1']
SPLITS={'train':(0,1000),'validation':(1100,1500),'test':(1600,2000)}


def standardize(raw,mean,scale):
    transformed=np.sign(raw)*np.log1p(abs(raw))
    return (transformed-mean)/scale


def prepare(source,root):
    source=Path(source);root=Path(root);report=json.loads((source/'report.json').read_text())
    with h5py.File(source/'environment.h5','r') as h:
        n=len(h['step']);step=h['step'][:];cell=h['cell'][0]
        basis=json.loads(h.attrs['feature_basis'])
        if basis['nrad']!=3 or basis['lmax']!=3:raise ValueError('Pilot probe indices require nrad=3,lmax=3')
        cutoff=float(basis['cutoff_A'])
        if n<2001 or not np.allclose(h['cell'][:],cell):raise ValueError('Pilot needs full fixed-cell trajectory')
        if not np.allclose(cell,np.diag(np.diag(cell))):raise ValueError('Current prototype declares an orthogonal source box')
        centers=h['centers'][:].astype(float);raw=[]
        for start in range(0,n,8):
            raw.extend(probe_features(h['equivariant_coefficients'][start:start+8],h['internal_shape_moments'][start:start+8]))
        raw=np.asarray(raw,float)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite generic raw descriptor')
        logged=np.sign(raw[:1000])*np.log1p(abs(raw[:1000]));mean=logged.mean(axis=(0,1));scale=logged.std(axis=(0,1))
        if np.any(scale<1e-10):raise ValueError('Descriptor channel has no training variability')
        standardized=standardize(raw,mean,scale)
        z=np.concatenate([np.ones((*raw.shape[:2],1)),standardized],axis=-1)
        sigma=.12;obs=np.asarray([moment_observables(r,zi,cell,sigma) for r,zi in zip(centers,z)])
        # Frozen readout reconstruction agrees with columns of saved 508 basis.
        valid=h['molecule_valid'][0].astype(bool);names=h['feature_names'].asstr()[:]
        idx=[list(names).index(name) for name in PROBE_NAMES]
        saved=h['features'][0][:,idx].astype(float)
        reconstruction=np.max(abs(raw[0][valid]-saved[valid]))
        if not np.allclose(raw[0][valid],saved[valid],atol=2e-4,rtol=2e-5):raise ValueError('Raw descriptor/508 basis mismatch')
        # Counterfactual geometry only: same centers and rigidly rotated H
        # groups; no energy or equilibrium labels assigned to this structure.
        pos=h['positions'][0].astype(float);owner=h['owner'][0].astype(int);elements=h['elements'][:]
        attached=h['attached_heavy_index'][0].astype(int)
        changed=pos.copy();selected=np.flatnonzero((elements==1)&(attached>=0))
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        relative=mic(pos[selected]-pos[attached[selected]],cell)
        changed[selected]=pos[attached[selected]]+relative@rotation.T
        baseline_encoded=encode_molecules(centers[0],pos,elements,owner,cell,valid=None,cutoff=cutoff,nrad=3,lmax=3)
        encoded=encode_molecules(centers[0],changed,elements,owner,cell,valid=None,cutoff=cutoff,nrad=3,lmax=3)
        baseline_raw=probe_features(baseline_encoded['coefficients'],baseline_encoded['shape_moments'])
        rotated_raw=probe_features(encoded['coefficients'],encoded['shape_moments'])
        rotated_z=np.column_stack([np.ones(len(centers[0])),standardize(rotated_raw,mean,scale)])
        baseline_z=np.column_stack([np.ones(len(centers[0])),standardize(baseline_raw,mean,scale)])
        original_field=field_3d(centers[0],baseline_z,cell,8,sigma)
        rotated_field=field_3d(centers[0],rotated_z,cell,8,sigma)
        density_difference=float(np.max(abs(original_field[...,0]-rotated_field[...,0])))
        environment_difference=float(np.linalg.norm(original_field[...,1:]-rotated_field[...,1:]))
        relative_environment_difference=environment_difference/max(float(np.linalg.norm(original_field[...,1:])),1e-20)
        if density_difference>1e-12 or relative_environment_difference<1e-4:raise ValueError('Same-density pose sensitivity check failed')
    np.savez_compressed(root/'observables.npz',moments=obs,step=step,descriptor_mean=mean,descriptor_scale=scale,
        same_density_original_field=original_field,same_density_rotated_field=rotated_field,cell=cell)
    record={'source_results':str(source.resolve()),'source_input':report['input'],'source_input_sha256':report['input_sha256'],
        'source_species':report['species'],'source_temperature_K':report['temperature_K'],'source_frames':n,
        'descriptor_names':PROBE_NAMES,'descriptor_basis':'raw 508-basis coefficients; signed log and train-only atom/frame normalization',
        'descriptor_raw_units':['dimensionless','dimensionless','Angstrom'],
        'descriptor_signed_log_reference_values':[1.,1.,1.],
        'number_of_centers':len(centers[0]),'volume_A3':float(abs(np.linalg.det(cell))),
        'source_feature_basis':basis,
        'Zi_channels':4,'field_domain':'R3 with four channels','training_geometry':'planar projection; first Fourier mode, plus environment globals',
        'moment_dimensions':obs.shape[1],'kernel_sigma_fractional':sigma,'attenuation':float(np.exp(-2*np.pi**2*sigma*sigma)),
        'splits':SPLITS,'seed':42,'kernel_fractional_product_not_general_isotropic_metric':True,
        'raw_reconstruction_max_error':float(reconstruction),'same_density_max_difference':density_difference,
        'same_density_environment_field_L2_difference':environment_difference,
        'same_density_environment_field_relative_difference':relative_environment_difference,
        'normalization_train_frames':[0,1000],'source_independent_target_simulations':False,
        'atomistic_smoothness':'Ownership can change discontinuously at H transfer; no atomistic force derivative claimed',
        'scope':'Frozen static input representation and finite-dimensional Legendre response feasibility, not complete xi-cDFT'}
    write_json(root/'representation_checks.json',record)
    return record


def field_configurations(train_density,attenuation):
    definitions={};angles={'train':(0.,.5*np.pi,np.pi,1.5*np.pi),
        'validation':(.25*np.pi,.75*np.pi,1.25*np.pi,1.75*np.pi),
        'test':(.125*np.pi,.625*np.pi,1.125*np.pi,1.625*np.pi)}
    levels={'train':(.025,.075,.15),'validation':(.05,.10),'test':(.04,.12)}
    for split in angles:
        result=[{'id':split+'_zero','beta_field_coefficients':[0.,0.],'log_weight_scale':0.}]
        for angle in angles[split]:
            direction=np.array([np.cos(angle),np.sin(angle)])
            raw=train_density@direction/attenuation;std=float(np.std(raw,ddof=1))
            for level in levels[split]:
                result.append({'id':f'{split}_phase{angle:.4f}_scale{level:.3f}',
                    'beta_field_coefficients':(direction*level/std).tolist(),'log_weight_scale':level})
        definitions[split]=result
    return definitions


def state_statistics(raw,configs,mean,s,inverse,attenuation):
    d=raw.shape[1];states=[];white=(raw-mean)@inverse;n=len(raw)
    base_mean=white.mean(axis=0);base_fluct=white-base_mean;base_cov=base_fluct.T@base_fluct/n
    base_influence=np.column_stack([base_fluct,(np.einsum('ni,nj->nij',base_fluct,base_fluct)-base_cov).reshape(n,-1)])
    base_g=np.array([statistical_inefficiency(v) for v in base_influence.T])
    moments=np.column_stack([white,np.einsum('ni,nj->nij',white,white).reshape(n,-1)])
    def mean_cov(v):
        m=v[:d];return np.r_[m,(v[d:].reshape(d,d)-np.outer(m,m)).ravel()]
    for config in configs:
        bias=np.zeros(d);bias[:2]=np.asarray(config['beta_field_coefficients'])/attenuation
        # Exactly -beta Uext; kernel attenuation is explicitly removed.
        logq=-(raw-mean)@bias;w=weights(logq)
        check=overlap_diagnostics(w,50)
        if not check['pass']:
            states.append({'configuration':config,'status':'overlap_rejected','overlap':check});continue
        m=w@white;fluct=white-m;cov=(fluct*w[:,None]).T@fluct
        eigen=np.linalg.eigvalsh(cov)
        if eigen[0]<=0 or eigen[-1]/eigen[0]>1e8:
            states.append({'configuration':config,'status':'covariance_rejected','overlap':check});continue
        target_influence=np.column_stack([fluct,(np.einsum('ni,nj->nij',fluct,fluct)-cov).reshape(n,-1)])*w[:,None]*n
        g=max(float(base_g.max()),max(statistical_inefficiency(v) for v in target_influence.T))
        required=max(50,int(np.ceil(5*g)))
        choices=[b for b in range(required,n//8+1) if n%b==0]
        block=choices[0] if choices else 50
        check=overlap_diagnostics(w,block)
        if not check['pass']:
            states.append({'configuration':config,'status':'block_overlap_rejected','overlap':check});continue
        base_estimate=block_estimate(moments,np.ones(n),block,mean_cov)
        target_estimate=block_estimate(moments,w,block,mean_cov)
        noise,_=paired_noise(base_estimate,target_estimate)
        if not noise['pass']:
            states.append({'configuration':config,'status':'added_noise_rejected','overlap':check,'noise':noise});continue
        masses=w.reshape(n//block,block).sum(axis=1)
        sums=(white*w[:,None]).reshape(n//block,block,d).sum(axis=1)
        leave=(m-sums)/(1-masses[:,None]);nb=len(leave)
        density_leave=leave@s[:,:2]/attenuation
        density_se=np.sqrt((nb-1)/nb*np.sum((density_leave-density_leave.mean(axis=0))**2,axis=0))
        states.append({'configuration':config,'status':'accepted_for_exploratory_fit','mean':m,'covariance':cov,
            'gradient':-s.T@bias,'raw_density_mean':(w@raw[:,:2])/attenuation,'density_stderr':density_se,
            'overlap':check,'paired_mean_covariance_added_noise':noise,'error_block_frames':block,
            'max_mean_and_covariance_influence_g':g,'independent_block_count_heuristic':float(n/max(50,5*g)),
            'sampling_diagnostics_passed':bool(choices),
            'strict_sampling_qualification':False,
            'note':'Finite-source empirical statistics; paired mean/covariance noise checked. If no adequate block size fits, training remains exploratory. No independent target validation.'})
    return states


def dataset_arrays(states):
    accepted=[x for x in states if x['status']=='accepted_for_exploratory_fit']
    if len(accepted)<3:raise ValueError('Too few overlap/noise-supported finite-covariance states for a meaningful pilot')
    return accepted,np.stack([x['mean'] for x in accepted]),np.stack([x['gradient'] for x in accepted]),np.stack([x['covariance'] for x in accepted])


def fit_one(mode,root,epochs=200,device_name='cpu'):
    root=Path(root);folder=root/mode;folder.mkdir()
    try:
        import torch
        from .xi_model import ConvexMomentFreeEnergy,derivatives,whiten,density_response
        torch.set_num_threads(1);torch.manual_seed(42)
        device=torch.device(device_name)
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; CPU fallback prohibited')
        with np.load(root/'observables.npz') as f:raw=f['moments'][:,:2] if mode=='density_only' else f['moments'][:]
        meta=json.loads((root/'representation_checks.json').read_text());lam=meta['attenuation']
        mean,s,inverse=whiten(raw[:1000]);defs=json.loads((root/'field_configurations.json').read_text())
        stats={name:state_statistics(raw[start:stop],defs[name],mean,s,inverse,lam) for name,(start,stop) in SPLITS.items() if name!='test'}
        # Test statistics are archived only after checkpoint selection below.
        train,x,g,c=dataset_arrays(stats['train']);val,vx,vg,vc=dataset_arrays(stats['validation'])
        as_t=lambda a:torch.tensor(a,dtype=torch.float64,device=device)
        tx,tg,tc=map(as_t,(x,g,c));tvx,tvg,tvc=map(as_t,(vx,vg,vc))
        model=ConvexMomentFreeEnergy(raw.shape[1],16).to(device=device,dtype=torch.float64);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        identity=torch.eye(raw.shape[1],dtype=torch.float64,device=device)
        def loss(xi,gi,ci,training):
            _,gradient,hessian=derivatives(model,xi.detach().requires_grad_(True),create_graph=training)
            first=(gradient-gi).square().mean();second=(hessian@ci-identity).square().mean()
            return first+second,first,second
        history=[];best=float('inf');checkpoint=None;best_epoch=None
        for epoch in range(epochs+1):
            if epoch:
                optimizer.zero_grad();value,first,second=loss(tx,tg,tc,True);value.backward();optimizer.step()
            if epoch%20==0 or epoch==epochs:
                vl,vf,vs=loss(tvx,tvg,tvc,False)
                history.append({'epoch':epoch,'validation_loss':float(vl.detach()),'validation_gradient_loss':float(vf.detach()),'validation_response_loss':float(vs.detach())})
                if float(vl.detach())<best:best=float(vl.detach());checkpoint=copy.deepcopy(model.state_dict());best_epoch=epoch
        if checkpoint is None:raise ValueError('No finite validation checkpoint')
        model.load_state_dict(checkpoint)
        torch.save({'state_dict':checkpoint,'mode':mode,'dimensions':raw.shape[1],'width':16,'mean':as_t(mean),'s':as_t(s),'inverse':as_t(inverse),
            'best_epoch':best_epoch,'scope':'finite-moment total Legendre free energy, not density Fex'},folder/'best.pt')
        # The final test is now evaluated once after model selection.
        start,stop=SPLITS['test'];stats['test']=state_statistics(raw[start:stop],defs['test'],mean,s,inverse,lam)
        tests,xx,gg,cc=dataset_arrays(stats['test']);test_loss,tfirst,tsecond=loss(as_t(xx),as_t(gg),as_t(cc),False)
        predictions=[]
        for state in tests:
            conjugate=as_t(state['gradient']);point=torch.zeros((1,raw.shape[1]),dtype=torch.float64,device=device)
            converged=False
            for iteration in range(40):
                point=point.detach().requires_grad_(True);value,grad,hess=derivatives(model,point)
                residual=grad[0]-conjugate
                if float(torch.linalg.norm(residual).detach())<1e-9:converged=True;break
                step=torch.linalg.solve(hess.detach()[0],residual.detach())
                objective=float((value-point@conjugate).detach()[0]);accepted_step=False
                for scale_step in (1.,.5,.25,.125,.0625):
                    candidate=point.detach()-scale_step*step[None]
                    candidate_objective=float((model(candidate)-candidate@conjugate).detach()[0])
                    if candidate_objective<objective:
                        point=candidate;accepted_step=True;break
                if not accepted_step:break
            point=point.detach().requires_grad_(True);_,final_gradient,hess=derivatives(model,point)
            residual_norm=float(torch.linalg.norm(final_gradient[0]-conjugate).detach())
            converged=residual_norm<1e-9
            raw_prediction=mean+point.detach().cpu().numpy()[0]@s
            pred_density=raw_prediction[:2]/lam;reference=np.asarray(state['raw_density_mean'])
            pred_cov=density_response(s,hess.detach().cpu().numpy()[0],lam)
            target_cov=s[:2,:]@np.asarray(state['covariance'])@s[:2,:].T/lam**2
            quadratic_density=(mean+np.asarray(state['gradient'])@s)[:2]/lam
            quadratic_cov=s[:2,:]@s[:2,:].T/lam**2
            predictions.append({'configuration':state['configuration'],'converged':converged,'iterations':iteration+1,'stationarity_residual_norm':residual_norm,
                'density_mean_prediction':pred_density,'density_mean_reference':reference,'density_reference_stderr':state['density_stderr'],
                'density_covariance_prediction':pred_cov,'density_covariance_reference':target_cov,
                'density_mean_error_L2':float(np.linalg.norm(pred_density-reference)),
                'density_covariance_relative_error':float(np.linalg.norm(pred_cov-target_cov)/np.linalg.norm(target_cov)),
                'quadratic_reference_density_mean_error_L2':float(np.linalg.norm(quadratic_density-reference)),
                'quadratic_reference_density_covariance_relative_error':float(np.linalg.norm(quadratic_cov-target_cov)/np.linalg.norm(target_cov)),
                'moment_component_bounds_pass':bool(np.all(raw_prediction>=raw[:1000].min(axis=0)) and np.all(raw_prediction<=raw[:1000].max(axis=0))),
                'full_moment_representability_verified':False,
                'strict_sampling_qualification':state['strict_sampling_qualification']})
        converged_predictions=[p for p in predictions if p['converged']]
        result={'mode':mode,'status':'completed','input_moment_dimensions':raw.shape[1],
            'torch_version':torch.__version__,'device':str(device),
            'accelerator_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'validation_best':best,
            'test_gradient_loss':float(tfirst.detach()),'test_response_loss':float(tsecond.detach()),
            'mean_density_mean_error_L2':float(np.mean([v['density_mean_error_L2'] for v in converged_predictions])) if converged_predictions else None,
            'mean_density_covariance_relative_error':float(np.mean([v['density_covariance_relative_error'] for v in converged_predictions])) if converged_predictions else None,
            'error_averages_scope':'converged stationary solutions only; failed solutions retained separately in predictions',
            'test_converged':sum(v['converged'] for v in predictions),'test_cases':len(predictions),'predictions':predictions,
            'scope':'Single-source finite-dimensional feasibility; compare only common density metrics, not unequal-dimensional training losses',
            'whitening_embeds_measured_source_covariance':True,
            'learned_continuous_translation_symmetry_enforced':False,
            'off_manifold_environment_free_energy_validated':False,'independent_target_validation':False}
        write_json(folder/'training_history.json',history);write_json(folder/'source_statistics.json',stats);write_json(folder/'result.json',result)
        return result
    except Exception as error:
        result={'mode':mode,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        write_json(folder/'failure.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=200);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=parser.parse_args(argv)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    try:
        record=prepare(a.source,root)
        with np.load(root/'observables.npz') as z:obs=z['moments'][:1000,:2]
        definitions=field_configurations(obs,record['attenuation']);write_json(root/'field_configurations.json',definitions)
        print('representation checks passed; start density/xi fits',flush=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fit_one,mode,str(root),a.epochs,a.device) for mode in ('density_only','xi')]
            results=[f.result() for f in futures]
        status='completed' if all(v['status']=='completed' for v in results) else 'completed_with_failures'
        write_json(root/'comparison.json',{'status':status,'representation_checks':record,'models':results,
            'scope':'Input/derivative/finite-response prototype only; no proof of full xi functional, transfer, or independent external-state prediction'})
        print(status,flush=True);return int(status!='completed')
    except Exception as error:
        write_json(root/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
