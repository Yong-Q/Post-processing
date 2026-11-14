"""Retain user-admitted source trajectories; qualify perturbations separately.

Reuses immutable v1 per-frame caches. Temperature/pressure stability is a user
declaration: caches lack velocities and have virial-only pressure, so this code
does not claim an independent thermodynamic stability check.
"""
import argparse
import errno
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import traceback
import numpy as np
from .reweight import KB,statistical_inefficiency,log_weights
from .reweight_collection import (write_json,sha256,read_energy,pilot_case,
    generate_case,_screen_logq,clean)


def direction_monitor(data,index,joint=False):
    u=np.asarray(data['energy'])[:,None]
    if index is None:return u
    f=np.asarray(data['fields'])[:,index:index+1]
    return np.column_stack([u,f]) if joint else f


def direction_window(data,index,joint=False):
    monitor=direction_monitor(data,index,joint)
    g=max(statistical_inefficiency(v) for v in monitor.T)
    block=max(50,int(np.ceil(5*g)));n=len(monitor)
    used=(n//(4*block))*4*block;nb=used//block
    return {'status':'diagnostics_passed' if nb>=8 else 'unqualified',
        'reason':None if nb>=8 else 'direction_insufficient_complete_time_blocks',
        'start':0,'stop':used,'block_frames':block,'blocks':nb,
        'statistical_inefficiency_max':g,'monitor_scope':'energy' if index is None else ('energy_and_coupled_field' if joint else 'coupled_field'),
        'source_admission':'user_confirmed_temperature_pressure_stability',
        'source_stability_independently_verified':False,'burn_fraction':0.}


def _task(root,case,name,**extra):
    return {'case':case,'output':str(Path(root)/'groups'/name),'propertywise':True,
            'include_direct':name=='direct','admit_direct_source':name=='direct',
            'dataset_namespace':case+'/'+name,**extra}


def _make_group(root,case,name,cache,provenance,window):
    task=_task(root,case,name);folder=Path(task['output'])/case;folder.mkdir(parents=True)
    try:os.link(cache/'frames.npz',folder/'frames.npz');link_kind='hardlink'
    except OSError as error:
        if error.errno not in (errno.EXDEV,errno.EPERM,errno.EOPNOTSUPP):raise
        (folder/'frames.npz').symlink_to((cache/'frames.npz').resolve());link_kind='symlink'
    p={**provenance,'window':window,'qualification_policy':'2.0_propertywise',
        'source_admission':'user_confirmed_temperature_pressure_stability',
        'original_cache_global_screen':provenance.get('window'),
        'source_stability_independently_verified':False,'cache_link_kind':link_kind}
    write_json(folder/'provenance.json',p)
    return task


def prepare(task):
    root=Path(task['root']);case=task['case'];cache=Path(task['cache_root'])/case
    p=json.loads((cache/'provenance.json').read_text())
    report=json.loads((Path(task['source_root'])/'analysis'/p['species']/case/'report.json').read_text())
    if report['input_sha256']!=p['source_trajectory_sha256']:raise ValueError('Source/cache trajectory identity mismatch')
    with np.load(cache/'frames.npz',allow_pickle=False) as z:
        data={'energy':z['energy'],'fields':z['fields']};step=z['step']
    energy_dir=Path(task['energy_dir'])
    e=read_energy(energy_dir/(case+'.energy.csv'),energy_dir/'npz'/(case+'.npz'),step)
    if not np.array_equal(e['energy'],data['energy']) or e['csv_sha256']!=p['energy_csv_sha256']:
        raise ValueError('Reused cache/energy mismatch')
    if e['meta']['model_sha256']!=p['source_model_sha256']:raise ValueError('Cache/model mismatch')
    p={**p,'reused_cache':str(cache),'reused_cache_sha256':sha256(cache/'frames.npz')}
    n=len(step);direct_window={'status':'source_admitted_user_TP_stability','start':0,'stop':(n//200)*200,
        'block_frames':50,'blocks':(n//200)*4,'statistical_inefficiency_max':None,
        'scope':'Bookkeeping window only; direct values retained and labels checked separately; no global decorrelation factor assigned.',
        'source_stability_independently_verified':False}
    t=_make_group(root,case,'direct',cache,p,direct_window)
    write_json(Path(t['output'])/case/'perturbation_ranges.json',{'candidates_for_label_qualification':[]})
    windows={}
    for index in [None,*range(15)]:
        name='temperature' if index is None else f'field_{index:02d}'
        win=direction_window(data,index);windows[name]=win
        _make_group(root,case,name,cache,p,win)
    write_json(root/'source_admission'/f'{case}.json',{'case':case,'source_admitted':True,
        'basis':'User confirmed production trajectory temperature/pressure stability',
        'independent_TP_verification':False,'prior_global_screen':p['window'],'direction_windows':windows,
        'cache_sha256':p['reused_cache_sha256'],'source_trajectory_sha256':p['source_trajectory_sha256']})
    return {'case':case,'status':'prepared','source_admitted':True,'direction_windows':windows}


def pilot(task):
    root=Path(task['root']);case=task['case'];answers=[];ranges={}
    for index in [None,*range(15)]:
        name='temperature' if index is None else f'field_{index:02d}'
        scope={'kind':'temperature' if index is None else 'field','field_index':index}
        gt=_task(root,case,name,perturbation_scope=scope)
        answer=pilot_case(gt);answers.append(answer)
        ranges[name]=json.loads((Path(gt['output'])/case/'perturbation_ranges.json').read_text())
    # Joint points have their own coupling window and new support/noise screen.
    temp_ranges=[r for r in ranges['temperature']['ranges'] if r['holdout_confirmation']['pass']]
    cache=Path(task['cache_root'])/case
    p=json.loads((cache/'provenance.json').read_text())
    source_record=json.loads((root/'source_admission'/f'{case}.json').read_text())
    p={**p,'reused_cache':str(cache),'reused_cache_sha256':source_record['cache_sha256']}
    with np.load(cache/'frames.npz',allow_pickle=False) as z:data={'energy':z['energy'],'fields':z['fields']}
    t0=p['source_temperature_K'];beta0=1/(KB*t0);joint_answers=[]
    for index in range(15):
        field_ranges=[r for r in ranges[f'field_{index:02d}']['ranges'] if r['holdout_confirmation']['pass']]
        if not temp_ranges or not field_ranges:continue
        win=direction_window(data,index,joint=True)
        if win['status']!='diagnostics_passed':continue
        name=f'joint_{index:02d}';gt=_make_group(root,case,name,cache,p,win)
        sl=slice(win['start'],win['stop']);u=data['energy'][sl];f=data['fields'][sl,index]
        monitor=direction_monitor(data,index,True)[sl];candidates=[];tests=[]
        for tr in temp_ranges:
            beta=beta0+.5*(1/(KB*tr['target_temperature_K'])-beta0);target=1/(KB*beta)
            for fr in field_ranges:
                amp=.5*fr['amplitude_beta0']
                check=_screen_logq(log_weights(u,t0,target,f,amp),monitor,win['block_frames'])
                point={'kind':'joint','field_index':index,'target_temperature_K':target,'amplitude_beta0':amp}
                tests.append({**point,**check})
                if check['pass']:candidates.append(point)
        result={'case':case,'source_window':win,'ranges':[],'candidates_for_label_qualification':candidates,'pilot_tests':tests}
        write_json(Path(gt['output'])/case/'perturbation_ranges.json',result)
        joint_answers.append({'group':name,'candidate_count':len(candidates),'tests':tests,'source_window':win})
    result={'case':case,'status':'pilot_completed','source_admitted':True,'directions':answers,'joint_directions':joint_answers,
        'candidate_count':sum(a['candidate_count'] for a in answers)+sum(a['candidate_count'] for a in joint_answers)}
    write_json(root/'ranges'/f'{case}.json',result)
    return result


def generate(task):
    root=Path(task['root']);case=task['case'];answers=[]
    for name in ['direct','temperature',*[f'field_{i:02d}' for i in range(15)],*[f'joint_{i:02d}' for i in range(15)]]:
        folder=root/'groups'/name/case
        if not folder.exists():continue
        if name!='direct':
            data=json.loads((folder/'perturbation_ranges.json').read_text())
            if not data['candidates_for_label_qualification']:continue
        answer=generate_case(_task(root,case,name));answers.append({'group':name,**answer})
    result={'case':case,'status':'generated','groups':answers,'source_admitted':True,
        'states':sum(a.get('states',0) for a in answers),
        'perturbed_states_with_qualified_label':sum(a.get('perturbed_states_with_qualified_label',0) for a in answers)}
    write_json(root/'generated'/f'{case}.json',result)
    return result


def _run(stage,task):
    try:return {'prepare':prepare,'pilot':pilot,'generate':generate}[stage](task)
    except Exception as e:
        result={'case':task['case'],'stage':stage,'status':'failed','error':str(e),'traceback':traceback.format_exc()}
        write_json(Path(task['root'])/'failures'/f'{task["case"]}.{stage}.json',result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('cache-root','source-root','energy-dir','output-root'):parser.add_argument('--'+option,required=True)
    parser.add_argument('--workers',type=int,default=24);a=parser.parse_args(argv)
    root=Path(a.output_root).resolve();root.mkdir(parents=True,exist_ok=False)
    for sub in ('source_admission','ranges','generated','failures'):(root/sub).mkdir()
    cache=Path(a.cache_root);cases=sorted(p.parent.name for p in cache.glob('*/frames.npz'))
    if not cases:raise ValueError('No reusable frame caches')
    tasks=[{'case':case,'cache_root':str(cache.resolve()),'source_root':str(Path(a.source_root).resolve()),
        'energy_dir':str(Path(a.energy_dir).resolve()),'root':str(root)} for case in cases]
    workers=min(a.workers,int(os.environ.get('SLURM_CPUS_PER_TASK',a.workers)),len(tasks))
    if workers<1:raise ValueError('Positive worker count required')
    write_json(root/'run_configuration.json',{'arguments':vars(a),'policy':'propertywise_v2','source_admission':'user_confirmed_stable_TP',
        'independent_TP_verification':False,'workers':workers,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'execution_order':['prepare_reuse_cache','pilot_all_directions','publish_ranges','generate']})
    summary={};active=tasks
    for stage in ('prepare','pilot','generate'):
        answers=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(_run,stage,t) for t in active]):
                answer=f.result();answers.append(answer);print(stage,answer['case'],answer['status'],flush=True)
        summary[stage]=answers;write_json(root/(stage+'_summary.json'),answers)
        if stage=='pilot':write_json(root/'perturbation_ranges_all.json',{'cases':answers,'scope':'Per-coupling ranges; only label-specific diagnostics confer label eligibility.'})
        good={r['case'] for r in answers if r['status']!='failed'};active=[t for t in active if t['case'] in good]
    with (root/'catalog.jsonl').open('x') as stream:
        for path in sorted(root.glob('groups/*/*/manifest.json')):
            for state in json.loads(path.read_text()).get('states',[]):stream.write(json.dumps(clean(state),ensure_ascii=False)+'\n')
    summary['status']='completed_with_failures' if any(x['status']=='failed' for values in summary.values() for x in values) else 'completed'
    write_json(root/'collection_summary.json',summary)
    return int(summary['status']!='completed')
