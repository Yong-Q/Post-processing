"""Staged collection processing; one immutable result directory per case."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .molecules import SPECS
from .pipeline import digest

ROOT=Path(__file__).resolve().parents[1]


def discover(input_dir):
    cases=[]
    for p in sorted(Path(input_dir).glob('*.dump.gz')):
        match=re.fullmatch(r'([A-Za-z0-9]+)-T([0-9.]+)-(r\d+)-rho([0-9.]+)\.dump\.gz',p.name)
        if not match or match[1] not in SPECS:raise ValueError('Unrecognized pure-species trajectory: '+p.name)
        cases.append({'id':p.name.removesuffix('.dump.gz'),'source':str(p.resolve()),'species':match[1],
                      'temperature_K':float(match[2]),'state_label':match[3],'density_label':float(match[4])})
    if not cases:raise ValueError('No .dump.gz inputs')
    return cases


def first_minimum(rdf_dir):
    with h5py.File(Path(rdf_dir)/'distributions.h5') as f:
        edges=f['rdf/centers/edges_0'][...];g=f['rdf/centers/g'][...]
    r=(edges[:-1]+edges[1:])/2;sm=gaussian_filter1d(g,2.)
    peaks,_=find_peaks(sm,prominence=.08,height=1.02)
    peaks=peaks[(r[peaks]>1.5)&(r[peaks]<7.)]
    if not len(peaks):raise ValueError('No resolved first RDF peak')
    minima,_=find_peaks(-sm,prominence=.01)
    minima=minima[(minima>peaks[0])&(r[minima]<min(r[peaks[0]]+3.5,edges[-1]-.5))]
    if not len(minima):raise ValueError('No resolved first RDF minimum')
    return float(r[minima[0]])


def choose_window(local_rdf,reference_rdf,env_cut=8.5):
    """Explicit reference fallback for weakly structured low-density states."""
    reason=None
    try:
        cut=first_minimum(local_rdf)
        if cut>=env_cut:raise ValueError('Local minimum lies outside common encoder cutoff')
        return {'first_cut_A':cut,'source':'local_RDF_minimum','reference_results':str(local_rdf),'warning':'Geometric analysis window; inspect physical shell interpretation'}
    except ValueError as e:reason=str(e)
    cut=first_minimum(reference_rdf)
    if cut>=env_cut:raise ValueError('Reference minimum also outside common encoder cutoff; manual window needed')
    return {'first_cut_A':cut,'source':'species_temperature_reference_RDF','reference_results':str(reference_rdf),
            'local_reason':reason,'warning':'No resolved local shell; fixed reference geometric window, not a measured minimum at this state'}


def _write_atomic(path,data):
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}.{threading.get_ident()}')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--phase',choices=['all','rdf','analysis','c1'],default='all')
    p.add_argument('--resume',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--env-cut',type=float,default=8.5);p.add_argument('--nrad',type=int,default=3);p.add_argument('--lmax',type=int,default=3)
    p.add_argument('--analysis-frames',type=int);p.add_argument('--model-id',default='MACELES_record24123911_same_Hamiltonian_assumed')
    a=p.parse_args(argv);cases=discover(a.input_dir);output=Path(a.output_root).resolve()
    if a.workers<1:raise ValueError('workers must be positive')
    allocation=int(os.environ.get('SLURM_CPUS_PER_TASK',a.workers))
    workers=min(a.workers,allocation,len(cases))
    if output.exists() and not a.resume:raise ValueError('Output root exists; use a new root or --resume')
    output.mkdir(parents=True,exist_ok=True);started=time.time();lock=threading.Lock()
    state={'status':'running','phase':'preparation','input_cases':len(cases),'workers':workers,'cases':cases,'stages':{},
           'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'hostname':os.uname().nodename,
           'common_basis':{'env_cut_A':a.env_cut,'nrad':a.nrad,'lmax':a.lmax},'equilibration':'No automatic removal; source frames preserved'}
    existing=output/'progress.json'
    if a.resume and existing.exists():state['stages']=json.loads(existing.read_text()).get('stages',{})
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def progress():
        with lock:_write_atomic(output/'progress.json',state)
    def job(stage,key,args,base,completed_status='completed'):
        identity=hashlib.sha256(json.dumps(args).encode()).hexdigest()
        marker=base.parent/(base.name+'.job.json');report=base/'report.json'
        record_key=stage+':'+key
        prior=state['stages'].get(record_key,{})
        if prior.get('status')=='completed' and prior.get('output'):
            candidate=Path(prior['output']);rp=candidate/'report.json';mk=candidate.parent/(candidate.name+'.job.json')
            if rp.exists() and mk.exists() and json.loads(rp.read_text()).get('status')==completed_status and json.loads(mk.read_text()).get('arguments_sha256')==identity:
                result={'status':'completed','reused':True,'output':str(candidate)}
                with lock:state['stages'][record_key]=result
                progress();return result
        if base.exists():
            previous=json.loads(report.read_text()) if report.exists() else {}
            old_marker=json.loads(marker.read_text()) if marker.exists() else {}
            if previous.get('status')==completed_status and old_marker.get('arguments_sha256')==identity:
                result={'status':'completed','reused':True,'output':str(base)}
                with lock:state['stages'][record_key]=result
                progress();return result
            if not a.retry_failed:raise ValueError('Preserving existing partial/different result: '+str(base))
            number=2
            while base.with_name(base.name+f'.attempt{number:02d}').exists():number+=1
            base=base.with_name(base.name+f'.attempt{number:02d}');marker=base.parent/(base.name+'.job.json')
        base.parent.mkdir(parents=True,exist_ok=True)
        log=base.parent/(base.name+'.log')
        if log.exists():raise ValueError('Preserving existing log '+str(log))
        marker.write_text(json.dumps({'arguments_sha256':identity,'arguments':args,'input_code_root':str(ROOT)},indent=2))
        with lock:state['stages'][record_key]={'status':'running','output':str(base),'log':str(log)}
        progress();print('START',stage,key,flush=True)
        with log.open('x') as stream:
            proc=subprocess.run([sys.executable,'-m','waterpost',*args,'--output',str(base)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result={'status':'completed' if proc.returncode==0 else 'failed','exit_code':proc.returncode,'output':str(base),'log':str(log)}
        with lock:state['stages'][record_key]=result
        progress();print('DONE' if proc.returncode==0 else 'FAILED',stage,key,flush=True)
        return result
    def phase(name,items):
        state['phase']=name;progress();answers={}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(job,*item):key for key,item in items.items()}
            for f in as_completed(futures):
                key=futures[f]
                try:answers[key]=f.result()
                except Exception as error:
                    answers[key]={'status':'failed','error_type':type(error).__name__,'error':str(error)}
                    with lock:state['stages'][name+':'+key]=answers[key]
                    progress();print('FAILED',name,key,type(error).__name__,flush=True)
        return answers
    progress();rdf_results={}
    if a.phase in ['all','rdf']:
        items={}
        for c in cases:
            args=['rdf-only','--species',c['species'],'--input',c['source'],'--rdf-max','12','--bins','240','--block-frames','100']
            items[c['id']]=('rdf',c['id'],args,output/'rdf'/c['species']/c['id'])
        rdf_results=phase('rdf',items)
    else:
        for c in cases:rdf_results[c['id']]=state['stages'].get('rdf:'+c['id'],{'status':'missing'})
    groups=defaultdict(list)
    for c in cases:
        if rdf_results.get(c['id'],{}).get('status')=='completed':groups[(c['species'],c['temperature_K'])].append(c)
    windows={}
    for group,rows in groups.items():
        references=sorted(rows,key=lambda c:(c['state_label']!='r03',-c['density_label']))
        reference=None
        for c in references:
            path=Path(rdf_results[c['id']]['output'])
            try:
                if first_minimum(path)<a.env_cut:reference=path;break
            except ValueError:pass
        for c in rows:
            local=Path(rdf_results[c['id']]['output'])
            try:windows[c['id']]=choose_window(local,reference or local,a.env_cut)
            except ValueError as e:windows[c['id']]={'status':'requires_manual_window','reason':str(e)}
    _write_atomic(output/'analysis_windows.json',windows)
    if a.phase in ['all','analysis']:
        items={}
        for c in cases:
            window=windows.get(c['id'],{})
            if 'first_cut_A' not in window:
                state['stages']['analysis:'+c['id']]={'status':'skipped','reason':'No usable local or same-species/T reference RDF window'};continue
            cut=window['first_cut_A'];args=['analyze','--species',c['species'],'--input',c['source'],
                '--env-cut',str(a.env_cut),'--nrad',str(a.nrad),'--lmax',str(a.lmax),'--first-cut',str(cut),'--lsi-cut',str(cut+.2),
                '--rdf-max','12','--bins','80','--block-frames','100','--pair-stride','10','--expensive-stride','10',
                '--max-lag','100','--bond-max-lag','20','--window-centers','32']
            if a.analysis_frames is not None:args+=['--max-frames',str(a.analysis_frames)]
            items[c['id']]=('analysis',c['id'],args,output/'analysis'/c['species']/c['id'])
        phase('analysis',items)
    if a.phase in ['all','c1']:
        items={};excluded={}
        for (species,T),rows in groups.items():
            valid=[];bad=[]
            for c in rows:
                directory=Path(rdf_results[c['id']]['output'])
                with h5py.File(directory/'direct_correlations.h5') as f:
                    good=np.isfinite(f['c2/c2_hat_zero'][()]) and bool(np.all(f['c2/valid_k'][...]))
                (valid if good else bad).append(str(directory))
            key=f'{species}-T{T:g}';excluded[key]=bad
            if len(valid)<2:
                state['stages']['c1:'+key]={'status':'skipped','reason':'Fewer than two usable same-species/T zero-mode estimates','excluded_states':bad};continue
            args=['c1-scan','--results',*valid,'--model-id',a.model_id,'--assume-single-component','--temperature-K',str(T)]
            items[key]=('c1',key,args,output/'c1_relative'/key,'relative_only_missing_c1_anchor')
        phase('c1',items);_write_atomic(output/'c1_excluded_states.json',excluded)
    progress();requested=['rdf','analysis','c1'] if a.phase=='all' else [a.phase]
    relevant={k:v for k,v in state['stages'].items() if k.split(':')[0] in requested}
    counts=defaultdict(int)
    for key,value in relevant.items():counts[key.split(':')[0]+':'+value['status']]+=1
    state.update(status='completed' if all(v['status']=='completed' for v in relevant.values()) else 'completed_with_failures_or_skips',
                 phase='finished',summary=dict(counts),elapsed_s=time.time()-started)
    progress();_write_atomic(output/'collection_summary.json',state)
    print(json.dumps({'status':state['status'],'summary':state['summary'],'output':str(output)},ensure_ascii=False),flush=True)
    return 0 if state['status']=='completed' else 1


if __name__=='__main__':sys.exit(main())
