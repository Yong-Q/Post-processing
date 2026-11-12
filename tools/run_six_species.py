"""Resume-safe staged real-data runs: RDF on paired states, full primary analysis."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import os
import hashlib
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
SPECIES=['H2O','NH3','CO2','CH4','C2H4','C2H6']


def first_minimum(path):
    with h5py.File(path/'distributions.h5') as f:
        e=f['rdf/centers/edges_0'][...];g=f['rdf/centers/g'][...]
    r=(e[1:]+e[:-1])/2;smoothed=gaussian_filter1d(g,2.)
    peaks,_=find_peaks(smoothed,prominence=.08,height=1.02)
    peaks=peaks[(r[peaks]>1.5)&(r[peaks]<7.)]
    if not len(peaks):raise ValueError('No resolved first center-RDF peak; specify cutoff manually')
    peak=peaks[0];valleys,_=find_peaks(-smoothed,prominence=.01)
    valleys=valleys[(valleys>peak)&(r[valleys]<min(r[peak]+3.5,e[-1]-.5))]
    if not len(valleys):raise ValueError('No resolved first center-RDF minimum; inspect and specify cutoff manually')
    cut=float(r[valleys[0]])
    return cut,{'method':'first smoothed RDF minimum after first peak; inspect physical interpretation','sigma_bins':2.,'peak_A':float(r[peak]),'minimum_A':cut}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--jobs',type=int,default=6);p.add_argument('--resume',action='store_true');p.add_argument('--analysis-frames',type=int)
    p.add_argument('--phase',choices=['all','rdf','analysis','c1'],default='all');p.add_argument('--model-id',default='MACELES_user_declared_same_model_per_species')
    a=p.parse_args();data=Path(a.input_dir).resolve();out=Path(a.output_root).resolve()
    if out.exists() and not a.resume:raise SystemExit('Output root exists; --resume only reuses completed stages, never overwrites failed results')
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    files={}
    for sp in SPECIES:
        for state in ['r03','r01']:
            candidates=sorted(data.glob(f'{sp}-T*-{state}-*.dump.gz'))
            if len(candidates)!=1:raise SystemExit(f'Need one {sp} {state} file, found {len(candidates)}')
            files[(sp,state)]=candidates[0]
    def job(label,target,arguments,required_status='completed'):
        report=target/'report.json'
        identity=hashlib.sha256(json.dumps(arguments,ensure_ascii=False).encode()).hexdigest()
        marker=target.parent/(target.name+'.job.json')
        if target.exists():
            previous=json.loads(report.read_text()) if report.exists() else {}
            if a.resume and previous.get('status')==required_status:
                if marker.exists() and json.loads(marker.read_text())['arguments_sha256']!=identity:
                    raise RuntimeError('Completed output used different arguments; choose a new output root: '+str(target))
                if arguments[0]=='analyze' and previous.get('configuration',{}).get('max_frames')!=a.analysis_frames:
                    raise RuntimeError('Cannot reuse short smoke result as full analysis: '+str(target))
                return {'label':label,'status':'reused','output':str(target)}
            raise RuntimeError('Existing incomplete or incompatible output must be preserved: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','waterpost']+arguments+['--output',str(target)]
        log=target.parent/(target.name+'.log')
        if log.exists():raise RuntimeError('Existing log: '+str(log))
        if not marker.exists():marker.write_text(json.dumps({'arguments_sha256':identity,'arguments':arguments},indent=2))
        print('START',label,flush=True)
        with log.open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(label+' failed, see '+str(log))
        summary={'label':label,'status':'completed','output':str(target),'log':str(log)}
        print('DONE',label,flush=True);return summary
    summaries=[]
    if a.phase in ['all','rdf']:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            tasks=[pool.submit(job,sp+' '+state+' RDF',out/'rdf'/sp/state,
                               ['rdf-only','--species',sp,'--input',str(source),'--rdf-max','12','--bins','240','--block-frames','100']) for (sp,state),source in files.items()]
            for f in as_completed(tasks):summaries.append(f.result())
    if a.phase in ['all','analysis']:
        cuts={}
        for sp in SPECIES:
            cuts[sp]=first_minimum(out/'rdf'/sp/'r03')
        config=out/'analysis_windows.json'
        if not config.exists():config.write_text(json.dumps({sp:info for sp,(_,info) in cuts.items()},indent=2))
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            tasks=[]
            for sp in SPECIES:
                cut,_=cuts[sp];window=max(8.,cut+1.5)
                arguments=['analyze','--species',sp,'--input',str(files[(sp,'r03')]),'--nrad','3','--lmax','3',
                           '--rdf-max','12','--bins','80','--first-cut',str(cut),'--env-cut',str(window),'--lsi-cut',str(cut+.2),
                           '--block-frames','100','--expensive-stride','10','--pair-stride','10','--max-lag','100','--bond-max-lag','20','--window-centers','32']
                if a.analysis_frames:arguments+=['--max-frames',str(a.analysis_frames)]
                tasks.append(pool.submit(job,sp+' molecular analysis',out/'analysis'/sp,arguments))
            for f in as_completed(tasks):summaries.append(f.result())
    if a.phase in ['all','c1']:
        for sp in SPECIES:
            summaries.append(job(sp+' relative c1',out/'c1_relative'/sp,
                                 ['c1-scan','--results',str(out/'rdf'/sp/'r03'),str(out/'rdf'/sp/'r01'),'--model-id',a.model_id,'--assume-single-component'],
                                 required_status='relative_only_missing_c1_anchor'))
    manifest=out/('run_'+a.phase+'_summary.json')
    if manifest.exists():manifest=out/('run_'+a.phase+'_summary_resume.json')
    if manifest.exists():raise RuntimeError('Preserving existing summary; inspect outputs before another resume')
    manifest.write_text(json.dumps({'phase':a.phase,'analysis_frames_limit':a.analysis_frames,'stages':summaries},indent=2))
    print('COMPLETE',str(manifest),flush=True)


if __name__=='__main__':main()
