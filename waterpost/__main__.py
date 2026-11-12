import argparse
import json
import sys
from pathlib import Path
import numpy as np
import h5py
from .pipeline import analyze,inspect


def add_oz_options(parser):
    parser.add_argument('--oz-window',choices=['none','cosine'],default='none')
    parser.add_argument('--oz-tail-start',type=float,default=.8)
    parser.add_argument('--oz-min-s',type=float,default=1e-8)
    parser.add_argument('--beta-mu',type=float)
    parser.add_argument('--beta-mu-ex',type=float)
    parser.add_argument('--reference-volume',type=float,default=1.,help='Lambda^3/zeta in Angstrom^3; PNAS convention=1')


def main():
    parser=argparse.ArgumentParser(description='Existing water MD postprocessing; no MD simulation or overwriting.')
    sub=parser.add_subparsers(dest='command',required=True)
    ip=sub.add_parser('inspect');ip.add_argument('--input',required=True);ip.add_argument('--scan',action='store_true')
    ap=sub.add_parser('analyze');ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    for name,default,helptext in [('frame-dt-ps',.5,'Time between source dump frames, BEFORE --stride'),
                                ('first-cut',3.5,'First-shell O cutoff in Angstrom'),('env-cut',6.,'Environment cutoff in Angstrom'),
                                ('rdf-max',8.,'RDF maximum radius in Angstrom'),('lsi-cut',3.7,'LSI shell boundary in Angstrom'),
                                ('hbond-cut',3.5,'Donor O to acceptor O cutoff in Angstrom'),('hbond-angle',30.,'Angle at donor O between OH and OO, degrees')]:
        ap.add_argument('--'+name,type=float,default=default,help=helptext)
    for name,default in [('start',0),('stride',1),('block-frames',100),('expensive-stride',10),('pair-stride',5),
                         ('bins',80),('seed',20261112),('window-centers',64),('k-nmax',2),('nrad',4),('lmax',3),('bond-max-lag',20),('split-gap-frames',1)]:
        ap.add_argument('--'+name,type=int,default=default)
    ap.add_argument('--max-frames',type=int);ap.add_argument('--max-lag',type=int);ap.add_argument('--blocks',type=int)
    ap.add_argument('--force-unit',default='native_unknown',help='Set only after checking original LAMMPS units')
    ap.add_argument('--topology-mode',choices=['fixed','nearest'],default='nearest',help='nearest: geometric assignment with nonwater masking; fixed: strict original mol IDs')
    ap.add_argument('--bond-cut',type=float,default=1.35,help='Nearest-O H assignment cutoff, Angstrom; sensitivity must be assessed')
    ap.add_argument('--no-voronoi',action='store_true');ap.add_argument('--no-equivariant',action='store_true')
    ap.add_argument('--no-direct-correlations',action='store_true');add_oz_options(ap)
    for p in [ip,ap]:
        p.add_argument('--oxygen-type',type=int,default=4);p.add_argument('--hydrogen-type',type=int,default=2)
    fp=sub.add_parser('fit-diffusion');fp.add_argument('--results',required=True);fp.add_argument('--start-ps',type=float,required=True);fp.add_argument('--end-ps',type=float,required=True)
    cp=sub.add_parser('compare');cp.add_argument('--results',nargs='+',required=True);cp.add_argument('--output',required=True)
    op=sub.add_parser('correlations');op.add_argument('--results',required=True);op.add_argument('--output',required=True);add_oz_options(op)
    rp=sub.add_parser('rdf-only');rp.add_argument('--input',required=True);rp.add_argument('--output',required=True)
    for name,default in [('start',0),('stride',1),('bins',240),('block-frames',100),('oxygen-type',4)]:rp.add_argument('--'+name,type=int,default=default)
    rp.add_argument('--rdf-max',type=float,default=12.);rp.add_argument('--frame-dt-ps',type=float,default=.5)
    rp.add_argument('--temperature-K',type=float);rp.add_argument('--max-frames',type=int);add_oz_options(rp)
    sp=sub.add_parser('c1-scan');sp.add_argument('--results',nargs='+',required=True);sp.add_argument('--output',required=True)
    sp.add_argument('--temperature-K',type=float);sp.add_argument('--model-id',required=True)
    sp.add_argument('--assume-single-component',action='store_true');sp.add_argument('--anchor-index',type=int,default=0);sp.add_argument('--anchor-c1',type=float)
    args=parser.parse_args()
    try:
        if args.command=='inspect':print(json.dumps(inspect(args.input,args.oxygen_type,args.hydrogen_type,args.scan),indent=2))
        elif args.command=='analyze':analyze(args)
        elif args.command=='rdf-only':
            from .rdf import rdf_only
            rdf_only(args)
        elif args.command=='correlations':
            from .correlations import run_correlations
            print(json.dumps(run_correlations(args),indent=2))
        elif args.command=='c1-scan':
            from .correlations import c1_density_scan
            print(json.dumps(c1_density_scan(args),indent=2))
        elif args.command=='fit-diffusion':
            with h5py.File(Path(args.results)/'dynamics.h5') as f:
                t=f['oxygen_motion/lag_ps'][...];x=f['oxygen_motion/msd'][...]
                mask=(t>=args.start_ps)&(t<=args.end_ps)&(t>0)
                if mask.sum()<3:raise ValueError('Need at least 3 positive-time points in explicit fit interval')
                slope,intercept=np.polyfit(t[mask],x[mask],1)
                residual=x[mask]-(slope*t[mask]+intercept)
                total=np.sum((x[mask]-x[mask].mean())**2)
                print(json.dumps({'D_A2_per_ps':slope/6,'D_m2_per_s':slope/6*1e-8,'intercept_A2':intercept,
                                  'R2':1-np.sum(residual**2)/total if total else None,'points':int(mask.sum()),
                                  'warning':'Descriptive fit only; inspect diffusive regime, unwrap validity, uncertainty and finite-size/thermostat effects.'},indent=2))
        else:
            out=Path(args.output)
            if out.exists():raise ValueError('Output already exists; refusing overwrite')
            reports=[];features=[]
            for directory in args.results:
                directory=Path(directory);r=json.loads((directory/'report.json').read_text())
                with h5py.File(directory/'collective.h5') as f:
                    features.append(f['feature_statistics/mean'][...])
                reports.append({'directory':str(directory.resolve()),'frames':r['frames_analyzed'],'density_g_cm3':r['density_g_cm3_from_18_01528_amu'],'input':r['input'],'configuration':r['configuration']})
            if len(set(len(x) for x in features))!=1:raise ValueError('Feature dimensions differ; compare runs with identical basis')
            means=np.asarray(features);dist=np.linalg.norm(means[:,None]-means[None,:],axis=-1)
            out.write_text(json.dumps({'runs':reports,'raw_feature_mean_distances':dist.tolist(),'warning':'States remain separate; raw feature distances are descriptor dependent, not thermodynamic distances.'},indent=2))
            print(str(out))
    except Exception as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
