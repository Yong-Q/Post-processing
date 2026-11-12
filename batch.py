"""Run waterpost over existing water dumps, keeping each state separate."""
import argparse
from pathlib import Path
import json
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--pattern',default='H2O*.dump.gz')
    p.add_argument('--mode',choices=['analyze','rdf-only'],default='analyze')
    a,extra=p.parse_known_args();files=sorted(Path(a.input_dir).glob(a.pattern))
    if not files:raise SystemExit('No matching water trajectories')
    root=Path(a.output_root)
    if root.exists():raise SystemExit('Refusing existing output root; choose a new directory')
    root.mkdir(parents=True);runs=[]
    for file in files:
        name=file.name.removesuffix('.gz').removesuffix('.dump');target=root/name
        command=[sys.executable,'-m','waterpost',a.mode,'--input',str(file.resolve()),'--output',str(target.resolve())]+extra
        with (root/(name+'.log')).open('x') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parent).returncode
        runs.append({'input':str(file.resolve()),'output':str(target.resolve()),'exit_code':code})
        print(name,'OK' if code==0 else 'FAILED',flush=True)
    (root/'batch_report.json').write_text(json.dumps(runs,indent=2))
    return int(any(r['exit_code'] for r in runs))


if __name__=='__main__':sys.exit(main())
