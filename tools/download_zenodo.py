"""Authorized read-only Zenodo downloads; token comes from environment only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import requests


SPECIES=['H2O','NH3','CO2','CH4','C2H4','C2H6']


def checksum(path):
    h=hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=int,default=24123911);p.add_argument('--output-dir',required=True)
    p.add_argument('--species',choices=SPECIES,nargs='+',default=SPECIES);p.add_argument('--states',nargs='+',default=['r03','r01'])
    p.add_argument('--jobs',type=int,default=2);a=p.parse_args()
    token=os.environ.get('ZENODO_TOKEN')
    if not token:raise SystemExit('Set ZENODO_TOKEN in your environment; do not put credentials in source or command arguments')
    if a.jobs<1 or a.record<1:raise SystemExit('Positive record and jobs required')
    headers={'Authorization':'Bearer '+token};data=None
    for attempt in range(8):
        session=requests.Session();session.trust_env=bool(attempt%2)
        try:
            r=session.get(f'https://zenodo.org/api/deposit/depositions/{a.record}',headers=headers,timeout=(15,60));r.raise_for_status();data=r.json();break
        except requests.RequestException as e:print('metadata retry',attempt+1,type(e).__name__,flush=True);time.sleep(min(2**(attempt//3),8))
    if data is None:raise SystemExit('Unable to read authorized record metadata')
    selected=[]
    for species in a.species:
        for state in a.states:
            choices=[f for f in data.get('files',[]) if Path(f['filename']).name.startswith(species+'-T') and '-'+state+'-' in f['filename'] and f['filename'].endswith('.dump.gz')]
            if len(choices)!=1:raise SystemExit(f'Expected exactly one {species} {state} trajectory; got {len(choices)}')
            selected.append(choices[0])
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest={'record_id':a.record,'selection':{'species':a.species,'states':a.states},'files':selected}
    key=hashlib.sha256(json.dumps(manifest['selection'],sort_keys=True).encode()).hexdigest()[:10]
    path=output/f'download_manifest_{key}.json'
    if not path.exists():path.write_text(json.dumps(manifest,indent=2))
    def download(item):
        name=Path(item['filename']).name;dest=output/name;part=output/(name+'.part')
        expected=item['checksum'].removeprefix('md5:').zfill(32)
        link=item['links']['download'];parsed=urlsplit(link)
        if parsed.scheme!='https' or parsed.hostname!='zenodo.org':raise RuntimeError('Refusing to send token outside zenodo.org')
        if dest.exists():
            if dest.stat().st_size==item['filesize'] and checksum(dest)==expected:print('VERIFIED EXISTING',name,flush=True);return
            raise RuntimeError('Existing file mismatch, refusing overwrite: '+name)
        for attempt in range(12):
            offset=part.stat().st_size if part.exists() else 0
            if offset==item['filesize']:
                if checksum(part)!=expected:raise RuntimeError('Complete partial checksum mismatch: '+name)
                part.rename(dest);print('VERIFIED',name,flush=True);return
            session=requests.Session();session.trust_env=bool(attempt%2);request_headers=headers.copy()
            if offset:request_headers['Range']=f'bytes={offset}-'
            try:
                with session.get(link,headers=request_headers,stream=True,timeout=(15,90)) as response:
                    response.raise_for_status();resume=bool(offset and response.status_code==206)
                    if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise RuntimeError('Bad range response')
                    with part.open('ab' if resume else 'wb') as stream:
                        for chunk in response.iter_content(1024*1024):stream.write(chunk)
                if part.stat().st_size!=item['filesize'] or checksum(part)!=expected:raise RuntimeError('Size/checksum mismatch: '+name)
                if dest.exists():raise RuntimeError('Destination appeared')
                part.rename(dest);print('VERIFIED',name,flush=True);return
            except requests.RequestException as e:print('RETRY',name,attempt+1,type(e).__name__,flush=True);time.sleep(min(2**(attempt//3),8))
        raise RuntimeError('Download retries exhausted: '+name)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(download,selected))
    print('ALL VERIFIED',len(selected),flush=True)


if __name__=='__main__':main()
