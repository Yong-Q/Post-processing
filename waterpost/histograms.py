import numpy as np


def rdf_normalize(counts,edges,na,nb,volume,same=False):
    shell=4*np.pi/3*np.diff(np.asarray(edges)**3)
    expected=na*(nb-1 if same else nb)/volume*shell
    return np.divide(counts,expected,out=np.full_like(expected,np.nan,dtype=float),where=expected>0)


class Histograms:
    def __init__(self,block_size=100):
        self.block_size=block_size;self.items={}

    def add(self,name,samples,edges,frame,weights=None,expected=None):
        samples=np.asarray(samples,dtype=float).reshape(-1,len(edges))
        finite=np.all(np.isfinite(samples),axis=1)
        if weights is not None:finite&=np.isfinite(np.asarray(weights))
        w=None if weights is None else np.asarray(weights)[finite]
        counts=np.histogramdd(samples[finite],bins=edges,weights=w)[0]
        block=frame//self.block_size
        if name not in self.items:self.items[name]={'edges':edges,'counts':{},'exposure':{},'samples':0,'finite':0}
        item=self.items[name]
        item['counts'][block]=item['counts'].get(block,np.zeros_like(counts))+counts
        item['samples']+=len(samples);item['finite']+=int(finite.sum())
        if expected is not None:item['exposure'][block]=item['exposure'].get(block,np.zeros_like(counts))+expected

    def add_binned(self,name,counts,edges,frame,expected,total_samples):
        counts=np.asarray(counts,float);expected=np.asarray(expected,float)
        shape=tuple(len(e)-1 for e in edges)
        if counts.shape!=shape or expected.shape!=shape or np.any(counts<0) or not np.all(np.isfinite(counts)):
            raise ValueError('Invalid prebinned counts/exposure shape')
        block=frame//self.block_size
        if name not in self.items:self.items[name]={'edges':edges,'counts':{},'exposure':{},'samples':0,'finite':0}
        item=self.items[name]
        item['counts'][block]=item['counts'].get(block,np.zeros_like(counts))+counts
        item['exposure'][block]=item['exposure'].get(block,np.zeros_like(expected))+expected
        item['samples']+=int(total_samples);item['finite']+=int(total_samples)

    def result(self,name):
        item=self.items[name];blocks=sorted(item['counts'])
        bc=np.array([item['counts'][b] for b in blocks]);counts=bc.sum(axis=0)
        total=counts.sum();prob=counts/total if total else np.full_like(counts,np.nan)
        measure=np.ones(counts.shape)
        for axis,edges in enumerate(item['edges']):
            shape=[1]*len(item['edges']);shape[axis]=len(edges)-1
            measure*=np.diff(edges).reshape(shape)
        totals=bc.reshape(len(bc),-1).sum(axis=1)
        bp=np.divide(bc,totals.reshape((-1,)+(1,)*(bc.ndim-1)),out=np.full_like(bc,np.nan),where=totals.reshape((-1,)+(1,)*(bc.ndim-1))>0)
        valid=totals>0
        se=np.std(bp[valid],axis=0,ddof=1)/np.sqrt(valid.sum()) if valid.sum()>1 else np.full_like(counts,np.nan)
        out={'counts':counts,'probability':prob,'density':prob/measure,'probability_stderr':se,
             'block_counts':bc,'block_ids':np.array(blocks),'n_samples':item['samples'],'n_finite':item['finite'],'n_in_range':total}
        if item['exposure']:
            exposure=np.array([item['exposure'][b] for b in blocks]);den=exposure.sum(axis=0)
            out['expected_counts']=den;out['g']=np.divide(counts,den,out=np.full_like(counts,np.nan),where=den>0)
            bg=np.divide(bc,exposure,out=np.full_like(bc,np.nan),where=exposure>0)
            out['block_g']=bg
            out['g_stderr']=np.std(bg,axis=0,ddof=1)/np.sqrt(len(bg)) if len(bg)>1 else np.full_like(counts,np.nan)
        return out

    def write(self,h5):
        for name,item in self.items.items():
            group=h5.require_group(name)
            for i,edges in enumerate(item['edges']):group.create_dataset(f'edges_{i}',data=edges)
            for k,v in self.result(name).items():group.create_dataset(k,data=v)
            if len(item['edges'])==2:
                p=self.result(name)['probability']
                if np.all(np.isfinite(p)):
                    independent=p.sum(axis=1)[:,None]*p.sum(axis=0)[None,:]
                    mask=p>0
                    group.attrs['plugin_mutual_information_nats']=float(np.sum(p[mask]*np.log(p[mask]/independent[mask])))
                    group.attrs['MI_warning']='Exploratory plug-in estimate; biased for sparse bins; not a thermodynamic entropy'
