"""Species and center conventions shared by molecular analyses."""
import json
from pathlib import Path
import numpy as np
from .molecules import MolecularTopology,SPECS,MASS,ELEMENTS
from .geometry import mic


def infer_species(path,species='auto'):
    if species in (None,'auto'):
        species=Path(path).name.split('-T')[0]
    if species not in SPECS:raise ValueError('Declare one of '+', '.join(SPECS)+'; cannot infer species from filename')
    return species


def parse_type_map(value):
    if value is None:return None
    data=json.loads(value) if isinstance(value,str) else value
    return {int(k):str(v) for k,v in data.items()}


class CenterTracker:
    """Vectorized heavy-skeleton COM for full-frame RDF, without H chemistry."""
    def __init__(self,frame,species,type_map=None):
        self.top=MolecularTopology(frame,species,type_map=type_map)
        self.indices=np.stack(self.top.heavy_indices)
        self.types=frame.field('type').copy()
        self.mass=np.array([[MASS[ELEMENTS[e]] for e in row] for row in self.top.element_codes[self.indices]])
        self.ids=self.top.ids;self.spec=self.top.spec;self.species=species

    def extract(self,frame):
        if not np.array_equal(frame.field('id'),self.ids) or not np.array_equal(frame.field('type'),self.types):raise ValueError('Atomic identities or types changed')
        expected=np.broadcast_to(self.top.molecule_ids[:,None],self.indices.shape)
        if not np.array_equal(frame.field('mol')[self.indices],expected):raise ValueError('Heavy skeleton molecule labels changed')
        p=frame.positions[self.indices];anchor=p[:,:1]
        unwrapped=anchor+mic(p-anchor,frame.cell)
        centers=np.sum(unwrapped*self.mass[:,:,None],axis=1)/self.mass.sum(axis=1)[:,None]
        centers=np.mod(centers@np.linalg.inv(frame.cell),1.)@frame.cell
        valid=np.ones(len(centers),bool)
        if self.species.startswith('C2'):
            length=np.linalg.norm(mic(p[:,1]-p[:,0],frame.cell),axis=1);valid=(length>=.9)&(length<=1.9)
        elif self.species=='CO2':
            codes=self.top.element_codes[self.indices]
            c=np.argmax(codes==0,axis=1);carbon=p[np.arange(len(p)),c]
            d=np.linalg.norm(mic(p-carbon[:,None],frame.cell),axis=-1)
            oxy=codes==3;valid=np.all(np.where(oxy,(d>=.8)&(d<=1.6),True),axis=1)
        return centers,valid
