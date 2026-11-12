from dataclasses import dataclass
from pathlib import Path
import gzip
import numpy as np
from .geometry import mic


@dataclass
class Frame:
    step: int
    cell: np.ndarray
    origin: np.ndarray
    columns: list
    data: np.ndarray

    def field(self, name):
        return self.data[:, self.columns.index(name)]

    @property
    def positions(self):
        for names, scaled in [(('x','y','z'),False),(('xu','yu','zu'),False),
                              (('xs','ys','zs'),True),(('xsu','ysu','zsu'),True)]:
            if all(x in self.columns for x in names):
                a=np.column_stack([self.field(x) for x in names])
                return a @ self.cell if scaled else a-self.origin
        raise ValueError('Dump needs x/y/z, xu/yu/zu, xs/ys/zs, or xsu/ysu/zsu')

    @property
    def unwrapped(self):
        for names,scaled in [(('xu','yu','zu'),False),(('xsu','ysu','zsu'),True)]:
            if all(x in self.columns for x in names):
                a=np.column_stack([self.field(x) for x in names])
                return a@self.cell if scaled else a-self.origin
        if all(x in self.columns for x in ['ix','iy','iz']):
            return self.positions + np.column_stack([self.field(x) for x in ['ix','iy','iz']]) @ self.cell
        return None


def read_dump(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        while True:
            line=f.readline()
            if not line: return
            if line.strip()!='ITEM: TIMESTEP':raise ValueError('Expected ITEM: TIMESTEP')
            step=int(f.readline())
            if f.readline().strip()!='ITEM: NUMBER OF ATOMS':raise ValueError('Missing atom count')
            n=int(f.readline());header=f.readline().split()
            if header[:3]!=['ITEM:','BOX','BOUNDS']:raise ValueError('Missing box')
            triclinic=header[3:6]==['xy','xz','yz']
            flags=header[6:] if triclinic else header[3:]
            if flags!=['pp','pp','pp']:raise ValueError('Only 3D periodic boxes supported; got '+str(flags))
            bounds=[list(map(float,f.readline().split())) for _ in range(3)]
            if triclinic:
                (xl,xh,xy),(yl,yh,xz),(zl,zh,yz)=bounds
                xl-=min(0,xy,xz,xy+xz);xh-=max(0,xy,xz,xy+xz)
                yl-=min(0,yz);yh-=max(0,yz)
                cell=np.array([[xh-xl,0,0],[xy,yh-yl,0],[xz,yz,zh-zl]])
            else:
                if any(len(b)!=2 for b in bounds):raise ValueError('Unsupported box format')
                (xl,xh),(yl,yh),(zl,zh)=bounds
                cell=np.diag([xh-xl,yh-yl,zh-zl])
            if np.linalg.det(cell)<=0:raise ValueError('Invalid cell volume')
            head=f.readline().split()
            if head[:2]!=['ITEM:','ATOMS']:raise ValueError('Missing ATOMS header')
            columns=head[2:]
            rows=''.join(f.readline() for _ in range(n))
            values=np.fromstring(rows,sep=' ')
            if values.size!=n*len(columns):raise ValueError('Truncated or malformed atom table')
            data=values.reshape(n,len(columns))
            if not np.all(np.isfinite(data)):raise ValueError('Nonfinite input')
            for key in ['id','mol','type']:
                if key not in columns:raise ValueError('Water topology requires '+key)
            data=data[np.argsort(data[:,columns.index('id')])]
            if len(np.unique(data[:,columns.index('id')]))!=n:raise ValueError('Duplicate atom IDs')
            yield Frame(step,cell,np.array([xl,yl,zl]),columns,data)


class WaterTopology:
    def __init__(self,frame,oxygen_type=4,hydrogen_type=2,mode='fixed',bond_cut=1.35):
        self.mode=mode;self.bond_cut=bond_cut
        self.ids=frame.field('id').astype(np.int64)
        self.mol=frame.field('mol').astype(np.int64)
        self.types=frame.field('type').astype(np.int64)
        if not np.all(np.isin(self.types,[oxygen_type,hydrogen_type])):
            raise ValueError('Input contains atom types outside configured O/H types')
        self.mol_ids=np.unique(self.mol);indices=[]
        for mid in self.mol_ids:
            oi=np.flatnonzero((self.mol==mid)&(self.types==oxygen_type))
            hi=np.flatnonzero((self.mol==mid)&(self.types==hydrogen_type))
            if len(oi)!=1 or len(hi)!=2:raise ValueError(f'Molecule {mid}: expected exactly 1 O and 2 H')
            indices.append([oi[0],hi[0],hi[1]])
        self.indices=np.asarray(indices)
        self.h_indices=np.flatnonzero(self.types==hydrogen_type)

    def extract(self,frame):
        for key,expected in [('id',self.ids),('mol',self.mol),('type',self.types)]:
            if not np.array_equal(frame.field(key),expected):raise ValueError('Atom topology changed: '+key)
        indices=self.indices.copy();o=frame.positions[indices[:,0]]
        hp=frame.positions[self.h_indices]
        if self.mode=='nearest':
            distance=np.linalg.norm(mic(hp[None,:,:]-o[:,None,:],frame.cell),axis=-1)
            owner=np.argmin(distance,axis=0);best=distance[owner,np.arange(len(hp))]
            owner[best>self.bond_cut]=-1
            counts=np.bincount(owner[owner>=0],minlength=len(o));valid=counts==2
            for i in np.flatnonzero(valid):indices[i,1:]=self.h_indices[np.flatnonzero(owner==i)]
        else:
            owner=np.searchsorted(self.mol_ids,self.mol[self.h_indices]);counts=np.full(len(o),2);valid=np.ones(len(o),bool)
        p=frame.positions[indices];oh=mic(p[:,1:]-o[:,None],frame.cell)
        lengths=np.linalg.norm(oh,axis=-1)
        if self.mode=='fixed' and np.any((lengths<.5)|(lengths>1.6)):
            raise ValueError('Invalid O-H mapping/bond length (expected Angstrom coordinates); dissociation requires a different topology model')
        valid&=np.all((lengths>=.5)&(lengths<=self.bond_cut if self.mode=='nearest' else lengths<=1.6),axis=1)
        oh[~valid]=np.nan
        current_ids=self.ids[indices].copy();current_ids[~valid,1:]=-1
        out={'o':o,'oh':oh,'mol_ids':self.mol_ids,'atom_ids':current_ids,
             'raw_h_positions':hp,'h_owner':owner,'assigned_h_count':counts,'water_valid':valid,
             'unassigned_h_ids':self.ids[self.h_indices[owner<0]]}
        for name,fields in [('forces',['fx','fy','fz']),('velocities',['vx','vy','vz'])]:
            if all(k in frame.columns for k in fields):
                values=np.column_stack([frame.field(k) for k in fields])[indices]
                values[~valid]=np.nan;out[name]=values
        if 'q' in frame.columns:
            charges=frame.field('q')[indices].copy();charges[~valid]=np.nan;out['charges']=charges
        if frame.unwrapped is not None:out['unwrapped_o']=frame.unwrapped[self.indices[:,0]]
        return out
