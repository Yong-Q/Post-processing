"""Reactive, symmetry-aware topology for six pure molecular species.

Coordinates use Frame's zero-origin convention. Axes and plane normals for
``director`` observables have arbitrary sign. Raw geometry describes current
assignments, including intermediates; validity must be applied for statistics.
Masses are standard elemental atomic weights (g/mol). Skeletal distance windows
are geometric identification checks, not a chemical bond-order model.
"""
import itertools
import numpy as np
from .geometry import mic, unit

ELEMENTS = ('C', 'H', 'N', 'O')
MASS = {'C': 12.011, 'H': 1.008, 'N': 14.007, 'O': 15.999}
DEFAULT_TYPE_MAP = {1: 'C', 2: 'H', 3: 'N', 4: 'O'}
SPECS = {}
for _name, _composition, _kind in [
    ('H2O', {'O': 1, 'H': 2}, 'polar'),
    ('NH3', {'N': 1, 'H': 3}, 'polar'),
    ('CO2', {'C': 1, 'O': 2}, 'director'),
    ('CH4', {'C': 1, 'H': 4}, 'none'),
    ('C2H4', {'C': 2, 'H': 4}, 'director'),
    ('C2H6', {'C': 2, 'H': 6}, 'director')]:
    SPECS[_name] = {'composition': _composition, 'molar_mass': sum(MASS[e]*n for e,n in _composition.items()),
                    'orientation_kind': _kind, 'center_definition': 'heavy_skeleton_com'}


def _wrap(p, cell):
    return np.mod(p @ np.linalg.inv(cell), 1.) @ cell


def _integer_field(frame, name):
    values = frame.field(name)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError(name + ' must contain finite integers')
    return values.astype(np.int64)


class MolecularTopology:
    """Keep original heavy skeletons; assign H to nearest eligible heavy atom.

    ``fixed`` retains original H membership but still checks bond distances.
    Every H must be between 0.5 Angstrom and ``bond_cut`` from its assigned heavy
    atom for a valid molecule. CO2 C-O: 0.8--1.6 A; C2 C-C: 0.9--1.9 A.
    """
    def __init__(self, frame, species, type_map=None, mode='nearest', bond_cut=1.35):
        if species not in SPECS:
            raise ValueError('Unsupported species: '+str(species))
        if mode not in ('nearest', 'fixed') or not np.isfinite(bond_cut) or bond_cut < .5:
            raise ValueError('Expected nearest/fixed mode and bond_cut >= 0.5 Angstrom')
        self.species, self.spec = species, SPECS[species]
        self.mode, self.bond_cut = mode, float(bond_cut)
        self.type_map = dict(DEFAULT_TYPE_MAP if type_map is None else type_map)
        if (not self.type_map or any(not isinstance(k, (int, np.integer)) or isinstance(k,bool)
                                      or v not in ELEMENTS for k,v in self.type_map.items())
                or len(set(self.type_map.values())) != len(self.type_map)):
            raise ValueError('type_map must map distinct integer atom types to distinct C/H/N/O symbols')
        self.ids = _integer_field(frame, 'id'); self.types = _integer_field(frame, 'type')
        self.mol = _integer_field(frame, 'mol')
        if not len(self.ids) or len(np.unique(self.ids)) != len(self.ids):
            raise ValueError('Atom IDs must be nonempty and unique')
        if any(t not in self.type_map for t in self.types):
            raise ValueError('Atom type absent from configured type_map')
        self.element_codes = np.array([ELEMENTS.index(self.type_map[t]) for t in self.types])
        self.molecule_ids = np.unique(self.mol)
        self.initial_owner = np.searchsorted(self.molecule_ids,self.mol)
        self.heavy_indices = []
        for i,mid in enumerate(self.molecule_ids):
            ind = np.flatnonzero(self.initial_owner == i)
            composition = {e:int(np.sum(self.element_codes[ind] == j)) for j,e in enumerate(ELEMENTS)}
            if any(composition[e] != self.spec['composition'].get(e,0) for e in ELEMENTS):
                raise ValueError(f'Molecule {mid}: composition disagrees with {species} / type_map')
            self.heavy_indices.append(ind[self.element_codes[ind] != 1])
        self.h_indices = np.flatnonzero(self.element_codes == 1)
        self._heavy = np.concatenate(self.heavy_indices)
        self._masses = np.array([MASS[ELEMENTS[e]] for e in self.element_codes])

    def extract(self, frame):
        for name, expected in [('id',self.ids),('type',self.types)]:
            if not np.array_equal(_integer_field(frame,name),expected):
                raise ValueError('Atom topology changed: '+name)
        current_mol = _integer_field(frame,'mol')
        if not np.array_equal(current_mol[self._heavy],self.mol[self._heavy]):
            raise ValueError('Original heavy skeleton molecule groups changed')
        cell = np.asarray(frame.cell,float)
        if cell.shape != (3,3) or not np.all(np.isfinite(cell)) or np.linalg.det(cell) <= 0:
            raise ValueError('Expected finite right-handed periodic cell')
        p = np.asarray(frame.positions,float)
        if p.shape != (len(self.ids),3) or not np.isfinite(p).all():
            raise ValueError('Invalid atomic coordinates')
        p = _wrap(p,cell)
        owner = self.initial_owner.copy(); attached = np.full(len(self.ids),-1,int)
        if len(self.h_indices):
            hp = p[self.h_indices]; heavy = p[self._heavy]
            if self.mode == 'nearest':
                if np.allclose(cell,np.diag(np.diag(cell)),atol=1e-12):
                    from scipy.spatial import cKDTree
                    dist, nearest = cKDTree(heavy,boxsize=np.diag(cell)).query(hp)
                else:
                    d = np.linalg.norm(mic(hp[:,None,:]-heavy[None,:,:],cell),axis=-1)
                    nearest = np.argmin(d,axis=1); dist = d[np.arange(len(hp)),nearest]
                hi = self._heavy[nearest]
                howner = self.initial_owner[hi].copy(); howner[dist > self.bond_cut] = -1
            else:
                hi = np.empty(len(hp),int); dist = np.empty(len(hp))
                for j,h in enumerate(self.h_indices):
                    candidates = self.heavy_indices[self.initial_owner[h]]
                    d = np.linalg.norm(mic(p[h]-p[candidates],cell),axis=1)
                    k = np.argmin(d); hi[j] = candidates[k]; dist[j] = d[k]
                howner = self.initial_owner[self.h_indices].copy()
            owner[self.h_indices] = howner
            attached[self.h_indices] = hi
            attached[self.h_indices[howner < 0]] = -1
        n = len(self.molecule_ids)
        centers = np.empty((n,3)); raw_com = np.empty((n,3)); rel = np.full_like(p,np.nan)
        axis = np.full((n,3),np.nan); normal = axis.copy()
        heavy_valid = np.ones(n,bool); valid = np.ones(n,bool)
        members = [np.flatnonzero(owner == i) for i in range(n)]
        counts = np.array([np.sum(self.element_codes[m] == 1) for m in members])
        # Reuse each covalent H-heavy vector across bond, angle and torsion
        # observables. Repeated per-angle MIC calls dominate ethane runs.
        h_vectors=np.full_like(p,np.nan);assigned=np.flatnonzero(attached>=0)
        h_vectors[assigned]=mic(p[assigned]-p[attached[assigned]],cell)
        h_lengths=np.linalg.norm(h_vectors,axis=1);h_units=unit(h_vectors)
        bonds=[]; angles=[]; torsions=[]
        for i,heavy in enumerate(self.heavy_indices):
            # Reconstruct the heavy skeleton before locating its center. Never
            # average wrapped C coordinates across a box face.
            anchor = p[heavy[0]]
            skeleton = anchor+mic(p[heavy]-anchor,cell)
            center = np.average(skeleton,axis=0,weights=self._masses[heavy])
            centers[i] = _wrap(center,cell)
            m=members[i]; rel[m] = mic(p[m]-center,cell)
            raw_com[i] = _wrap(centers[i]+np.average(rel[m],axis=0,weights=self._masses[m]),cell)
            h = m[self.element_codes[m] == 1]
            hb = []
            for a in h:
                b=attached[a]; length=h_lengths[a]
                bonds.append((length,i,b,a)); hb.append(length)
            valid[i] = counts[i] == self.spec['composition'].get('H',0) and all(.5 <= x <= self.bond_cut for x in hb)
            if self.species == 'CO2':
                c=heavy[self.element_codes[heavy]==0][0]
                oxy=heavy[self.element_codes[heavy]==3]
                lengths=np.linalg.norm(mic(p[oxy]-p[c],cell),axis=1)
                heavy_valid[i]=np.all((lengths>=.8)&(lengths<=1.6))
                bonds.extend((length,i,c,o) for length,o in zip(lengths,oxy))
                axis[i]=unit(mic(p[oxy[1]]-p[oxy[0]],cell))
                a,b=unit(mic(p[oxy]-p[c],cell)); angles.append((np.dot(a,b),i,oxy[0],c,oxy[1]))
            elif self.species.startswith('C2'):
                c0,c1=heavy; cc=mic(p[c1]-p[c0],cell); length=np.linalg.norm(cc)
                heavy_valid[i]=.9 <= length <= 1.9; bonds.append((length,i,c0,c1))
                axis[i]=unit(cc)
                per_c=2 if self.species=='C2H4' else 3
                valid[i] &= all(np.sum(attached[h]==c)==per_c for c in heavy)
                if self.species == 'C2H6':
                    left=h[attached[h]==c0];right=h[attached[h]==c1];z=axis[i]
                    u=h_vectors[left];v=h_vectors[right]
                    u=u-(u@z)[:,None]*z;v=v-(v@z)[:,None]*z
                    numerator=np.einsum('ijk,k->ij',np.cross(u[:,None,:],v[None,:,:]),z)
                    angle=np.arctan2(numerator,u@v.T)
                    good=(np.linalg.norm(u,axis=1)[:,None]>1e-12)&(np.linalg.norm(v,axis=1)[None,:]>1e-12)
                    angle[~good]=np.nan
                    torsions.extend((angle[a,b],i,ha,c0,c1,hb) for a,ha in enumerate(left) for b,hb in enumerate(right))
                if self.species == 'C2H4' and len(h)>=2:
                    _,s,vh=np.linalg.svd(rel[m],full_matrices=False)
                    if len(s)==3 and s[1]>1e-10 and s[1]-s[2]>1e-10:
                        normal[i]=vh[-1]
            elif self.species in ('H2O','NH3') and len(h):
                vectors=h_vectors[h]
                axis[i]=unit(np.sum(unit(vectors),axis=0))
                if self.species=='H2O' and len(h)==2:
                    normal[i]=unit(np.cross(vectors[0],vectors[1]))
            # All equivalent H pairs, and H-heavy-heavy angles for carbon dimers.
            for c in heavy:
                neighbors=list(h[attached[h]==c])
                directions=[h_units[a] for a in neighbors]
                if self.species.startswith('C2'):
                    other=list(heavy[heavy!=c]);neighbors+=other
                    directions += [axis[i] if c==heavy[0] else -axis[i] for _ in other]
                direction=np.asarray(directions,float).reshape(-1,3);products=direction@direction.T
                for a,b in itertools.combinations(range(len(neighbors)),2):
                    angles.append((products[a,b],i,neighbors[a],c,neighbors[b]))
        valid &= heavy_valid
        axis[~valid]=np.nan; normal[~valid]=np.nan
        com=raw_com.copy(); com[~valid]=np.nan
        geometry={}
        for rows,key,label,width in [(bonds,'bond','lengths_A',2),(angles,'angle','cos',3),(torsions,'torsion','rad',4)]:
            a=np.asarray(rows,float).reshape(-1,width+2)
            value_key='bond_lengths_A' if key=='bond' else key+'_'+label
            geometry[value_key]=a[:,0]
            geometry[key+'_molecule_index']=a[:,1].astype(int)
            geometry[key+'_atom_indices']=a[:,2:].astype(int)
        out={'centers':centers,'com':com,'raw_com':raw_com,'molecule_valid':valid,'heavy_valid':heavy_valid,
             'owner':owner,'positions':p,'elements':self.element_codes.copy(),'relative_positions':rel,
             'atom_ids':self.ids.copy(),'molecule_ids':self.molecule_ids.copy(),'members':members,
             'hydrogen_count':counts,'axis':axis,'axis_kind':self.spec['orientation_kind'],
             'plane_normal':normal,'geometry_samples':geometry,'attached_heavy_index':attached,
             'center_definition':'heavy_skeleton_com'}
        if all(k in frame.columns for k in ('fx','fy','fz')):
            forces=np.column_stack([frame.field(k) for k in ('fx','fy','fz')])
            raw_force=np.array([forces[m].sum(axis=0) for m in members])
            raw_torque=np.array([np.cross(rel[m],forces[m]).sum(axis=0) for m in members])
            force=raw_force.copy(); torque=raw_torque.copy(); force[~valid]=np.nan; torque[~valid]=np.nan
            out.update(forces=forces,molecular_force=force,molecular_torque=torque,
                       raw_molecular_force=raw_force,raw_molecular_torque=raw_torque)
        return out
