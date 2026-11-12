"""Common center statistics with species-aware orientation and H-bond semantics.

Packing q describes angles between four nearest *molecular centers*, independent
of chemistry. Geometry histograms pool equivalent atoms and retain element
labels. All molecular orientation/geometry samples require valid molecules.
Distances are Angstrom; hydrogen bonds are geometric matches, not bond orders.
"""
import itertools
import numpy as np
import networkx as nx

from .geometry import mic, unit
from .static import shortest_rings

_ELEMENTS=('C','H','N','O')


class MolecularStatistics:
    def __init__(self,hist,first_cut=3.5,env_cut=6.,bins=80,
                 hbond_cut=3.5,hbond_angle=30.,lsi_cut=3.7):
        """H bonds: heavy-heavy distance < hbond_cut and D-H vs D-A <= angle.

        They apply only to pure H2O or NH3. LSI uses neighbor spacing variance
        inside lsi_cut, including the first neighbor beyond that cutoff; it is
        undefined when that outer neighbor is unavailable. Pair distributions
        below are raw histograms, without ideal-gas/RDF normalization.
        """
        if not 0<first_cut<=env_cut or bins<2 or int(bins)!=bins:
            raise ValueError('require 0 < first_cut <= env_cut and integer bins >= 2')
        if hbond_cut<=0 or not 0<=hbond_angle<=180 or lsi_cut<=0:
            raise ValueError('invalid hydrogen-bond or LSI cutoff')
        self.h=hist; self.first=float(first_cut); self.cut=float(env_cut)
        self.hbond_cut=float(hbond_cut); self.hbond_angle=float(hbond_angle); self.lsi_cut=float(lsi_cut)
        self.bins=int(bins); self.r=np.linspace(0,env_cut,bins+1)
        self.cos=np.linspace(-1,1,bins+1); self.absolute=np.linspace(0,1,bins+1)
        self.qedges=np.linspace(-3,1,bins+1); self.integer=np.arange(-.5,128.5)
        self.applicability={'hydrogen_bonds':False,'axis_orientation':False,'plane_orientation':False}
        self.descriptions={
            'local/packing_q':'Four-nearest-molecular-center angular packing; not a chemical water-order assignment.',
            'local/lsi_A2':'Variance of consecutive center-distance spacings inside lsi_cut including first outside neighbor.',
            'geometry/center_pair_distance_A':'Unordered center pairs; raw distance counts, not g(r).',
            'triplet/center_angle_cos':'Unordered neighbor pairs within first_cut of a valid center; generic packing angles.',
            'orientation':'Ordered pairs of valid molecules within env_cut; director observables are head-tail invariant.',
            'hbond':'Only H2O/NH3; D-A distance and D-H/D-A angle criterion; H index is global atom index.',
            'network':'Shortest alternate-path cycles per undirected H-bond edge; contractible and winding separated.'}

    def _add(self,name,values,edges,frame):
        edges=edges if isinstance(edges,list) else [edges]
        values=np.asarray(values,float).reshape(-1,len(edges))
        self.h.add(name,values,edges,frame)

    def _geometry(self,data,valid,frame):
        samples=data.get('geometry_samples',{}); elements=np.asarray(data['elements'])
        for kind,key,suffix,edges in [('bond','bond_lengths_A','A',np.linspace(0,3,self.bins+1)),
                                      ('angle','angle_cos','cos',self.cos)]:
            if key not in samples: continue
            values=np.asarray(samples[key]); mol=np.asarray(samples[kind+'_molecule_index'],int)
            atoms=np.asarray(samples[kind+'_atom_indices'],int)
            labels={}
            for row,aa in enumerate(atoms):
                codes=elements[aa]
                if kind=='bond': codes=np.sort(codes)
                else:
                    codes=np.array([min(codes[0],codes[2]),codes[1],max(codes[0],codes[2])])
                label='_'.join(_ELEMENTS[e] for e in codes)
                labels.setdefault(label,[]).append(row)
            for label,rows in labels.items():
                rows=np.asarray(rows,int); rows=rows[valid[mol[rows]]]
                self._add(f'geometry/{kind}_{label}_{suffix}',values[rows],edges,frame)
        if 'torsion_rad' in samples:
            values=np.asarray(samples['torsion_rad']); owners=np.asarray(samples['torsion_molecule_index'],int)
            values=values[valid[owners]]
            self._add('geometry/torsion_rad',np.arctan2(np.sin(values),np.cos(values)),np.linspace(-np.pi,np.pi,self.bins+1),frame)
            self._add('geometry/torsion_cos3',np.cos(3*values),self.cos,frame)

    def _hydrogen_bonds(self,data,cell,delta,distance,valid,frame):
        n=len(valid); elements=np.asarray(data['elements']); present=set(elements.tolist())
        species=data.get('species')
        applicable=(species in ('H2O','NH3') if species is not None else present in ({1,3},{1,2}))
        self.applicability['hydrogen_bonds']=bool(applicable)
        edges=np.empty((0,3),int)
        if not applicable: return edges,None,None
        owner=np.asarray(data['owner'],int); attached=np.asarray(data['attached_heavy_index'],int)
        positions=np.asarray(data['positions'],float)
        h=np.flatnonzero((elements==1)&(owner>=0)&(attached>=0))
        donors=owner[h]; h=h[valid[donors]]; donors=owner[h]
        if len(h):
            directions=unit(mic(positions[h]-positions[attached[h]],cell))
            hh,aa=np.where((distance[donors]<self.hbond_cut)&valid[None,:])
            dd=donors[hh]; length=distance[dd,aa]
            direction=np.divide(delta[dd,aa],length[:,None],out=np.full((len(dd),3),np.nan),where=length[:,None]>0)
            cosines=np.clip(np.einsum('ij,ij->i',directions[hh],direction),-1,1)
            self._add('hbond/distance_angle_cos',np.column_stack([length,cosines]),[np.linspace(0,self.hbond_cut,self.bins+1),self.cos],frame)
            keep=cosines>=np.cos(np.deg2rad(self.hbond_angle))
            edges=np.column_stack([dd[keep],h[hh[keep]],aa[keep]]).astype(int)
        donor=np.bincount(edges[:,0],minlength=n).astype(float)
        acceptor=np.bincount(edges[:,2],minlength=n).astype(float)
        donor[~valid]=np.nan; acceptor[~valid]=np.nan
        for name,x in [('donors',donor),('acceptors',acceptor),('total',donor+acceptor)]:
            self._add('hbond/'+name,x,self.integer,frame)
        self._add('joint/hbond_donor_acceptor',np.column_stack([donor,acceptor]),[self.integer,self.integer],frame)
        multiplicity=np.bincount(edges[:,1],minlength=len(elements))
        self._add('hbond/acceptors_per_H',multiplicity[h],self.integer,frame)
        return edges,donor,acceptor

    def _network(self,n,valid,edges,delta,cell,frame):
        graph=nx.Graph(); graph.add_nodes_from(np.flatnonzero(valid)); graph.add_edges_from(edges[:,[0,2]])
        sizes=np.array([len(x) for x in nx.connected_components(graph)],int)
        self._add('network/component_size',sizes,np.arange(.5,n+1.5),frame)
        self._add('network/degree',[graph.degree(i) for i in graph],self.integer,frame)
        rings=shortest_rings(n,graph.edges(),max_size=8); closed=[]; winding=[]; winding_vectors=[]
        for ring in rings:
            closure=sum((delta[a,b] for a,b in zip(ring,ring[1:]+ring[:1])),np.zeros(3))
            image=np.rint(closure@np.linalg.inv(cell)).astype(int)
            if np.any(image): winding.append(ring); winding_vectors.append(image)
            else: closed.append(ring)
        self._add('network/contractible_shortest_ring_size',[len(x) for x in closed],np.arange(2.5,9.5),frame)
        self._add('network/winding_shortest_ring_size',[len(x) for x in winding],np.arange(2.5,9.5),frame)
        return dict(largest_component=int(sizes.max()) if len(sizes) else 0,
                    component_sizes=sizes,ring_count=len(closed),winding_ring_count=len(winding),
                    winding_vectors=np.asarray(winding_vectors,int).reshape(-1,3))

    def frame(self,data,cell,frame_index,expensive=False,voronoi=None):
        centers=np.asarray(data['centers'],float); n=len(centers)
        valid=np.asarray(data['molecule_valid'],bool); cell=np.asarray(cell,float)
        delta=mic(centers[None,:,:]-centers[:,None,:],cell)
        distance=np.linalg.norm(delta,axis=-1); np.fill_diagonal(distance,np.inf)
        order=np.argsort(distance,axis=1); ordered=np.take_along_axis(distance,order,axis=1)
        nearest=ordered[:,:min(6,max(0,n-1))].copy()
        coordination=np.sum(distance<self.first,axis=1).astype(float)
        second=np.sum((distance>=self.first)&(distance<self.cut),axis=1).astype(float)
        q=np.full(n,np.nan); cv=q.copy(); lsi=q.copy()
        if n>=5:
            vectors=unit(delta[np.arange(n)[:,None],order[:,:4]])
            a,b=np.triu_indices(4,1)
            cosines=np.einsum('nid,njd->nij',vectors,vectors)[:,a,b]
            q=1-3/8*np.sum((cosines+1/3)**2,axis=1)
            cv=np.divide(np.std(ordered[:,:4],axis=1),np.mean(ordered[:,:4],axis=1),out=np.full(n,np.nan),where=np.mean(ordered[:,:4],axis=1)>0)
        for i in range(n):
            inside=int(np.sum(ordered[i]<self.lsi_cut))
            if 0<inside<n-1: lsi[i]=np.var(np.diff(ordered[i,:inside+1]))
        for x in (nearest,coordination,second,q,cv,lsi): x[~valid]=np.nan
        m=dict(delta=delta,distance=distance,coordination=coordination,second_shell=second,
               packing_q=q,lsi=lsi,nearest=nearest,radial_cv=cv)
        f=frame_index
        upper=np.triu_indices(n,1)
        self._add('geometry/center_pair_distance_A',distance[upper],self.r,f)
        self._add('geometry/valid_center_pair_distance_A',distance[upper][valid[upper[0]]&valid[upper[1]]],self.r,f)
        for name,x,ed in [('packing_q',q,self.qedges),('coordination',coordination,self.integer),
                          ('second_shell',second,self.integer),('lsi_A2',lsi,np.linspace(0,4,self.bins+1)),
                          ('radial_cv',cv,self.absolute)]: self._add('local/'+name,x,ed,f)
        for rank in range(nearest.shape[1]): self._add(f'local/nearest_{rank+1}_A',nearest[:,rank],self.r,f)
        for name,x,y,xe,ye in [('packing_coordination',q,coordination,self.qedges,self.integer),
                              ('first_second_shell',coordination,second,self.integer,self.integer),
                              ('packing_lsi',q,lsi,self.qedges,np.linspace(0,4,self.bins+1))]:
            self._add('joint/'+name,np.column_stack([x,y]),[xe,ye],f)
        if nearest.shape[1]>=5:
            self._add('joint/r4_r5_A',nearest[:,3:5],[self.r,self.r],f)
            self._add('local/r5_minus_r4_A',nearest[:,4]-nearest[:,3],self.r,f)
        i,j=np.where((distance<self.cut)&(distance>0)&valid[:,None]&valid[None,:]); rr=distance[i,j]
        rh=delta[i,j]/rr[:,None]
        kind=data['axis_kind']; axis=unit(data['axis']); normal=unit(data['plane_normal'])
        self.applicability['axis_orientation']=kind in ('polar','director')
        if kind in ('polar','director'):
            ci=np.clip(np.einsum('ij,ij->i',axis[i],rh),-1,1); cj=np.clip(np.einsum('ij,ij->i',axis[j],-rh),-1,1)
            mutual=np.clip(np.einsum('ij,ij->i',axis[i],axis[j]),-1,1)
            values=[('mutual_P2',(3*mutual**2-1)/2,self.cos)]
            if kind=='polar': values += [('center_axis_cos',ci,self.cos),('neighbor_axis_cos',cj,self.cos),('mutual_P1',mutual,self.cos)]
            else: values += [('center_axis_abs_cos',abs(ci),self.absolute),('neighbor_axis_abs_cos',abs(cj),self.absolute),('mutual_abs_cos',abs(mutual),self.absolute)]
            for name,x,ed in values: self._add('orientation/'+name+'_vs_r',np.column_stack([rr,x]),[self.r,ed],f)
            self._add('joint/center_neighbor_axis',np.column_stack([ci,cj]) if kind=='polar' else np.column_stack([abs(ci),abs(cj)]),[self.cos,self.cos] if kind=='polar' else [self.absolute,self.absolute],f)
        self.applicability['plane_orientation']=bool(np.any(np.all(np.isfinite(normal[valid]),axis=1)))
        if self.applicability['plane_orientation']:
            plane=np.clip(abs(np.einsum('ij,ij->i',normal[i],normal[j])),0,1)
            self._add('orientation/plane_abs_cos_vs_r',np.column_stack([rr,plane]),[self.r,self.absolute],f)
        self._geometry(data,valid,f)
        # Unordered neighbor-pair angles; raw counts have no c3 interpretation.
        for center in np.flatnonzero(valid):
            nb=np.flatnonzero(distance[center]<self.first)
            if len(nb)<2: continue
            a,b=np.triu_indices(len(nb),1); u=unit(delta[center,nb]); cosine=np.clip(np.einsum('ij,ij->i',u[a],u[b]),-1,1)
            self._add('triplet/center_angle_cos',cosine,self.cos,f)
            self._add('triplet/r_r_cos',np.column_stack([distance[center,nb[a]],distance[center,nb[b]],cosine]),[np.linspace(0,self.first,21)]*2+[np.linspace(-1,1,21)],f)
        edges,donor,acceptor=self._hydrogen_bonds(data,cell,delta,distance,valid,f)
        m['hbond_triplets']=edges
        if donor is not None:
            m.update(hbond_donors=donor,hbond_acceptors=acceptor)
            if expensive: m['network']=self._network(n,valid,edges,delta,cell,f)
        if voronoi is not None:
            volume=np.asarray(voronoi['volume'],float).copy(); faces=np.asarray(voronoi['faces'],float).copy(); surface=np.asarray(voronoi['surface_area'],float).copy()
            for x in (volume,faces,surface): x[~valid]=np.nan
            shape=np.divide(surface**3,36*np.pi*volume**2,out=np.full(n,np.nan),where=volume>0)
            m.update(voronoi_volume=volume,voronoi_faces=faces,voronoi_surface_area=surface,voronoi_shape=shape)
            for name,x,ed in [('volume_A3',volume,np.linspace(0,2000,self.bins+1)),('faces',faces,self.integer),('surface_area_A2',surface,np.linspace(0,1000,self.bins+1)),('shape',shape,np.linspace(1,10,self.bins+1))]:
                self._add('local/voronoi_'+name,x,ed,f)
        m['valid_count']=int(valid.sum()); m['invalid_count']=int((~valid).sum())
        return m
