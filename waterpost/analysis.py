"""Local, pair and joint statistics. Counts remain available for rebinning."""
import numpy as np
import networkx as nx
from .geometry import mic,unit
from .static import shortest_rings


class Statistics:
    def __init__(self,hist,first_cut,env_cut,rdf_max,bins=80):
        self.h=hist;self.first=first_cut;self.cut=env_cut
        self.r=np.linspace(0,rdf_max,bins+1);self.er=np.linspace(0,env_cut,41)
        self.cos=np.linspace(-1,1,41);self.qedges=np.linspace(-3,1,81)
        self.integer=np.arange(-.5,60.5)
        self.spatial_means=[]

    def add(self,name,x,edges,frame,**kwargs):
        if not isinstance(edges,list):edges=[edges]
        x=np.asarray(x)
        if len(edges)==1:x=x.reshape(-1,1)
        self.h.add(name,x,edges,frame,**kwargs)

    def frame(self,o,oh,cell,m,edges,frame,do_pairs=True,do_network=False,vor=None,raw_h=None,h_owner=None):
        n=len(o);d=m['distance'];delta=m['delta'];u=m['frame'][:,0];normal=m['frame'][:,2]
        water=np.all(np.isfinite(oh),axis=(1,2))
        ii,jj=np.where(d<self.cut);rr=d[ii,jj];rh=delta[ii,jj]/rr[:,None]
        water_pair=water[ii]&water[jj]
        ci=np.einsum('ij,ij->i',u[ii],rh);cj=np.einsum('ij,ij->i',u[jj],-rh)
        cc=np.einsum('ij,ij->i',u[ii],u[jj]);plane=np.abs(np.einsum('ij,ij->i',normal[ii],normal[jj]))
        p1=normal[ii]-np.einsum('ij,ij->i',normal[ii],rh)[:,None]*rh
        p2=normal[jj]-np.einsum('ij,ij->i',normal[jj],rh)[:,None]*rh
        twist=np.abs(np.einsum('ij,ij->i',unit(p1),unit(p2)))
        donor=np.bincount(edges[:,0],minlength=n);acceptor=np.bincount(edges[:,2],minlength=n)
        hset=set(map(tuple,edges[:,[0,2]]))
        bonded=np.array([(i,j) in hset or (j,i) in hset for i,j in zip(ii,jj)],bool)
        for name,val in [('center',ci),('neighbor',cj),('bisector_pair',cc),('plane_abs',plane),('twist_abs',twist),
                         ('plane_vs_r_abs',np.abs(np.einsum('ij,ij->i',normal[ii],rh)))]:
            for label,mask in [('all',np.ones(len(ii),bool)),('first',rr<self.first),('second',rr>=self.first),('hbond',bonded),('non_hbond',~bonded)]:
                self.add(f'orientation_all_O/{name}/{label}',np.column_stack([rr[mask],val[mask]]),[self.er,self.cos],frame)
                mask=mask&water_pair
                self.add(f'orientation/{name}/{label}',np.column_stack([rr[mask],val[mask]]),[self.er,self.cos],frame)
        for h in range(2):
            cosoh=np.einsum('ij,ij->i',unit(oh)[ii,h],rh)
            self.add('orientation_all_O/OH_vs_r',np.column_stack([rr,cosoh]),[self.er,self.cos],frame)
            self.add('orientation/OH_vs_r',np.column_stack([rr[water_pair],cosoh[water_pair]]),[self.er,self.cos],frame)
        self.add('orientation/center_neighbor_joint',np.column_stack([ci[water_pair],cj[water_pair]]),[self.cos,self.cos],frame)
        self.add('orientation/distance_center_neighbor',np.column_stack([rr[water_pair],ci[water_pair],cj[water_pair]]),[np.linspace(0,self.cut,21),np.linspace(-1,1,21),np.linspace(-1,1,21)],frame)
        self.add('selection/water_pair_distance',rr[water_pair],self.er,frame)
        self.add('selection/other_pair_distance',rr[~water_pair],self.er,frame)
        for axis in range(3):
            self.add(f'isotropy/bisector_axis{axis}',u[:,axis],self.cos,frame)
            self.add(f'isotropy/normal_abs_axis{axis}',abs(normal[:,axis]),np.linspace(0,1,41),frame)
            self.add(f'isotropy/OH_axis{axis}',unit(oh)[:,:,axis],self.cos,frame)
        # Coordinates in molecular frame, symmetrized under H1<->H2.
        voxel_edges=[np.linspace(-self.cut,self.cut,25)]*3
        local=np.einsum('ij,ikj->ik',delta[ii,jj],m['frame'][ii])
        for symmetry in [np.ones(3),np.array([1,-1,-1])]:
            self.add('bodyframe/O_voxels',local*symmetry,voxel_edges,frame)
            hp=(o[:,None,:]+oh).reshape(-1,3) if raw_h is None else raw_h
            owner=np.repeat(np.arange(n),2) if h_owner is None else h_owner
            hv=mic(hp[None,:,:]-o[:,None,:],cell)
            hi,hj=np.where((np.linalg.norm(hv,axis=-1)<self.cut)&(np.arange(n)[:,None]!=owner[None,:]))
            lh=np.einsum('ij,ikj->ik',hv[hi,hj],m['frame'][hi])
            self.add('bodyframe/H_voxels',lh*symmetry,voxel_edges,frame)
        # Complete per-H geometric donor angle, not only threshold-selected bonds.
        for h in range(2):
            anglecos=np.einsum('ij,ij->i',unit(oh)[ii,h],rh)
            self.add('hbond/distance_angle',np.column_stack([rr,anglecos]),[self.er,self.cos],frame)
        for name,val,ed in [('q',m['q'],self.qedges),('coordination',m['coordination'],self.integer),
                            ('second_shell',m['second_shell'],self.integer),('lsi',m['lsi'],np.linspace(0,4,81)),
                            ('radial_cv',m['radial_cv'],np.linspace(0,1,81)),
                            ('donors',donor,self.integer),('acceptors',acceptor,self.integer),('hbond_total',donor+acceptor,self.integer)]:
            self.add('local/'+name,val,ed,frame)
        for k in range(m['nearest'].shape[1]):self.add(f'nearest/r{k+1}',m['nearest'][:,k],self.r,frame)
        if m['nearest'].shape[1]>=5:
            self.add('nearest/r5_minus_r4',m['nearest'][:,4]-m['nearest'][:,3],self.r,frame)
            self.add('joint/r4_r5',m['nearest'][:,3:5],[self.r,self.r],frame)
            self.add('joint/q_r5',np.column_stack([m['q'],m['nearest'][:,4]]),[self.qedges,self.r],frame)
        for name,x,y,ex,ey in [('q_coord',m['q'],m['coordination'],self.qedges,self.integer),
                             ('q_hbond',m['q'],donor+acceptor,self.qedges,self.integer),
                             ('donor_acceptor',donor,acceptor,self.integer,self.integer),
                             ('first_second',m['coordination'],m['second_shell'],self.integer,self.integer)]:
            self.add('joint/'+name,np.column_stack([x,y]),[ex,ey],frame)
        lengths=np.linalg.norm(oh,axis=-1)
        hoh=np.einsum('ij,ij->i',unit(oh)[:,0],unit(oh)[:,1])
        self.add('intramolecular/OH',lengths,np.linspace(.5,1.6,111),frame)
        self.add('intramolecular/HOH_cos',hoh,self.cos,frame)
        self.add('intramolecular/bond_angle',np.column_stack([lengths.mean(axis=1),hoh]),[np.linspace(.5,1.6,56),self.cos],frame)
        # Triplets retain both neighbor distances; no c3 interpretation.
        for center in range(n):
            nb=np.flatnonzero(d[center]<self.first)
            if len(nb)<2:continue
            a,b=np.triu_indices(len(nb),1);j=nb[a];k=nb[b]
            cosine=np.einsum('ij,ij->i',unit(delta[center,j]),unit(delta[center,k]))
            orient=np.einsum('ij,ij->i',u[j],u[k])
            self.add('triplet/OOO_cos',cosine,self.cos,frame)
            wp=water[center]&water[j]&water[k]
            self.add('triplet/water_OOO_cos',cosine[wp],self.cos,frame)
            for x,y in [(j,k),(k,j)]:
                self.add('triplet/r_r_cos',np.column_stack([d[center,x],d[center,y],cosine]),
                         [np.linspace(0,self.first,21)]*2+[np.linspace(-1,1,21)],frame)
            self.add('triplet_all_O/angle_neighbor_orientation',np.column_stack([cosine,orient]),[self.cos,self.cos],frame)
            self.add('triplet/angle_neighbor_orientation',np.column_stack([cosine[wp],orient[wp]]),[self.cos,self.cos],frame)
        tri_edges=[np.linspace(0,self.first,21)]*2+[np.linspace(-1,1,21)]
        dr3=np.diff(tri_edges[0]**3)/3
        expected=n*(n-1)*(n-2)/np.linalg.det(cell)**2*8*np.pi**2*dr3[:,None,None]*dr3[None,:,None]*np.diff(tri_edges[2])[None,None,:]
        self.add('triplet/r_r_cos',np.empty((0,3)),tri_edges,frame,expected=expected)
        # Per-H acceptor multiplicity identifies bifurcated geometric matches.
        multiplicity=np.bincount(edges[:,0]*2+edges[:,1],minlength=n*2)
        self.add('hbond/acceptors_per_H',multiplicity,self.integer,frame)
        # Pair-weighted raw products; connected result is assembled after global means.
        for label,x in [('q',m['q']),('coordination',m['coordination']),('hbond',donor+acceptor)]:
            good=np.isfinite(x[ii])&np.isfinite(x[jj]);count=np.histogram(rr[good],bins=self.er)[0]
            self.add('spatial/product_'+label,rr[good],self.er,frame,weights=x[ii[good]]*x[jj[good]],expected=count)
        longitudinal=np.einsum('ij,ij->i',u[ii],rh)*np.einsum('ij,ij->i',u[jj],rh)
        for label,x in [('longitudinal',longitudinal),('transverse',cc-longitudinal),('P1',cc),('P2',(3*cc**2-1)/2)]:
            good=np.isfinite(x);count=np.histogram(rr[good],bins=self.er)[0]
            self.add('spatial/orient_'+label,rr[good],self.er,frame,weights=x[good],expected=count)
        if do_pairs:self.rdf(o,oh,cell,d,frame,raw_h,h_owner)
        graph_result={}
        if do_network:
            graph=nx.Graph();graph.add_nodes_from(range(n));graph.add_edges_from(edges[:,[0,2]])
            components=list(nx.connected_components(graph));sizes=np.array([len(c) for c in components])
            self.add('network/component_size',sizes,np.arange(.5,n+1.5),frame)
            degree=np.array([graph.degree(i) for i in range(n)])
            self.add('network/degree',degree,self.integer,frame)
            self.add('network/clustering',list(nx.clustering(graph).values()),np.linspace(0,1,41),frame)
            rings=shortest_rings(n,graph.edges(),max_size=8);closed=[];winding=[]
            for ring in rings:
                v=np.sum([delta[ring[z],ring[(z+1)%len(ring)]] for z in range(len(ring))],axis=0)
                (closed if np.linalg.norm(v)<1e-5 else winding).append(ring)
            self.add('network/contractible_shortest_ring_size',[len(r) for r in closed],np.arange(2.5,9.5),frame)
            self.add('network/winding_shortest_ring_size',[len(r) for r in winding],np.arange(2.5,9.5),frame)
            graph_result={'ring_count':len(closed),'winding_ring_count':len(winding),'largest_component':int(sizes.max()),
                          'mean_clustering':float(np.mean(list(nx.clustering(graph).values())))}
            dihedral=[]
            for i,j in graph.edges():
                for a in graph[i]:
                    if a==j:continue
                    for b in graph[j]:
                        if b in (i,a):continue
                        n1=unit(np.cross(delta[i,a],delta[i,j]));n2=unit(np.cross(-delta[i,j],delta[j,b]))
                        dihedral.append(float(n1@n2))
            self.add('network/chain_dihedral_cos',dihedral,self.cos,frame)
        if vor is not None:
            for label,val,ed in [('volume',vor['volume'],np.linspace(0,500,101)),('faces',vor['faces'],self.integer),
                                 ('surface_area',vor['surface_area'],np.linspace(0,500,101))]:
                self.add('voronoi/'+label,val,ed,frame)
            self.add('joint/q_voronoi_volume',np.column_stack([m['q'],vor['volume']]),[self.qedges,np.linspace(0,500,51)],frame)
            self.add('voronoi/shape_A3_over_36piV2',vor['surface_area']**3/(36*np.pi*vor['volume']**2),np.linspace(1,8,81),frame)
        return donor,acceptor,graph_result

    def rdf(self,o,oh,cell,oo_distance,frame,raw_h=None,h_owner=None):
        n=len(o);volume=np.linalg.det(cell);shell=4*np.pi/3*np.diff(self.r**3)
        self.add('rdf/OO',oo_distance[np.isfinite(oo_distance)],self.r,frame,expected=n*(n-1)/volume*shell)
        hp=(o[:,None,:]+oh).reshape(-1,3) if raw_h is None else raw_h
        owner=np.repeat(np.arange(n),2) if h_owner is None else h_owner
        od=np.linalg.norm(mic(hp[None,:,:]-o[:,None,:],cell),axis=-1)
        mask=np.arange(n)[:,None]!=owner[None,:]
        self.add('rdf/OH',od[mask],self.r,frame,expected=np.sum(mask)/volume*shell)
        hd=np.linalg.norm(mic(hp[None,:,:]-hp[:,None,:],cell),axis=-1)
        mask=(owner[:,None]!=owner[None,:])|((owner[:,None]<0)&(owner[None,:]<0))
        np.fill_diagonal(mask,False)
        self.add('rdf/HH',hd[mask],self.r,frame,expected=np.sum(mask)/volume*shell)
