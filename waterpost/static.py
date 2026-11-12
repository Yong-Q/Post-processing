from collections import deque
import numpy as np
import networkx as nx
from scipy.special import sph_harm
from .geometry import mic,unit,water_frame


def hydrogen_bonds(o,oh,cell,cutoff=3.5,angle=30):
    delta=mic(o[None,:,:]-o[:,None,:],cell)
    r=np.linalg.norm(delta,axis=-1);np.fill_diagonal(r,np.inf)
    donors,acceptors=np.where(r<cutoff)
    if len(donors)==0:return np.empty((0,3),dtype=int)
    rh=delta[donors,acceptors]/r[donors,acceptors,None]
    cosine=np.einsum('ihd,id->ih',unit(oh)[donors],rh)
    pairs,h=np.where(cosine>=np.cos(np.deg2rad(angle)))
    return np.column_stack([donors[pairs],h,acceptors[pairs]]).astype(int)


def local_metrics(o,oh,cell,first_cut=3.5,env_cut=6.,lsi_cut=3.7):
    delta=mic(o[None,:,:]-o[:,None,:],cell)
    distance=np.linalg.norm(delta,axis=-1);np.fill_diagonal(distance,np.inf)
    order=np.argsort(distance,axis=1);nearest=np.take_along_axis(distance,order,axis=1)
    n=len(o);nn=min(6,n-1)
    frame=water_frame(oh)
    q=np.full(n,np.nan);radial_cv=np.full(n,np.nan)
    if n>=5:
        v=unit(delta[np.arange(n)[:,None],order[:,:4]])
        dot=np.einsum('nid,njd->nij',v,v)[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]]
        q=1-3/8*np.sum((dot+1/3)**2,axis=1)
        radial_cv=np.std(nearest[:,:4],axis=1)/np.mean(nearest[:,:4],axis=1)
    lsi=np.full(n,np.nan)
    for i in range(n):
        count=int(np.sum(nearest[i]<lsi_cut))
        if 0<count<n-1:lsi[i]=np.var(np.diff(nearest[i,:count+1]))
    return {'delta':delta,'distance':distance,'order':order,'nearest':nearest[:,:nn],
            'coordination':np.sum(distance<first_cut,axis=1),
            'second_shell':np.sum((distance>=first_cut)&(distance<env_cut),axis=1),
            'q':q,'radial_cv':radial_cv,'lsi':lsi,'frame':frame}


def steinhardt(delta,distance,cutoff,l):
    n=len(distance);qlm=np.zeros((n,2*l+1),dtype=complex)
    degree=np.sum(distance<cutoff,axis=1)
    i,j=np.where(distance<cutoff);v=delta[i,j];r=distance[i,j]
    theta=np.arccos(np.clip(v[:,2]/r,-1,1));phi=np.arctan2(v[:,1],v[:,0])
    for mi,m in enumerate(range(-l,l+1)):
        np.add.at(qlm[:,mi],i,sph_harm(m,l,phi,theta))
    np.divide(qlm,degree[:,None],out=qlm,where=degree[:,None]>0)
    q=np.sqrt(4*np.pi/(2*l+1)*np.sum(abs(qlm)**2,axis=1));q[degree==0]=np.nan
    avg=qlm.copy();np.add.at(avg,i,qlm[j]);avg/=(degree+1)[:,None]
    qbar=np.sqrt(4*np.pi/(2*l+1)*np.sum(abs(avg)**2,axis=1));qbar[degree==0]=np.nan
    return q,qbar


def shortest_rings(n,edges,max_size=8):
    """Deduplicated shortest alternate-path cycles per undirected edge.

    Not an enumeration of all primitive rings. Bounded BFS avoids exponential
    simple-cycle enumeration. Molecular IDs canonicalized only for counting.
    """
    graph=nx.Graph();graph.add_nodes_from(range(n));graph.add_edges_from(edges)
    cycles=set()
    for a,b in sorted(graph.edges()):
        queue=deque([(a,[a])]);seen={a}
        while queue:
            node,path=queue.popleft()
            if len(path)>=max_size:continue
            found=False
            for v in sorted(graph[node]):
                if {node,v}=={a,b}:continue
                if v==b:
                    cycle=path+[b]
                    start=cycle.index(min(cycle));forward=cycle[start:]+cycle[:start]
                    reverse=[forward[0]]+list(reversed(forward[1:]))
                    cycles.add(min(tuple(forward),tuple(reverse)));found=True;break
                if v not in seen:seen.add(v);queue.append((v,path+[v]))
            if found:break
    return sorted(cycles)


def periodic_voronoi(o,cell):
    try:
        import freud
    except ImportError as e:
        raise ImportError('Periodic Voronoi requires optional freud-analysis; install with the supplied setup command') from e
    box=freud.box.Box.from_matrix(cell.T)
    positions=box.wrap(o-cell.sum(axis=0)/2)
    vor=freud.locality.Voronoi().compute((box,positions))
    return {'volume':np.asarray(vor.volumes),'faces':np.bincount(vor.nlist.query_point_indices,minlength=len(o)),
            'surface_area':np.bincount(vor.nlist.query_point_indices,weights=vor.nlist.weights,minlength=len(o))}
