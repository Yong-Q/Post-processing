"""Strictly convex scalar finite-moment free energy for an exploratory pilot.

This is a Legendre effective free energy, not an unrestricted molecular Fex.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def whiten(observations):
    a=np.asarray(observations,float);mean=a.mean(axis=0);centered=a-mean
    covariance=centered.T@centered/len(a)
    eigen,vectors=np.linalg.eigh(covariance)
    if not np.isfinite(eigen).all() or eigen[0]<=0 or eigen[-1]/eigen[0]>1e8:
        raise ValueError('Unsupported or ill-conditioned covariance; no clipping/pseudoinverse labels')
    s=(vectors*np.sqrt(eigen))@vectors.T
    inverse=(vectors/np.sqrt(eigen))@vectors.T
    return mean,s,inverse


class ConvexMomentFreeEnergy(nn.Module):
    def __init__(self,dimensions,width=16):
        super().__init__();self.dimensions=dimensions
        initial=torch.zeros(dimensions,dimensions)
        initial.diagonal().fill_(float(np.log(np.expm1(np.sqrt(.999)))))
        self.raw_cholesky=nn.Parameter(initial)
        self.hidden=nn.Linear(dimensions,width)
        self.positive_readout=nn.Parameter(torch.full((width,),-3.))

    def nonlinear(self,x):
        return .03*(F.softplus(self.hidden(x))*F.softplus(self.positive_readout)).sum(-1)

    def forward(self,x):
        lower=torch.tril(self.raw_cholesky,-1)+torch.diag(F.softplus(self.raw_cholesky.diagonal()))
        zero=x.new_zeros((1,self.dimensions),requires_grad=True)
        f0=self.nonlinear(zero)
        grad0=torch.autograd.grad(f0.sum(),zero,create_graph=True)[0]
        return .5*(x@lower).square().sum(-1)+.0005*x.square().sum(-1)+self.nonlinear(x)-f0-(x*grad0).sum(-1)


def derivatives(model,x,create_graph=False):
    if not x.requires_grad:x=x.detach().requires_grad_(True)
    value=model(x)
    gradient=torch.autograd.grad(value.sum(),x,create_graph=True)[0]
    hessian=torch.stack([torch.autograd.grad(gradient[:,j].sum(),x,create_graph=create_graph,retain_graph=True)[0]
                         for j in range(x.shape[1])],dim=1)
    return value,gradient,hessian


def density_response(s,hessian,attenuation):
    """Relax all environment coordinates, then project to bare density modes."""
    projection=np.asarray(s,float)[:2]/attenuation
    return projection@np.linalg.solve(np.asarray(hessian,float),projection.T)
