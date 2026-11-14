import numpy as np
import pytest
torch=pytest.importorskip('torch')
from waterpost.xi_model import ConvexMomentFreeEnergy,derivatives,whiten,density_response


def test_scalar_functional_gradient_hessian_finite_difference():
    torch.manual_seed(7);model=ConvexMomentFreeEnergy(3,8).double()
    x=torch.tensor([[.2,-.1,.3]],dtype=torch.float64,requires_grad=True)
    value,g,h=derivatives(model,x,create_graph=True)
    assert value.shape==(1,) and h.shape==(1,3,3)
    np.testing.assert_allclose(h.detach()[0],h.detach()[0].T,atol=1e-12)
    assert torch.linalg.eigvalsh(h.detach()).min()>0
    eps=1e-5
    for j in range(3):
        plus=x.detach().clone();minus=plus.clone();plus[0,j]+=eps;minus[0,j]-=eps
        numeric=(model(plus)-model(minus))/(2*eps)
        np.testing.assert_allclose(numeric.detach(),g.detach()[:,j],rtol=1e-5,atol=1e-8)
    zero=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
    v0,g0,_=derivatives(model,zero)
    np.testing.assert_allclose(v0.detach(),0,atol=1e-12);np.testing.assert_allclose(g0.detach(),0,atol=1e-12)
    loss=(h@torch.eye(3,dtype=torch.float64)*1.2-torch.eye(3,dtype=torch.float64)).square().mean()
    loss.backward();assert model.raw_cholesky.grad.norm()>0


def test_whitening_and_bias_chain_rule():
    rng=np.random.default_rng(42);a=rng.normal(size=(1000,3))*[1,2,3]
    mean,s,inverse=whiten(a);x=(a-mean)@inverse
    np.testing.assert_allclose(x.T@x/len(x),np.eye(3),atol=1e-12)
    v=np.array([.1,-.05,.2])
    np.testing.assert_allclose((a-mean)@v,x@(s@v),atol=1e-12)
    with pytest.raises(ValueError):whiten(np.ones((30,3)))


def test_gaussian_tilt_kernel_scale_and_relaxed_density_response():
    c=np.array([[2.,.3,.7],[.3,1.5,.5],[.7,.5,2.]])
    eigen,v=np.linalg.eigh(c);s=(v*np.sqrt(eigen))@v.T;lam=.75
    physical=np.array([.2,-.1]);conjugate=np.r_[-physical/lam,0.]
    shifted=c@conjugate;white=np.linalg.solve(s,shifted)
    np.testing.assert_allclose(white,s.T@conjugate,atol=1e-12)
    np.testing.assert_allclose(density_response(s,np.eye(3),lam),c[:2,:2]/lam**2,atol=1e-12)
    h=np.linalg.inv(c);schur=h[:2,:2]-h[:2,2:]@np.linalg.solve(h[2:,2:],h[2:,:2])
    np.testing.assert_allclose(schur,np.linalg.inv(c[:2,:2]),atol=1e-12)


def test_generated_weights_are_exact_known_field_ratio():
    from waterpost.xi_pilot import state_statistics
    from waterpost.reweight import weights
    rng=np.random.default_rng(71);a=rng.normal(size=(400,5));mean,s,inverse=whiten(a)
    lam=.75;physical=np.array([.01,-.02]);cfg={'beta_field_coefficients':physical.tolist()}
    st=state_statistics(a,[cfg],mean,s,inverse,lam)[0]
    w=weights(-a[:,:2]@physical/lam)
    np.testing.assert_allclose(st['mean'],w@((a-mean)@inverse),atol=1e-14)
    np.testing.assert_allclose(st['gradient'],-s.T@np.r_[physical/lam,np.zeros(3)],atol=1e-14)


def test_tiny_gaussian_fit_and_final_checkpoint_path(tmp_path):
    import json
    from waterpost.xi_pilot import field_configurations,fit_one
    rng=np.random.default_rng(24);a=rng.normal(size=(2001,5))
    np.savez(tmp_path/'observables.npz',moments=a)
    (tmp_path/'representation_checks.json').write_text(json.dumps({'attenuation':.75}))
    (tmp_path/'field_configurations.json').write_text(json.dumps(field_configurations(a[:1000,:2],.75)))
    result=fit_one('xi',str(tmp_path),epochs=2)
    assert result['status']=='completed',result
    assert result['test_cases']==9 and result['test_converged']>=8
    assert result['independent_target_validation'] is False
    assert (tmp_path/'xi/best.pt').is_file()
