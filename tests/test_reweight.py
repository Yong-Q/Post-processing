import numpy as np
import pytest
from waterpost.reweight import (weights, log_weights, periodic_shape,
    block_estimate, response_c2, tangent_basis, overlap_diagnostics, KB, window_cumulants)


def test_weights_hand_calculation_and_energy_zero():
    u=np.array([1.,3.]);t=300.;target=600.
    expected=np.array([1.,np.exp((1/(KB*t)-1/(KB*target))*2)])
    expected/=expected.sum()
    np.testing.assert_allclose(weights(log_weights(u,t,target)),expected)
    np.testing.assert_allclose(weights(log_weights(u+1e5,t,target)),expected)
    np.testing.assert_allclose(log_weights(u,t,t,field=np.array([2.,4.]),a=.1),[-.2,-.4])
    with pytest.raises(ValueError):log_weights(u,t,0.)


def test_periodic_shape_triclinic_images_and_phase():
    cell=np.array([[8.,0.,0.],[2.,9.,0.],[1.,2.,10.]])
    s=np.array([[0.,.25,.5],[.25,.5,.75]])
    np.testing.assert_allclose(periodic_shape(s@cell,cell,1,'sin1'),[1.,0.],atol=1e-14)
    np.testing.assert_allclose(periodic_shape(s@cell+np.array([2,-3,1])@cell,cell,1,'sin1'),[1.,0.],atol=1e-14)


def test_block_jackknife_mean_sem_and_paired_zero():
    x=np.arange(20.).reshape(10,2);w=np.ones(10)/10
    value,se,leave=block_estimate(x,w,2)
    np.testing.assert_allclose(value,x.mean(axis=0))
    np.testing.assert_allclose(se,x.reshape(5,2,2).mean(axis=1).std(axis=0,ddof=1)/np.sqrt(5))
    assert leave.shape==(5,2)
    assert overlap_diagnostics(w,2)['frame_ess_fraction']==pytest.approx(1.)
    assert overlap_diagnostics(weights(np.arange(10.)*10),2)['pass'] is False


def test_multinomial_ideal_gas_response_is_zero():
    # Exact multinomial N=2,m=3 enumeration, probabilities p.
    import itertools,math
    p=np.array([.2,.3,.5]);counts=[];probs=[]
    for assignment in itertools.product(range(3),repeat=2):
        counts.append(np.bincount(assignment,minlength=3));probs.append(np.prod(p[list(assignment)]))
    counts=np.asarray(counts,float);probs=np.array(probs)
    c=response_c2(counts,probs)
    np.testing.assert_allclose(c,0,atol=1e-14)
    q=tangent_basis(3)
    np.testing.assert_allclose(q.T@q,np.eye(2),atol=1e-14)
    np.testing.assert_allclose(q.T@np.ones(3),0,atol=1e-14)
    with pytest.raises(ValueError):response_c2(np.zeros((10,3)),np.ones(10)/10)


def test_window_cumulants_exact_poisson_moments():
    lam=2.
    moments=np.array([lam,lam**2+lam,lam**3+3*lam**2+lam,lam**4+6*lam**3+7*lam**2+lam,np.exp(-lam)])
    np.testing.assert_allclose(window_cumulants(moments),[lam,lam,lam,lam,np.exp(-lam)])
