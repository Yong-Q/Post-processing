import inspect
import numpy as np
from waterpost.features import encode
from waterpost.histograms import Histograms
from waterpost.analysis import Statistics
from waterpost.static import local_metrics


def test_water_only_encoding_omits_defect_neighbors_but_context_retains_them():
    assert 'neighbor_water_only' in inspect.signature(encode).parameters
    o=np.array([[0.,0,0],[2.8,0,0],[0,2.8,0]])
    oh=np.tile(np.array([[.96,0,0],[-.24,.93,0]]),(3,1,1));oh[2]=np.nan
    cell=np.eye(3)*20
    general=encode(o,oh,cell)
    water=encode(o,oh,cell,neighbor_water_only=True)
    assert water['names']==general['names']
    assert np.all(water['invariants'][0,:4]<general['invariants'][0,:4])
    assert water['valid_water_neighbors'].tolist()==[1,1,2]
    assert water['other_oxygen_neighbors'].tolist()==[1,1,0]
    assert np.isnan(water['invariants'][2]).all()


def test_water_pair_angles_do_not_include_nonwater_oxygen_neighbors():
    o=np.array([[0.,0,0],[2.8,0,0],[0,2.8,0]])
    oh=np.tile(np.array([[.96,0,0],[-.24,.93,0]]),(3,1,1));oh[2]=np.nan
    cell=np.eye(3)*20;m=local_metrics(o,oh,cell,3.5,6.)
    hist=Histograms();s=Statistics(hist,3.5,6.,8.)
    s.frame(o,oh,cell,m,np.empty((0,3),int),0,do_pairs=False)
    # Only the two directed pairs between the two actual H2O centers.
    assert hist.result('orientation/center/all')['counts'].sum()==2
    # Context statistics preserve the 4 center/neighbor directions with a valid central water.
    assert hist.result('orientation_all_O/center/all')['counts'].sum()==4


def test_molecular_rotation_does_not_cross_an_observed_h_exchange():
    import waterpost.dynamics as d
    assert hasattr(d,'persistent_water_orientation')
    u=np.array([[[1.,0,0]],[[1.,0,0]],[[1.,0,0]]])
    ids=np.array([[[2,3]],[[3,4]],[[2,3]]])
    result=d.persistent_water_orientation(u,ids,[0,1,2])
    assert result['surviving_water_pairs'].tolist()==[3,0,0]
    assert np.isnan(result['c1'][1:]).all()
