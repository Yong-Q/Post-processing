import numpy as np
from waterpost.io import Frame,WaterTopology


def test_geometry_mode_flags_nonwater_and_preserves_raw_hydrogen():
    columns=['id','mol','type','x','y','z']
    # Original mol 1 lost its second H to O2. O2 now has three H; neither is H2O.
    rows=np.array([[1,1,4,1,1,1],[2,1,2,1.9,1,1],[3,1,2,5,1.9,1],
                   [4,2,4,5,1,1],[5,2,2,5.9,1,1],[6,2,2,5,1,1.9]])
    frame=Frame(0,np.eye(3)*20,np.zeros(3),columns,rows)
    try: top=WaterTopology(frame,4,2,mode='nearest',bond_cut=1.35)
    except TypeError:
        import pytest
        pytest.fail('Missing reactive topology mode')
    w=top.extract(frame)
    np.testing.assert_array_equal(w['assigned_h_count'],[1,3])
    assert not np.any(w['water_valid'])
    assert np.all(np.isnan(w['oh']))
    assert w['raw_h_positions'].shape==(4,3)
    assert len(w['unassigned_h_ids'])==0


def test_free_h_is_not_forced_into_a_water():
    columns=['id','mol','type','x','y','z']
    rows=np.array([[1,1,4,1,1,1],[2,1,2,1.9,1,1],[3,1,2,8,8,8]])
    frame=Frame(0,np.eye(3)*20,np.zeros(3),columns,rows)
    try: w=WaterTopology(frame,4,2,mode='nearest',bond_cut=1.35).extract(frame)
    except TypeError:
        import pytest
        pytest.fail('Missing reactive topology mode')
    assert w['unassigned_h_ids'].tolist()==[3]
    assert w['assigned_h_count'].tolist()==[1]


def test_oh_time_correlation_matches_h_identity_across_slot_exchange():
    import waterpost.dynamics as d
    assert hasattr(d,'bond_orientation'), 'Missing identity-aware OH correlation'
    oh=np.array([[[[1.,0,0],[0,1,0]]],[[[0,1.,0],[1.,0,0]]]])
    ids=np.array([[[2,3]],[[3,2]]])
    out=d.bond_orientation(oh,ids,[0,1])
    np.testing.assert_allclose(out['c1'],[1,1])
    assert out['matched_bonds'][1]==2
