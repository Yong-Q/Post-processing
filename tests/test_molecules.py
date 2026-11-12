import numpy as np
import pytest
from waterpost.io import Frame
from waterpost.molecules import MolecularTopology, SPECS


def frame(elements, positions, mol=None, cell=None, forces=None):
    types = {'C': 1, 'H': 2, 'N': 3, 'O': 4}
    n = len(elements)
    columns = ['id', 'mol', 'type', 'x', 'y', 'z']
    data = np.column_stack([np.arange(1, n+1), np.ones(n) if mol is None else mol,
                            [types[e] for e in elements], positions])
    if forces is not None:
        columns += ['fx', 'fy', 'fz']; data = np.column_stack([data, forces])
    return Frame(0, np.diag([12.]*3) if cell is None else cell, np.zeros(3), columns, data)


def configurations():
    return {
        'H2O': ('OHH', [[0,0,0],[.8,.6,0],[-.8,.6,0]]),
        'NH3': ('NHHH', [[0,0,0],[1,0,.3],[-.5,.866,.3],[-.5,-.866,.3]]),
        'CO2': ('COO', [[0,0,0],[-1.16,0,0],[1.16,0,0]]),
        'CH4': ('CHHHH', np.vstack([[0,0,0],np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*.63])),
        'C2H4': ('CCHHHH', [[-.67,0,0],[.67,0,0],[-1.2,.92,0],[-1.2,-.92,0],[1.2,.92,0],[1.2,-.92,0]]),
        'C2H6': ('CCHHHHHH', [[-.77,0,0],[.77,0,0],[-1.1,1,0],[-1.1,-.5,.866],[-1.1,-.5,-.866],[1.1,.5,.866],[1.1,.5,-.866],[1.1,-1,0]])}


@pytest.mark.parametrize('species', list(configurations()))
def test_six_species_centers_mass_and_orientation(species):
    elements, p = configurations()[species]; p = np.asarray(p)+4
    f = frame(elements,p); t = MolecularTopology(f,species); out = t.extract(f)
    assert out['molecule_valid'].tolist() == [True]
    assert out['heavy_valid'].tolist() == [True]
    assert np.allclose(out['centers'], [4,4,4])
    masses = np.array([{'H':1.008,'C':12.011,'N':14.007,'O':15.999}[e] for e in elements])
    assert np.allclose(out['com'][0], np.average(p,axis=0,weights=masses))
    assert np.isclose(SPECS[species]['molar_mass'], masses.sum())
    if species == 'CH4': assert np.isnan(out['axis']).all()
    else: assert np.isclose(np.linalg.norm(out['axis'][0]),1)
    if species in ['H2O','NH3']: assert out['axis_kind'] == 'polar'
    if species == 'C2H6': assert len(out['geometry_samples']['torsion_rad']) == 9


def test_triclinic_independent_images_and_equivalent_carbon_permutation():
    elements, p = configurations()['C2H4']; p = np.asarray(p)+[11.7,3,4]
    cell = np.array([[12.,0,0],[4,11,0],[2,1,10]])
    f = frame(elements,p,cell=cell); a = MolecularTopology(f,'C2H4').extract(f)
    shifted = p+np.array([[2,-1,1],[-2,1,0],[0,3,0],[1,-2,1],[-1,0,-2],[3,1,-1]])@cell
    g=frame(elements,shifted,cell=cell); b=MolecularTopology(g,'C2H4').extract(g)
    assert np.allclose(a['centers'],b['centers'])
    assert np.allclose(a['relative_positions'],b['relative_positions'])
    assert np.allclose(a['com'],b['com'])
    perm=[1,0,5,4,3,2]; h=frame(elements,shifted[perm],cell=cell)
    c=MolecularTopology(h,'C2H4').extract(h)
    assert np.allclose(a['centers'],c['centers'])
    assert np.isclose(abs(a['axis'][0]@c['axis'][0]),1)
    assert np.isclose(abs(a['plane_normal'][0]@c['plane_normal'][0]),1)


def test_transfer_and_unassigned_hydrogen_preserve_raw_data():
    el,p=configurations()['H2O']; p=np.asarray(p)+3
    f=frame(el*2,np.vstack([p,p+[3,0,0]]),[1]*3+[2]*3,forces=np.ones((6,3)))
    t=MolecularTopology(f,'H2O'); f.data[2,3:6]=[5.8,3.6,3]
    o=t.extract(f)
    assert o['hydrogen_count'].tolist()==[1,3]
    assert not o['molecule_valid'].any()
    assert np.isnan(o['axis']).all()
    assert np.isfinite(o['positions']).all()
    assert np.isfinite(o['raw_molecular_force']).all()
    f.data[2,3:6]=[9,9,9]; o=t.extract(f)
    assert o['owner'][2]==-1


def test_force_and_torque_about_heavy_center():
    f=frame('OHH',np.array([[4,4,4],[4.8,4.6,4],[3.2,4.6,4]]),forces=[[0,0,0],[0,0,1],[0,0,2]])
    o=MolecularTopology(f,'H2O').extract(f)
    assert np.allclose(o['molecular_force'],[[0,0,3]])
    assert np.allclose(o['molecular_torque'],[[1.8,.8,0]])


def test_skeletal_breakup_and_local_carbon_coordination():
    el,p=configurations()['C2H6']; f=frame(el,np.asarray(p)+4)
    t=MolecularTopology(f,'C2H6'); f.data[1,3]+=3
    assert not t.extract(f)['heavy_valid'][0]
    f=frame(el,np.asarray(p)+4); t=MolecularTopology(f,'C2H6')
    f.data[2,3:6]=[5.1,3,4]
    o=t.extract(f)
    assert o['hydrogen_count'][0]==6
    assert not o['molecule_valid'][0]


def test_validation_configuration_and_constant_heavy_groups():
    f=frame('OHH',[[3,3,3],[3.8,3.6,3],[2.2,3.6,3]])
    with pytest.raises(ValueError): MolecularTopology(f,'CH4')
    with pytest.raises(ValueError): MolecularTopology(f,'H2O',type_map={1:'C',2:'H',3:'N'})
    t=MolecularTopology(f,'H2O'); f.data[0,1]=2
    with pytest.raises(ValueError): t.extract(f)


def test_no_arbitrary_azimuth_for_ammonia_and_director_for_carbon_dioxide():
    el,p=configurations()['NH3']; p=np.asarray(p)+4
    a=MolecularTopology(frame(el,p),'NH3').extract(frame(el,p))
    q=p[[0,3,1,2]]; b=MolecularTopology(frame(el,q),'NH3').extract(frame(el,q))
    assert np.allclose(a['axis'],b['axis'])
    assert np.isnan(a['plane_normal']).all()


def test_ethane_equivalent_torsions_staggered_and_eclipsed():
    el,p=configurations()['C2H6'];p=np.asarray(p)+4
    f=frame(el,p);out=MolecularTopology(f,'C2H6').extract(f)
    np.testing.assert_allclose(np.cos(3*out['geometry_samples']['torsion_rad']),-1,atol=1e-12)
    angle=np.pi/3;rotation=np.array([[1.,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    p[5:]=(p[5:]-p[1])@rotation.T+p[1]
    f=frame(el,p);out=MolecularTopology(f,'C2H6').extract(f)
    np.testing.assert_allclose(np.cos(3*out['geometry_samples']['torsion_rad']),1,atol=1e-12)


def test_ethane_mic_work_is_batched_not_repeated_per_torsion(monkeypatch):
    import waterpost.molecules as module
    actual=module.mic;calls=[]
    def counted(v,cell):calls.append(np.asarray(v).shape);return actual(v,cell)
    monkeypatch.setattr(module,'mic',counted)
    el,p=configurations()['C2H6'];f=frame(el,np.asarray(p)+4)
    module.MolecularTopology(f,'C2H6').extract(f)
    assert len(calls)<=10, 'Bond and torsion vectors must share bulk MIC calculations'
    el,p=configurations()['CO2']; p=np.asarray(p)+4
    a=MolecularTopology(frame(el,p),'CO2').extract(frame(el,p))
    q=p[[0,2,1]]; b=MolecularTopology(frame(el,q),'CO2').extract(frame(el,q))
    assert a['axis_kind']=='director'
    assert np.isclose(abs(a['axis'][0]@b['axis'][0]),1)
    assert a['hydrogen_count'].tolist()==[0]
    assert a['geometry_samples']['torsion_atom_indices'].shape==(0,4)


def test_nonzero_origin_and_type_mapping():
    f=frame('OHH',[[3,3,3],[3.8,3.6,3],[2.2,3.6,3]])
    f.data[:,2]=[8,7,7]; f.origin=np.array([100.,-50,20]); f.data[:,3:6]+=f.origin
    o=MolecularTopology(f,'H2O',type_map={7:'H',8:'O'}).extract(f)
    assert np.allclose(o['centers'],[[3,3,3]])
    assert np.allclose(o['positions'][0],[3,3,3])


def test_geometry_indices_and_fixed_distance_guard():
    el,p=configurations()['C2H6']; f=frame(el,np.asarray(p)+4)
    t=MolecularTopology(f,'C2H6',mode='fixed'); out=t.extract(f)
    g=out['geometry_samples']
    assert len(g['bond_lengths_A'])==7
    assert len(g['angle_cos'])==12
    assert np.array_equal(g['torsion_molecule_index'],np.zeros(9,int))
    assert np.all(g['torsion_atom_indices'][:,1:3]==[0,1])
    f.data[2,3:6]=[8,8,8]
    bad=t.extract(f)
    assert not bad['molecule_valid'][0]
    assert np.isfinite(bad['geometry_samples']['bond_lengths_A']).all()


def test_constant_atom_id_type_and_composition():
    f=frame('OHH',[[3,3,3],[3.8,3.6,3],[2.2,3.6,3]])
    t=MolecularTopology(f,'H2O'); f.data[1,0]=99
    with pytest.raises(ValueError,match='id'): t.extract(f)
    f.data[1,0]=2; f.data[1,2]=1
    with pytest.raises(ValueError,match='type'): t.extract(f)
    f=frame('CCHHHHHH',np.zeros((8,3)),mol=[1,1,1,1,1,1,1,2])
    with pytest.raises(ValueError,match='composition'): MolecularTopology(f,'C2H6')
