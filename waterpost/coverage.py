"""Honest connection to the 86-item discussion inventory."""
from pathlib import Path
import re


def coverage(report,output):
    inventory=Path(__file__).resolve().parents[1]/'docs'/'water_MD_property_inventory_20261111.md'
    rows=[]
    if inventory.exists():
        for line in inventory.read_text().splitlines():
            match=re.match(r'\| ([A-Z]\d\d) \| ([^|]+) \|',line)
            if match:rows.append((match[1],match[2].strip()))
    deferred={
        'Q02':'Requires original thermodynamic log; dump provides no temperature/energy/pressure.',
        'Q06':'Cross-trajectory comparison is available through compare command; sample duplicate audit is descriptor-based, not a proof of full coverage.',
        'D11':'Four-point dynamical susceptibility requires an explicit overlap/ensemble definition; not enabled by this local-feature task.',
        'M06':'Model-training ablation needs a defined target; no neural model training requested.',
        'F01':'OZ/c2 inversion intentionally not run; finite-box/tail/ensemble choices require a separate configuration.',
        'F02':'Full orientation-dependent OZ is not a scalar OO inversion and is outside current postprocessing.',
        'F03':'No free-energy/chemical-potential labels in coordinate dump.',
        'F04':'No new state points or external-field MD requested.',
        'F05':'c3/c4 explicitly deferred by user.',
        'F06':'No learned free-energy functional or independent thermodynamic benchmarks supplied.'}
    partial={
        'O07':'Invariant relative-plane and twist measures plus raw full molecular frames saved; no unique Euler angles under water symmetry.',
        'T05':'Radial coefficient of variation saved with explicit definition; not mislabeled as a universal radial tetrahedral order.',
        'T09':'OOO g3 versus Kirkwood pair product, with finite bins; not a direct correlation c3.',
        'H07':'Deduplicated shortest alternate-path cycles per edge up to 8 nodes; winding cycles separate; not exhaustive primitive rings.',
        'H08':'Local HB chain dihedral cosines and relative orientation; not all possible long paths.',
        'C07':'OO position pair-entropy approximation only; not full molecular entropy.',
        'C08':'Exploratory low-dimensional plug-in mutual information; no high-dimensional entropy or bias-free estimator.',
        'D01':'MSD implemented; diffusion must use explicit user-selected linear fit window; no automatic unqualified D.',
        'D03':'F_s/F calculated for a documented subset of box modes, not a continuous wavevector surface.',
        'D07':'Intermittent correlation and continuous-at-saved-frames survival only; fast missed breaks are unresolved.',
        'D09':'D/A environment transition counts at several lags; no validated Markov model.',
        'M04':'Feature PCA, marginals and covariance plus compare command; embedding is descriptive, not complete coverage proof.',
        'M07':'All cutoffs/bin/basis parameters exposed for separate runs; a sensitivity conclusion requires comparing those runs.'}
    partial.update({'Q05':'Block-length scan and block-bootstrap of frame-averaged scalar observables; confidence requires an error plateau.',
                    'Q06':'Compare command keeps states separate; no comprehensive duplicate-configuration detector.',
                    'C05':'q, coordination and HB connected spatial products; Voronoi volume spatial correlation not computed.',
                    'D05':'Finite-lag angle distributions and C1/C2 only; no unqualified infinite-time relaxation fit.',
                    'D06':'First/second-shell pair survival and intermittency, describing exchange; not an exact continuous residence-time distribution.',
                    'M05':'Descriptor-target Pearson associations; no trained reconstruction decoder.',
                    'M08':'Saved whole-frame temporal roles with configurable gap; whole-state holdouts require batch-level selection.'})
    destinations={'Q':'report.json; dynamics.h5/sampling','R':'environment.h5; distributions.h5/rdf,nearest,local,intramolecular',
                  'O':'distributions.h5/orientation,bodyframe,isotropy; environment.h5/features',
                  'T':'environment.h5/q,lsi,expensive; distributions.h5/triplet,joint',
                  'H':'environment.h5/neighbors; distributions.h5/hbond,network,joint',
                  'C':'collective.h5; distributions.h5/spatial,rdf',
                  'D':'dynamics.h5','E':'environment.h5/forces,molecule_force,molecule_torque_about_O',
                  'M':'environment.h5/features,equivariant_coefficients,neighbors; collective.h5/feature_statistics',
                  'F':'not run'}
    result=[]
    for key,title in rows:
        status='implemented';reason='See definitions, analysis strides and finite-sample limitations.'
        if key in deferred:status='requires_separate_data_or_task';reason=deferred[key]
        if key in partial:status='implemented_with_scope';reason=partial[key]
        if key=='R07' and not report['voronoi_frames']:status='skipped_missing_dependency';reason='Install optional freud-analysis.'
        if key=='F01' and 'direct_correlations' in report:
            reason='Scalar OO RDF -> OZ -> c2(r), c2_hat(k), planar projection; finite cutoff and fixed-N effects explicit. See direct_correlations.h5.'
            status='implemented_with_scope' if report['direct_correlations']['status']!='invalid_positive_k_spectrum' else 'diagnostic_only_invalid_spectrum'
        if key.startswith('E'):
            if key in ['E02','E03']:
                status='implemented' if all(f in report['source_columns'] for f in ['fx','fy','fz']) else 'skipped_missing_fields'
                reason='Forces in declared native units; torque about central O.'
            else:
                status='requires_separate_data_or_task';reason='Needs matching energy/log, dipole/polarizability, high-frequency velocity/stress/heat-current fields or ensemble validation; no surrogate values emitted.'
        if key.startswith('D') and key!='D11':
            import h5py
            with h5py.File(Path(output)/'dynamics.h5') as f:
                if 'oxygen_motion' not in f:
                    status='skipped_runtime';reason=str(f.attrs.get('status','No valid dynamics output'))
        result.append({'id':key,'item':title,'status':status,'output':destinations[key[0]],'scope':reason})
    return result
