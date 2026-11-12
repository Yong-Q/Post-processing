import numpy as np


def mic(vectors, cell):
    """Shortest periodic displacement; cell vectors are rows, units unchanged."""
    a = np.asarray(vectors, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if np.allclose(cell, np.diag(np.diag(cell)), atol=1e-12):
        return a - np.rint(a / np.diag(cell)) * np.diag(cell)
    from ase.geometry import find_mic
    shape = a.shape
    return find_mic(a.reshape(-1, 3), cell, pbc=True)[0].reshape(shape)


def unit(v):
    a = np.asarray(v, dtype=float)
    length = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.divide(a, length, out=np.full_like(a, np.nan), where=length > 1e-12)


def tetrahedral_q(vectors):
    if len(vectors) != 4:
        return np.nan
    u = unit(vectors)
    angles = (u @ u.T)[np.triu_indices(4, 1)]
    return float(1 - 3/8 * np.sum((angles + 1/3)**2))


def water_frame(oh):
    """Rows: bisector u, H-label-odd in-plane v, H-label-odd normal n."""
    bonds = unit(oh)
    u = unit(bonds[:, 0] + bonds[:, 1])
    n = unit(np.cross(bonds[:, 0], bonds[:, 1]))
    v = unit(np.cross(n, u))
    return np.stack([u, v, n], axis=1)


def force_torque(relative_positions, forces):
    return np.sum(forces, axis=-2), np.sum(np.cross(relative_positions, forces), axis=-2)


def unique_radius(cell):
    """Conservative inscribed sphere: half the minimum box-face height."""
    inv = np.linalg.inv(cell)
    return .5 / np.max(np.linalg.norm(inv, axis=0))
