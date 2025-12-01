import numpy as np
from dolfinx import mesh

def focal(r_long_endo: float, r_short_endo: float):
    return np.sqrt(r_long_endo**2 - r_short_endo**2)

def full_arctangent(x, y):
    t = np.arctan2(x, y)
    if t < 0:
        return t + 2 * np.pi
    else:
        return t
    
def get_level(regions, mu):
    A = np.intersect1d(
        np.where((regions.T[3] <= mu))[0],
        np.where((mu <= regions.T[0]))[0],
    )
    if len(A) == 0:
        return [np.shape(regions)[0] + 1]
    else:
        return A

def get_sector(regions, theta):
    if not (np.count_nonzero(regions.T[1] <= regions.T[2]) >= 0.5 * np.shape(regions)[0]):
        raise ValueError("Surfaces are flipped")

    sectors = []
    for i, r in enumerate(regions):
        if r[1] == r[2]:
            sectors.append(i)
        else:
            if r[1] > r[2]:
                if theta > r[1] or r[2] > theta:
                    sectors.append(i)

            else:
                if r[1] < theta < r[2]:
                    sectors.append(i)

    return sectors    

def cartesian_to_prolate_ellipsoidal(x, y, z, a):
    b1 = np.sqrt((x + a) ** 2 + y**2 + z**2)
    b2 = np.sqrt((x - a) ** 2 + y**2 + z**2)

    sigma = 1 / (2.0 * a) * (b1 + b2)
    tau = 1 / (2.0 * a) * (b1 - b2)
    phi = full_arctangent(z, y)
    nu = np.arccosh(sigma)
    mu = np.arccos(tau)
    return nu, mu, phi

def get_aha_segments(domain, foc, mu_base, dmu):
    
    tdim = domain.topology.dim
    num_local_cells = domain.topology.index_map(tdim).size_local

    # Cell midpoints
    mid = mesh.compute_midpoints(domain, tdim, np.arange(num_local_cells))

    segarray = np.zeros(num_local_cells)

    # -------------------------------------------------------
    # Loop through cells, classify geometrically, assign value
    # -------------------------------------------------------
    for c in range(num_local_cells):

        x, y, z = mid[c]
        nu, mu, phi = cartesian_to_prolate_ellipsoidal(x, y, z, a=foc)

        segment = None
        
        #  classify geometrically
        if mu_base < mu <= mu_base + dmu:
            # BASE (sectors 1–6)
            if 0 < phi <= np.pi/3:
                segment = 1
            elif np.pi/3 < phi <= 2*np.pi/3:
                segment = 2
            elif 2*np.pi/3 < phi <= np.pi:
                segment = 3
            elif np.pi < phi <= 4*np.pi/3:
                segment = 4
            elif 4*np.pi/3 < phi <= 5*np.pi/3:
                segment = 5
            else:
                segment = 6

        elif mu_base + dmu < mu <= mu_base + 2*dmu:
            # MID (sectors 7–12)
            if 0 < phi <= np.pi/3:
                segment = 7
            elif np.pi/3 < phi <= 2*np.pi/3:
                segment = 8
            elif 2*np.pi/3 < phi <= np.pi:
                segment = 9
            elif np.pi < phi <= 4*np.pi/3:
                segment = 10
            elif 4*np.pi/3 < phi <= 5*np.pi/3:
                segment = 11
            else:
                segment = 12

        elif mu_base + 2*dmu < mu <= mu_base + 3*dmu:
            # APICAL (13–16)
            if 0 < phi <= np.pi/2:
                segment = 13
            elif np.pi/2 < phi <= np.pi:
                segment = 14
            elif np.pi < phi <= 3*np.pi/2:
                segment = 15
            else:
                segment = 16
        else:
            # APEX CAP
            segment = 17

        segarray[c] = segment
    
    return segarray