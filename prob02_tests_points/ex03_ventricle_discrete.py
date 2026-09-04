"""
ex03_ventricle_discrete.py
--------------------------
Forward problem: passive ventricular filling.
Generates synthetic displacement and deformation gradient data
at selected nodes, to be used as reference in the inverse problem (ex04).
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI

import dolfinx
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from ufl import inner

import cardiac_geometries

from cardiac_utils import (
    compute_Fh,
    save_results_xdmf,
    save_selected_nodes,
    select_distributed_nodes,
)

print(dolfinx.__version__)


# =============================================================================
# forward problem
# =============================================================================

def prob_ventricle_passive_filling(geo, ndofs_data=32):
    """
    Solves the passive filling forward problem and returns synthetic data.

    Parameters
    ----------
    geo        : cardiac_geometries geometry object
    ndofs_data : int — number of measurement nodes to select (must
                 match the value used later in ex04 for the inverse
                 problem, since both need the SAME set of selected
                 nodes -- same N and same seed -- to be consistent)

    Returns
    -------
    u_mat : np.ndarray, shape (Npoints, 3)  — displacement at selected nodes
    Fd    : np.ndarray, shape (Npoints, 3, 3) — deformation gradient at selected nodes
    CC    : dolfinx.fem.Function — true parameter field
    """
    print(f"\nsolving forward problem to generate synthetic data")

    # -------------------------------------------------------------------------
    # mesh
    # -------------------------------------------------------------------------

    domain     = geo.mesh
    facet_tags = geo.ffun
    markers    = geo.markers

    print(markers)

    xdmf = XDMFFile(domain.comm, "lv_ellipsoid/fiber.xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(geo.f0)
    xdmf.close()

    # -------------------------------------------------------------------------
    # function spaces
    # -------------------------------------------------------------------------

    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    v = ufl.TestFunction(V)
    u = fem.Function(V)

    # -------------------------------------------------------------------------
    # boundary conditions (base fixed)
    # -------------------------------------------------------------------------

    u_bc      = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
    base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
    bcs       = [fem.dirichletbc(u_bc, base_dofs, V)]

    # -------------------------------------------------------------------------
    # nodes selected for data measurement
    # -------------------------------------------------------------------------

    dim        = domain.geometry.dim
    dof_coords = V.tabulate_dof_coordinates().reshape(-1, dim)

    selected_dofs, selected_coords = select_distributed_nodes(
        dof_coords, N=ndofs_data, min_dist=4.0, seed=42
    )

    Npoints = len(selected_dofs)
    print(f"Número de nodes: {Npoints}")
    print("Coordenadas nodes selecionados:")
    print(selected_coords)

    save_selected_nodes(selected_coords, selected_dofs)

    # -------------------------------------------------------------------------
    # kinematics
    # -------------------------------------------------------------------------

    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    J = ufl.variable(ufl.det(F))

    f0 = geo.f0
    s0 = geo.s0
    n0 = geo.n0

    # -------------------------------------------------------------------------
    # true parameter field (spatially varying CC)
    # -------------------------------------------------------------------------

    def c_expr(x):
        xx = x[0]
        return 2.0 + (xx - xx.min()) / (xx.max() - xx.min())

    Va = fem.functionspace(domain, ("Lagrange", 1))
    CC = fem.Function(Va)
    CC.interpolate(c_expr)

    # -------------------------------------------------------------------------
    # constitutive model (Holzapfel-Ogden passive + volumetric)
    # -------------------------------------------------------------------------

    bf    = default_scalar_type(6.6)
    bt    = default_scalar_type(4.0)
    bfs   = default_scalar_type(2.6)
    kappa = fem.Constant(domain, 1e2)

    e1, e2, e3 = f0, s0, n0
    Cs  = J**(-2/3) * F.T * F
    Es  = 0.5 * (Cs - I)

    E11 = ufl.inner(Es*e1, e1)
    E12 = ufl.inner(Es*e1, e2)
    E13 = ufl.inner(Es*e1, e3)
    E21 = ufl.inner(Es*e2, e1)
    E22 = ufl.inner(Es*e2, e2)
    E23 = ufl.inner(Es*e2, e3)
    E31 = ufl.inner(Es*e3, e1)
    E32 = ufl.inner(Es*e3, e2)
    E33 = ufl.inner(Es*e3, e3)

    Q = (
        bf  * E11**2
        + bt  * (E22**2 + E33**2 + E23**2 + E32**2)
        + bfs * (E12**2 + E21**2 + E13**2 + E31**2)
    )

    Wpassive      = CC / 2.0 * (ufl.exp(Q) - 1)
    Wvolume       = kappa * (J * ufl.ln(J) - J + 1)
    strain_energy = Wpassive + Wvolume
    P             = ufl.diff(strain_energy, F)

    # -------------------------------------------------------------------------
    # variational form (internal + endocardial pressure)
    # -------------------------------------------------------------------------

    p_endo   = fem.Constant(domain, 0.0)
    metadata = {"quadrature_degree": 4}
    ds       = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata=metadata)
    dx       = ufl.Measure("dx", domain=domain, metadata=metadata)

    N     = ufl.FacetNormal(domain)
    Gendo = -p_endo * inner(v, J * ufl.transpose(ufl.inv(F)) * N) * ds(6)
    F     = inner(P, ufl.grad(v)) * dx + Gendo

    # -------------------------------------------------------------------------
    # solver
    # -------------------------------------------------------------------------

    problem     = NonlinearProblem(F, u, bcs)
    solver      = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8

    # -------------------------------------------------------------------------
    # step-wise loading + XDMF output
    # -------------------------------------------------------------------------

    load_steps  = 10
    target_load = -3.0
    loads       = np.linspace(0, target_load, load_steps)

    filename   = "out_ex03_ventricle_u"
    xdmf       = XDMFFile(domain.comm, f"{filename}.xdmf", "w")
    xdmf.write_mesh(domain)
    u_out      = fem.Function(V)
    u_out.name = "u"

    for step in range(load_steps):
        p_endo.value = loads[step]
        num_its, converged = solver.solve(u)
        assert converged
        u_out.interpolate(u)
        xdmf.write_function(u_out, step)

    xdmf.close()

    # -------------------------------------------------------------------------
    # extract solution at selected DOFs
    # -------------------------------------------------------------------------

    u.x.scatter_forward()

    save_results_xdmf(domain, {"displacement": u, "parameter": CC},
                      filename="out_ex03_novo.xdmf")

    u_values = u.x.array.reshape(-1, dim)
    u_mat    = np.zeros((Npoints, 3))
    for j, dof in enumerate(selected_dofs):
        u_mat[j, :] = u_values[dof, :]

    Fd = compute_Fh(domain, u, selected_dofs)

    print("end of synthetic data generation")

    return u_mat, Fd, CC


# =============================================================================
# entry point
# =============================================================================

if __name__ == "__main__":

    geodir = Path("lv_ellipsoid")
    geo    = cardiac_geometries.mesh.lv_ellipsoid(
        outdir           = geodir,
        create_fibers    = True,
        fiber_space      = "P_1",
        psize_ref        = 3,
        r_short_epi      = 10,
        aha              = True,
        fiber_angle_endo = 40.0,
        fiber_angle_epi  = -50.0,
    )

    u, Fd, c = prob_ventricle_passive_filling(geo)
