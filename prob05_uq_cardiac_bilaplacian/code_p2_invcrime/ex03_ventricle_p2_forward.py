"""
ex03_ventricle_p2_forward.py
----------------------------
Synthetic-data generator on a P2 discretization (to mitigate the inverse crime).

The inversion (ex06) uses P1 displacement. Here the SAME passive-filling
problem is solved with P2 displacement, and the true stiffness field is also
represented in P2 (interpolated from the analytic expression), following the
idea of Bui-Thanh et al. (2013): the data are generated with a higher-order
model than the one used for the inversion, so the inversion cannot reproduce
them exactly even at the true parameter.

Same geometry, fibres, constitutive law, boundary conditions and load as ex03.

Main entry point:
    u2, C1, C2 = solve_forward_p2(geo, case_type, target_load, kappa_bulk)

    u2 : dolfinx Function, P2 vector  -- "true" displacement
    C1 : dolfinx Function, P1 scalar  -- true C (kPa) on the inversion space
    C2 : dolfinx Function, P2 scalar  -- true C (kPa) used to generate the data

ex06 then selects the observation nodes, interpolates u2 -> P1 and evaluates
F = I + grad(u) in each sensor cell (or the cell average of F from u2).

Standalone:
    python ex03_ventricle_p2_forward.py --case fibrosis
writes out_ex03_p2_<case>.xdmf with the P2 solution interpolated to P1.
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver


# =============================================================================
# true stiffness field C(x) [kPa]  (same definitions as ex03)
# =============================================================================

def c_true_expr(case_type, domain):
    """Vectorized analytic C(x) for 'fibrosis' or 'linear' (kPa)."""
    if case_type == "fibrosis":
        xc, yc, zc = -5.0, -2.0, -9.0
        c_center, c_far = 6.0, 2.0
        r0, r1 = 5.0, 10.0

        def c_expr(x):
            r = np.sqrt((x[0] - xc) ** 2 + (x[1] - yc) ** 2 + (x[2] - zc) ** 2)
            t = np.clip((r1 - r) / (r1 - r0), 0.0, 1.0)      # 1 inside r0, 0 beyond r1
            return c_far + (c_center - c_far) * (3 * t * t - 2 * t * t * t)
        return c_expr

    if case_type == "linear":
        # use the MESH extents (not the interpolation points) so that the P1
        # and P2 interpolants represent the same function
        xmin = float(domain.geometry.x[:, 0].min())
        xmax = float(domain.geometry.x[:, 0].max())

        def c_expr(x):
            return 2.0 + (x[0] - xmin) / (xmax - xmin)
        return c_expr

    raise ValueError(f"Unknown case_type {case_type!r}; use 'fibrosis' or 'linear'.")


# =============================================================================
# P2 forward solve
# =============================================================================

def solve_forward_p2(geo, case_type="fibrosis", target_load=-10.0, kappa_bulk=1e2,
                     load_steps=20, quad_degree=4, newton_maxit=50, verbose=True):
    """
    Solve the passive-filling problem with P2 displacement and P2 true C.

    The load ramp uses `load_steps` nonzero steps up to `target_load`
    (the ramp in the original ex03 included a wasted zero-load step).
    """
    domain = geo.mesh
    dim = domain.geometry.dim
    facet_tags = geo.ffun

    V2 = fem.functionspace(domain, ("Lagrange", 2, (dim,)))
    u2 = fem.Function(V2, name="u_true_p2")
    v = ufl.TestFunction(V2)

    base_dofs = fem.locate_dofs_topological(V2, facet_tags.dim, facet_tags.find(5))
    bcs = [fem.dirichletbc(np.zeros(dim, dtype=default_scalar_type), base_dofs, V2)]

    # true C on the inversion space (P1) and on the data space (P2)
    c_expr = c_true_expr(case_type, domain)
    C1 = fem.Function(fem.functionspace(domain, ("Lagrange", 1)), name="C_true")
    C1.interpolate(c_expr)
    C2 = fem.Function(fem.functionspace(domain, ("Lagrange", 2)), name="C_true_p2")
    C2.interpolate(c_expr)

    # kinematics + Holzapfel-Ogden passive law (identical to ex03, C in kPa)
    I = ufl.variable(ufl.Identity(dim))
    F = ufl.variable(I + ufl.grad(u2))
    J = ufl.variable(ufl.det(F))
    e1, e2, e3 = geo.f0, geo.s0, geo.n0
    bf, bt, bfs = default_scalar_type(6.6), default_scalar_type(4.0), default_scalar_type(2.6)
    kappa = fem.Constant(domain, float(kappa_bulk))

    Es = 0.5 * (J ** (-2 / 3) * F.T * F - I)
    E = [[ufl.inner(Es * a, b) for b in (e1, e2, e3)] for a in (e1, e2, e3)]
    Q = (bf * E[0][0] ** 2
         + bt * (E[1][1] ** 2 + E[2][2] ** 2 + E[1][2] ** 2 + E[2][1] ** 2)
         + bfs * (E[0][1] ** 2 + E[1][0] ** 2 + E[0][2] ** 2 + E[2][0] ** 2))
    W = C2 / 2.0 * (ufl.exp(Q) - 1) + kappa * (J * ufl.ln(J) - J + 1)
    P = ufl.diff(W, F)

    p_endo = fem.Constant(domain, 0.0)
    md = {"quadrature_degree": quad_degree}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, metadata=md)
    dx = ufl.Measure("dx", domain=domain, metadata=md)
    N = ufl.FacetNormal(domain)
    R = (ufl.inner(P, ufl.grad(v)) * dx
         - p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N) * ds(6))

    problem = NonlinearProblem(R, u2, bcs)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-10
    solver.rtol = 1e-10
    solver.max_it = newton_maxit
    ksp = solver.krylov_solver
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    if verbose:
        print(f"\n[P2 data] solving passive filling on P2 ({V2.dofmap.index_map.size_global * dim} "
              f"dofs), case={case_type}, load={target_load}, steps={load_steps}")
    for step, load in enumerate(np.linspace(0.0, target_load, load_steps + 1)[1:]):
        p_endo.value = load
        its, converged = solver.solve(u2)
        if not converged:
            raise RuntimeError(f"P2 forward Newton failed at load step {step}")
        if verbose:
            print(f"  [P2 data] load step {step}: load={load:.3f}, newton its {its}")
    u2.x.scatter_forward()
    return u2, C1, C2


# =============================================================================
# standalone run
# =============================================================================

if __name__ == "__main__":
    import cardiac_geometries

    ap = argparse.ArgumentParser(description="P2 synthetic data for the LV problem")
    ap.add_argument("--case", choices=["fibrosis", "linear"], default="fibrosis")
    ap.add_argument("--load", type=float, default=-10.0)
    ap.add_argument("--kappa", type=float, default=1e2)
    ap.add_argument("--load-steps", type=int, default=10)
    a = ap.parse_args()

    geodir = Path("lv_ellipsoid")
    try:
        geo = cardiac_geometries.geometry.Geometry.from_folder(geodir)
    except Exception:
        geo = cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir, create_fibers=True, fiber_space="DG_0",
            psize_ref=3, r_short_epi=10,
            fiber_angle_endo=40.0, fiber_angle_epi=-50.0)

    u2, C1, C2 = solve_forward_p2(geo, a.case, a.load, a.kappa, a.load_steps)

    # XDMF needs the mesh degree (P1): write the nodal interpolant
    dim = geo.mesh.geometry.dim
    u1 = fem.Function(fem.functionspace(geo.mesh, ("Lagrange", 1, (dim,))),
                      name="u_true_p2_at_vertices")
    u1.interpolate(u2)
    with XDMFFile(MPI.COMM_WORLD, f"out_ex03_p2_{a.case}.xdmf", "w") as xf:
        xf.write_mesh(geo.mesh)
        xf.write_function(u1)
        xf.write_function(C1)
    print(f"Wrote out_ex03_p2_{a.case}.xdmf")
