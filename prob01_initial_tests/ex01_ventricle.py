from dolfinx import fem, mesh
from dolfinx import log, default_scalar_type
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from ufl import Identity, det, inner, ln, exp, as_vector
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
import sys

import cardiac_geometries
from pathlib import Path
from aha import focal, get_aha_segments

# log.set_log_level(log.LogLevel.INFO) # WARNING


def prob_ventricle_passive_filling(geo):

    # -----------------------------------------------------------------------------
    # Handle mesh
    # -----------------------------------------------------------------------------

    print(f"\nsolving forward problem to generate synthetic data")

    domain = geo.mesh
    facet_tags = geo.ffun
    markers = geo.markers

    # print(geo)
    # print(domain)
    # print(markers)

    xdmf = XDMFFile(domain.comm, "lv_ellipsoid/fiber.xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(geo.f0)
    xdmf.close()    

    # -----------------------------------------------------------------------------
    # Function spaces
    # -----------------------------------------------------------------------------

    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

    v = ufl.TestFunction(V)
    u = fem.Function(V)

    # -----------------------------------------------------------------------------
    # Boundary conditions
    # -----------------------------------------------------------------------------

    u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
    base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))

    bcs = [fem.dirichletbc(u_bc, base_dofs, V)]

    # -----------------------------------------------------------------------------
    # Kinematics
    # -----------------------------------------------------------------------------

    # Spatial dimension
    d = len(u)

    # Identity tensor
    I = ufl.variable(ufl.Identity(d))

    # Deformation gradient
    F = ufl.variable(I + ufl.grad(u))
    J = ufl.variable(ufl.det(F))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)
    E = 0.5*(C - I)             # the Green-Lagrange strain tensor

    # Tissue microstructure (needed by the GuccioneMaterial class)
    f0 = geo.f0
    s0 = geo.s0
    n0 = geo.n0

    # f0 = as_vector([1.0, 0.0, 0.0])
    # s0 = as_vector([0.0, 1.0, 0.0])
    # n0 = as_vector([0.0, 0.0, 1.0])

    # -------------------------------------------------------
    # CC distribution -  Compute AHA regions
    # -------------------------------------------------------

    foc = focal(r_long_endo=geo.info['r_long_endo'], 
                r_short_endo=geo.info['r_short_endo'])
    mu_base = geo.info['mu_base_endo']
    mu_base = abs(mu_base) 
    dmu = (np.pi - mu_base) / 4.0

    V0 = fem.functionspace(domain, ("DG", 0)) 
    cval = fem.Function(V0)
    values = cval.x.array

    segments = get_aha_segments(domain, foc, mu_base, dmu)
    for i in range(len(segments)):
        if segments[i] == 12:
            values[i] = 4.0
        else:
            values[i] = 2.0

    cval.x.array[:] = values
    cval.x.scatter_forward()

    Va = fem.functionspace(domain, ("Lagrange", 1))
    CC = fem.Function(Va)
    CC.interpolate(cval)

    # -------------------------------------------------------
    # CC distribution - Smooth gaussian
    # -------------------------------------------------------

    # def c_expr(x):
    #     # x has shape (3, N)
    #     xc,yc,zc = -5, -2, -9
    #     sigma=8.0
    #     dx = x[0] - xc
    #     dy = x[1] - yc
    #     dz = x[2] - zc
    #     r2 = dx*dx + dy*dy + dz*dz
    #     r = np.sqrt(dx*dx + dy*dy + dz*dz)
    #     c_center = 8.0
    #     c_far = 2.0
    #     return c_far + (c_center - c_far) * np.exp(-r2 / sigma**2)
    #     # return c_far + (c_center - c_far) * np.exp(-r / sigma)

    # def c_expr(x):
    #     xc, yc, zc = -5, -2, -9
    #     dx = x[0] - xc
    #     dy = x[1] - yc
    #     dz = x[2] - zc
    #     r = np.sqrt(dx*dx + dy*dy + dz*dz)

    #     c_center = 3.0
    #     c_far = 2.0

    #     r0 = 5.0   # plateau radius
    #     r1 = 10.0 #15.0  # transition outer radius

    #     # smoothstep
    #     def smoothstep(t):
    #         return 3*t*t - 2*t*t*t

    #     if isinstance(r, np.ndarray):
    #         C = np.zeros_like(r)

    #         inside = r <= r0
    #         outside = r >= r1
    #         middle = (~inside) & (~outside)

    #         C[inside] = c_center
    #         C[outside] = c_far

    #         t = (r1 - r[middle]) / (r1 - r0)
    #         C[middle] = c_far + (c_center - c_far) * smoothstep(t)
    #         return C
    #     else:
    #         if r <= r0:
    #             return c_center
    #         elif r >= r1:
    #             return c_far
    #         else:
    #             t = (r1 - r) / (r1 - r0)
    #             return c_far + (c_center - c_far) * smoothstep(t)
            
    # initial guess
    # def c_expr(x):
    #     return 1.1 - 0.0*x[1] 

    # Va = fem.functionspace(domain, ("Lagrange", 1)) 
    # CC = fem.Function(Va)
    # CC.interpolate(c_expr)

    # -------------------------------------------------------

    # CC = default_scalar_type(2.0)
    bf = default_scalar_type(6.6)
    bt = default_scalar_type(4.0)
    bfs = default_scalar_type(2.6)
    # bf = default_scalar_type(8.0)
    # bt = default_scalar_type(2.0)
    # bfs = default_scalar_type(4.0)
    kappa = 1e2
    kappa = fem.Constant(domain,kappa)

    e1, e2, e3 = f0, s0, n0

    Cs = J**(-2/3) * F.T * F   # Tensor derecho de Cauchy-Green
    Es = 0.5 * (Cs - I)        # Tensor de deformacion de Green-Lagrange

    E11 = ufl.inner(Es*e1, e1)
    E12 = ufl.inner(Es*e1, e2)
    E13 = ufl.inner(Es*e1, e3)
    E21 = ufl.inner(Es*e2, e1)
    E22 = ufl.inner(Es*e2, e2)
    E23 = ufl.inner(Es*e2, e3)
    E31 = ufl.inner(Es*e3, e1)
    E32 = ufl.inner(Es*e3, e2)
    E33 = ufl.inner(Es*e3, e3)

    Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) + bfs*(E12**2 + E21**2 + E13**2 + E31**2)

    # Energia pasiva
    Wpassive = CC/2.0 * (exp(Q) - 1)
    Wactive  = 0.0
    Wvolume  = kappa * (J*ln(J) - J + 1)

    strain_energy = Wpassive + Wactive + Wvolume
    P = ufl.diff(strain_energy, F)

    #for this example we keep the pressure at zero, but keep it here for flexibility
    p_endo = fem.Constant(domain,0.0) 

    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)

    # Definition of the weak form:
    N = ufl.FacetNormal(domain)
    Gendo = -p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N) * ds(6) # Endo
    F = inner(P, ufl.grad(v)) * dx + Gendo 

    # Solver
    problem = NonlinearProblem(F, u, bcs)
    solver = NewtonSolver(domain.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    # solver.convergence_criterion = "incremental"

    # Step-wise loading (for plotting and convergence)
    load_steps = 10
    target_load = -3.0
    loads = np.linspace(0, target_load, load_steps)

    # output
    filename = "out_ex01_ventricle_u"
    xdmf = XDMFFile(domain.comm, f"{filename}.xdmf", "w")
    xdmf.write_mesh(domain) 
    u_out = fem.Function(V)
    u_out.name = "u"
    for step in range(load_steps):
        p_endo.value = loads[step]
        num_its, converged = solver.solve(u)
        assert(converged)    
        u_out.interpolate(u)
        xdmf.write_function(u_out, step)
    xdmf.close()    

    filename = "out_ex01_ventricle_cc"
    xdmf = XDMFFile(domain.comm, f"{filename}.xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(CC)
    xdmf.close()    

    print("end of synthetic data generation")

    return u, CC

if __name__ == "__main__":

    geodir = Path("lv_ellipsoid")
    geo = cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, 
                                            create_fibers=True, 
                                            fiber_space="P_1", 
                                            psize_ref=3,
                                            r_short_epi=10, 
                                            aha=True,
                                            fiber_angle_endo=40.0,
                                            fiber_angle_epi=-50.0)

    u, c = prob_ventricle_passive_filling(geo)

    