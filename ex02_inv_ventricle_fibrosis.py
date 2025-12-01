from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx import log
from dolfinx.io import XDMFFile
from ufl import Identity, det, inner, ln, exp, as_vector
import ufl
import numpy as np
import dolfinx
from pathlib import Path
import sys
from scipy.optimize import minimize
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.mesh import meshtags
from dolfinx.cpp.mesh import cell_num_entities

import cardiac_geometries
from ex01_ventricle import prob_ventricle_passive_filling 

# log.set_log_level(log.LogLevel.INFO)

# -----------------------------------------------------------------------------
# create geometry and mesh
# -----------------------------------------------------------------------------
geodir = Path("lv_ellipsoid") 
geo = cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir,
                                           create_fibers=True,
                                           fiber_space="DG_0",
                                           psize_ref=3,                                          
                                           r_short_epi=10,
                                           fiber_angle_endo=40.0,
                                           fiber_angle_epi=-50.0
                                           )
                                        #    aha=True)

# -----------------------------------------------------------------------------
# generate synthetic data
# -----------------------------------------------------------------------------
ud, cd = prob_ventricle_passive_filling(geo) 
domain = geo.mesh

# print("\n\n\n\n")

# noise_level = 1e-3  # adjust
# ud.x.array[:] += noise_level * np.random.randn(*ud.x.array.shape)

# -----------------------------------------------------------------------------
# function spaces
# -----------------------------------------------------------------------------
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
Va = dolfinx.fem.functionspace(domain, ("Lagrange", 1,))

du = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
uh = dolfinx.fem.Function(V)
vh = dolfinx.fem.Function(V)

# -----------------------------------------------------------------------------
# boundary conditions
# -----------------------------------------------------------------------------
facet_tags = geo.ffun
u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
base_dofs = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(5))
bcs = [fem.dirichletbc(u_bc, base_dofs, V)]

# -----------------------------------------------------------------------------
# kinematics
# -----------------------------------------------------------------------------
d = len(uh)
I = ufl.variable(ufl.Identity(d))
F = ufl.variable(I + ufl.grad(uh))
J = ufl.variable(ufl.det(F))
C = ufl.variable(F.T * F)

# fiber
f0 = geo.f0
s0 = geo.s0
n0 = geo.n0

# f0 = as_vector([1.0, 0.0, 0.0])
# s0 = as_vector([0.0, 1.0, 0.0])
# n0 = as_vector([0.0, 0.0, 1.0])

# xdmf = XDMFFile(domain.comm, "lv_ellipsoid/fiber_f0.xdmf", "w")
# xdmf.write_mesh(domain)
# xdmf.write_function(f0)
# xdmf.close()   
# xdmf = XDMFFile(domain.comm, "lv_ellipsoid/fiber_s0.xdmf", "w")
# xdmf.write_mesh(domain)
# xdmf.write_function(s0)
# xdmf.close()   
# xdmf = XDMFFile(domain.comm, "lv_ellipsoid/fiber_n0.xdmf", "w")
# xdmf.write_mesh(domain)
# xdmf.write_function(n0)
# xdmf.close()    

# initial guess
def cinit_expr(x):
    # return 4.0 - 3.8 * ((x[0] - 0.7)**2 + (x[1] - 0.5)**2 + (x[2] - 0.5)**2 < 0.3**2)
    # return 4.0 - 2.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2 < 0.2**2)
    # return 5.0 - 0.00*x[1]
    
    # OK
    return 2.0 - 0.000*x[1] 

# def cinit_expr(x):
#     # x has shape (3, N)
#     # Return random values in [2, 10] for each column
#     return 2.0 + 8.0 * np.random.rand(x.shape[1])

# def cinit_expr(x):
#     # x has shape (3, N)
#     xc,yc,zc = -5, -2, -9
#     sigma=1.0
#     dx = x[0] - xc
#     dy = x[1] - yc
#     dz = x[2] - zc
#     r2 = dx*dx + dy*dy + dz*dz

#     c_center = 10.0
#     c_far = 4.0
#     return c_far + (c_center - c_far) * np.exp(-r2 / sigma**2)

Va = fem.functionspace(domain, ("Lagrange", 1)) 
CC = fem.Function(Va)
CC.interpolate(cinit_expr)

# random field in [2,3]
CC.x.array[:] = 2.0 + 0.1*np.random.rand(len(CC.x.array))
CC.x.scatter_forward()

# Exportar archivos XDMF (igual que tu segundo script)
with XDMFFile(MPI.COMM_WORLD, "out_cc_init.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(CC)    

# constitutive parameters
# bf = default_scalar_type(8.0)
# bt = default_scalar_type(2.0)
# bfs = default_scalar_type(4.0)
bf = default_scalar_type(6.6)
bt = default_scalar_type(4.0)
bfs = default_scalar_type(2.6)
kappa = 1e2
kappa = fem.Constant(domain, kappa)

e1, e2, e3 = f0, s0, n0
Cs = J**(-2/3) * F.T * F
Es = 0.5 * (Cs - I)

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

#Energia
Wpassive = CC/2.0 * (ufl.exp(Q) - 1)
Wactive = 0.0
Wvolume = kappa * (J*ufl.ln(J) - J + 1)

strain_energy = Wpassive + Wactive + Wvolume
P = ufl.diff(strain_energy, F)

# external pressure (endocardial)
p_endo = fem.Constant(domain, 0.0)

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

N = ufl.FacetNormal(domain)
Gendo = -p_endo * ufl.inner(v, J * ufl.transpose(ufl.inv(F)) * N) * ds(6)
Fun = ufl.inner(P, ufl.grad(v)) * dx + Gendo

# -----------------------------------------------------------------------------
# functional for optimization 
# -----------------------------------------------------------------------------
alpha = dolfinx.fem.Constant(domain, 1.0e+1 )#-3 ) #+2 )

# Constant function = 1
one = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(1.0))
volume_form = dolfinx.fem.form(one * dx)
volume_mesh = dolfinx.fem.assemble_scalar(volume_form)

epsd = ufl.sym(ufl.grad(ud))
epsh = ufl.sym(ufl.grad(uh))

Fd = ufl.variable(I + ufl.grad(ud))
Fh = ufl.variable(I + ufl.grad(uh))


# Jfunctional = (1/2) * ufl.inner(uh - ud, uh - ud) * dx + (alpha/2) * ufl.inner(ufl.grad(CC), ufl.grad(CC)) * dx
# Jfunctional = (1/2) * ufl.inner(epsd-epsh, epsd-epsh) * dx + (alpha/2) * ufl.inner(ufl.grad(CC), ufl.grad(CC)) * dx
# Jfunctional = (1/2) * ufl.inner(epsd-epsh, epsd-epsh) * dx + (alpha/2) * ufl.inner(CC, CC) * dx
# Jfunctional = (1/2) * ufl.inner(epsd-epsh, epsd-epsh) * dx + (alpha/2) * ufl.inner(CC, CC) * dx

Jdata = (1/2) * ufl.inner(Fh-Fd, Fh-Fd) * dx
Jsmooth = (1.0/volume_mesh) * ufl.inner(ufl.grad(CC), ufl.grad(CC)) * dx

Jfunctional = Jdata + alpha*Jsmooth


# forward problem function
def solve_nl_prob(uh):
    forward_problem = NonlinearProblem(Fun, uh, bcs)
    solver = NewtonSolver(domain.comm, forward_problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    
    ksp = solver.krylov_solver
    ksp.setType("preonly")        # do not use CG/GMRES/etc
    ksp.getPC().setType("lu")     # use LU factorization

    # step-wise loading
    load_steps = 10
    target_load = -3.0
    loads = np.linspace(0, target_load, load_steps)

    for step in range(load_steps):
        p_endo.value = loads[step]
        num_its, converged = solver.solve(uh)
        assert(converged)
        print(f" load step {step} - newton its {num_its}")   
        # uh.x.scatter_forward()
    print(" ")

# derivatives for the adjoint gradient computation
lmbda = dolfinx.fem.Function(V)
dFdu = ufl.derivative(Fun, uh, du)
dFdu_adj = ufl.adjoint(dFdu)
dJdu = ufl.derivative(Jfunctional, uh, v)

# the adjoint problem
adj_problem = LinearProblem(dFdu_adj, -dJdu, bcs=bcs)
lmbda = adj_problem.solve()

# derivatives wrt to parameter CC
W = Va
q = ufl.TrialFunction(W)
dJdf = ufl.derivative(Jfunctional, CC, q)
dFdf = ufl.action(ufl.adjoint(ufl.derivative(Fun, CC, q)), lmbda)
dJdf_compiled = dolfinx.fem.form(dJdf)
dFdf_compiled = dolfinx.fem.form(dFdf)
dLdf = dolfinx.fem.Function(W)

Jh = dolfinx.fem.form(Jfunctional)

vals_func = []
vals_data = []
vals_smooth = []
Jits = [0,]

# function to evaluate functional
def eval_J(x):
    CC.x.array[:] = x
    print("solving forward problem (eval_functional)")
    solve_nl_prob(uh)
    local_J = dolfinx.fem.assemble_scalar(Jh)
    return domain.comm.allreduce(local_J, op=MPI.SUM)

# function to evaluate gradient of functional
def eval_gradient(x):
    CC.x.array[:] = x
    print("solving forward problem (eval_gradient)")
    solve_nl_prob(uh)
    print("solving adjoint problem (eval_gradient)")
    lmbda = adj_problem.solve()
    print("")
    dLdf.x.array[:] = 0
    dolfinx.fem.assemble_vector(dLdf.x.array, dJdf_compiled)
    dolfinx.fem.assemble_vector(dLdf.x.array, dFdf_compiled)
    return dLdf.x.array

# callback function
def callback(intermediate_result):
    # print(intermediate_result)
    fval = intermediate_result.fun
    vals_func.append(fval)
    Jits[0] = Jits[0] + 1
    print(f"optimization iteration {Jits[0]}")
    print(f"value of functional J: {fval}\n")
    # print(intermediate_result.x)

print("\nbegin optimization\n")

ne = int(np.shape(CC.x.array)[0])
cbounds = [(1.0, 6.0)]*ne

opt_sol = minimize(
    eval_J,
    CC.x.array,
    jac=eval_gradient,
    # method="CG",
    method="L-BFGS-B",
    # method="SLSQP",
    tol=1e-12,
    callback=callback,
    bounds=cbounds,
    # options={"disp": True},
)

print("optimization results:")
print(opt_sol)

# save final results
CC.x.array[:] = opt_sol.x

# evaluate forward problem
solve_nl_prob(uh)

# -----------------------------------------------------------------------------
# final computations for output
# -----------------------------------------------------------------------------
print("summary of results:\n")

# error
error = fem.Function(Va)
error.x.array[:] = np.abs(CC.x.array[:] - cd.x.array[:])
abs_err = np.abs(error.x.array)
i_max = np.argmax(abs_err)
den = abs(cd.x.array[i_max])
num = abs_err[i_max]
rel_error_max = num / den

# compute functional terms separately (data + smooth)
Jdata_form = dolfinx.fem.form(Jdata)
Jdata_value = dolfinx.fem.assemble_scalar(Jdata_form)
Jsmooth_form = dolfinx.fem.form(Jsmooth)
Jsmooth_value = dolfinx.fem.assemble_scalar(Jsmooth_form)
Jtotal_form = dolfinx.fem.form(Jfunctional)
Jtotal_value = dolfinx.fem.assemble_scalar(Jtotal_form)

print(f"alpha (reg.)   : {float(alpha):.8e}")
print(f"max rel error  : {rel_error_max:.8e}")
print(f"J data value   : {Jdata_value:.8e}")
print(f"J smooth value : {Jsmooth_value:.8e}")
print(f"J total value  : {Jtotal_value:.8e}")

np.savetxt('functional_history.txt', vals_func)

# Exportar archivos XDMF (igual que tu segundo script)
with XDMFFile(MPI.COMM_WORLD, "out_cc_estimated.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(CC)

with XDMFFile(MPI.COMM_WORLD, "out_cd_true.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(cd)

with XDMFFile(MPI.COMM_WORLD, "out_uh_estimated.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

with XDMFFile(MPI.COMM_WORLD, "out_ud_true.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(ud)

with XDMFFile(MPI.COMM_WORLD, "out_lambda.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(lmbda)

with XDMFFile(MPI.COMM_WORLD, "out_cc_error.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(error)

print("\nend optimization")
