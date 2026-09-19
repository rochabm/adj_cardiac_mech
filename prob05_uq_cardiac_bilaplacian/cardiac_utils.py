"""
cardiac_utils.py
----------------
Shared utilities for cardiac mechanics simulations.

Used by:
    ex03_ventricle_discrete.py  — forward problem
    ex04_ventricle_inverse_discrete_var.py  — inverse problem
"""

import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile


# =============================================================================
# mesh / node selection
# =============================================================================

def select_distributed_nodes(dof_coords, N, min_dist=None, seed=None):
    """
    Selects N nodes distributed as uniformly as possible over the domain
    using FARTHEST POINT SAMPLING (greedy algorithm).

    This gives much more uniform coverage than random Poisson disk sampling,
    especially on curved surfaces like the ventricle where Euclidean distance
    can be misleading due to mesh curvature and non-uniform mesh density.

    Algorithm:
        1. Pick a random starting node.
        2. At each step, pick the node that is FARTHEST from all already
           selected nodes (maximizes the minimum distance to selected set).
        3. Repeat N times.

    This guarantees that selected nodes are spread as uniformly as possible
    over the point cloud, independent of mesh density variations.

    Parameters
    ----------
    dof_coords : np.ndarray, shape (M, 3)
    N          : int   — number of nodes to select
    min_dist   : float — not used (kept for API compatibility)
    seed       : int   — seed for reproducibility (affects starting node only)

    Returns
    -------
    selected_dofs   : np.ndarray, shape (N,)
    selected_coords : np.ndarray, shape (N, 3)
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    M   = len(dof_coords)
    N   = min(N, M)

    # distance from each candidate to the nearest already-selected node
    # initialised to infinity (no selected nodes yet)
    min_dist_to_selected = np.full(M, np.inf)

    # random starting node
    first = int(rng.integers(0, M))
    selected = [first]

    # update distances after adding first node
    d = np.linalg.norm(dof_coords - dof_coords[first], axis=1)
    min_dist_to_selected = np.minimum(min_dist_to_selected, d)

    for _ in range(N - 1):
        # pick the node farthest from all selected nodes
        next_idx = int(np.argmax(min_dist_to_selected))
        selected.append(next_idx)

        # update minimum distances
        d = np.linalg.norm(dof_coords - dof_coords[next_idx], axis=1)
        min_dist_to_selected = np.minimum(min_dist_to_selected, d)

    selected_dofs   = np.array(selected)
    selected_coords = dof_coords[selected_dofs]

    # report achieved minimum distance for diagnostics
    tree = cKDTree(selected_coords)
    dists, _ = tree.query(selected_coords, k=2)
    achieved_min_dist = dists[:, 1].min()
    print(f"[select_distributed_nodes] farthest-point sampling: "
          f"{len(selected_dofs)} nodes, "
          f"min inter-node dist = {achieved_min_dist:.4f}")

    return selected_dofs, selected_coords


# =============================================================================
# deformation gradient
# =============================================================================

def compute_Fh(mesh, uu, selected_dofs):
    """
    Calcula F = I + grad(u) para cada nodo em selected_dofs.

    Para cada nodo, localiza um elemento que o contém e usa
    todos os nodos desse elemento (com u extraído de uh completo)
    para calcular grad(u) via mínimos quadrados local.

    Parâmetros
    ----------
    mesh          : dolfinx.mesh.Mesh
    uu            : dolfinx.fem.Function  — solução completa
    selected_dofs : np.ndarray, shape (Npoints,)

    Retorna
    -------
    Fh : np.ndarray, shape (Npoints, 3, 3)
    """
    dim    = mesh.geometry.dim
    I      = np.eye(dim)
    coords = mesh.geometry.x
    u_all  = uu.x.array.reshape(-1, dim)
    dofmap = mesh.geometry.dofmap

    # índice inverso: nodo → elementos que o contêm
    Ncells = mesh.topology.index_map(mesh.topology.dim).size_local
    node_to_cells = {}
    for cell_id in range(Ncells):
        for node in dofmap[cell_id]:
            node_to_cells.setdefault(int(node), []).append(cell_id)

    Npoints = len(selected_dofs)
    Fh = np.zeros((Npoints, dim, dim))

    for j, dof in enumerate(selected_dofs):
        cell_id  = node_to_cells[int(dof)][0]
        node_ids = dofmap[cell_id]

        X_local = coords[node_ids, :dim]
        U_local = u_all[node_ids, :dim]

        dX = X_local[1:] - X_local[0]
        dU = U_local[1:] - U_local[0]

        grad_u = np.zeros((dim, dim))
        for i in range(dim):
            g, _, _, _ = np.linalg.lstsq(dX, dU[:, i], rcond=None)
            grad_u[i, :] = g

        Fh[j] = I + grad_u

    return Fh


# =============================================================================
# error statistics
# =============================================================================

def print_F_error_statistics(Fh, Fd):
    """
    Prints global and per-component error statistics
    between estimated and reference deformation gradients.

    Parameters
    ----------
    Fh : np.ndarray, shape (Npoints, 3, 3) — estimated F
    Fd : np.ndarray, shape (Npoints, 3, 3) — reference F
    """
    erro   = Fh - Fd
    labels = [["F11", "F12", "F13"],
              ["F21", "F22", "F23"],
              ["F31", "F32", "F33"]]

    abs_err = np.abs(erro)
    rel_err = abs_err / (np.abs(Fd) + 1e-12)

    print()
    print("=== global error statistics ===")
    print(f"  max  abs error : {abs_err.max():.4e}")
    print(f"  mean abs error : {abs_err.mean():.4e}")
    print(f"  max  rel error : {rel_err.max():.4e}")
    print(f"  mean rel error : {rel_err.mean():.4e}")
    print(f"  Frobenius norm : {np.linalg.norm(erro):.4e}")

    print()
    print("=== per-component error statistics ===")
    print(f"  {'comp':<6} {'max abs':>10} {'mean abs':>10} {'max rel':>10} {'mean rel':>10}")
    print(f"  {'-'*46}")

    for i in range(3):
        for j in range(3):
            comp  = erro[:, i, j]
            ref   = Fd[:, i, j]
            abs_c = np.abs(comp)
            rel_c = abs_c / (np.abs(ref) + 1e-12)
            print(f"  {labels[i][j]:<6} {abs_c.max():>10.4e} {abs_c.mean():>10.4e} {rel_c.max():>10.4e} {rel_c.mean():>10.4e}")


# =============================================================================
# I/O
# =============================================================================

def save_results_xdmf(domain, fields: dict, filename="out_results.xdmf"):
    """
    Salva múltiplos campos num único arquivo XDMF compatível com ParaView.

    Parâmetros
    ----------
    domain   : dolfinx.mesh.Mesh
    fields   : dict — {"nome_campo": Function, ...}
               O nome vira o label visível no ParaView.
    filename : str  — caminho do arquivo de saída
    """
    with XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        for name, func in fields.items():
            func.name = name
            xdmf.write_function(func)


def save_selected_nodes(selected_coords, selected_dofs):
    """Exporta as coordenadas dos nodos selecionados em CSV, VTK e XYZ."""

    # 1. CSV
    np.savetxt(
        "dof_coords.csv",
        selected_coords,
        delimiter=",",
        header="x,y,z",
        comments=""
    )
    print("Saved: dof_coords.csv")

    # 2. VTK (legacy ASCII)
    n = len(selected_coords)
    with open("dof_coords.vtk", "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("DOF Coordinates\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for x, y, z in selected_coords:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        f.write(f"\nVERTICES {n} {2 * n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"\nPOINT_DATA {n}\n")
        f.write("SCALARS dof_index int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for idx in selected_dofs:
            f.write(f"{idx}\n")
    print("Saved: dof_coords.vtk")

    # 3. XYZ
    with open("dof_coords.xyz", "w") as f:
        f.write(f"{n}\n")
        f.write("DOF coordinates exported\n")
        for (x, y, z), idx in zip(selected_coords, selected_dofs):
            f.write(f"X {x:.6f} {y:.6f} {z:.6f}  # dof={idx}\n")
    print("Saved: dof_coords.xyz")
