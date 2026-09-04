"""
plot_ex04_uq_results.py
-------------------------
PyVista visualization of ex04's UQ results, covering three analyses:

  1. Interactive field viewer for all fields in out_ex04_novo.h5
     (displacement, c_init, c_estimated, c_true, c_error, lambda,
     posterior_variance, posterior_stddev), with rescale-on-switch.

  2. Posterior std-dev map overlaid with the 32 measurement points
     (read from dof_coords.csv, written by save_selected_nodes),
     to visually check whether uncertainty is higher far from
     measurement locations.

  3. Scatter plot + per-dof correlation between c_error (|C_est -
     C_true|) and posterior_stddev, to check whether the Laplace
     approximation's uncertainty estimate tracks the actual
     estimation error.

Usage:
    python plot_ex04_uq_results.py
"""

import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

H5_FILE          = "out_ex04_novo.h5"      # adjust name if MPI ranks > 1 changed it
MEASURED_POINTS_CSV = "dof_coords.csv"     # written by save_selected_nodes()


# =============================================================================
# shared: read mesh + fields from the dolfinx-written H5 file
# =============================================================================

def build_grid_from_h5(h5_file: str):
    fields = {}

    with h5py.File(h5_file, "r") as f:
        print("=== file structure ===")

        def print_tree(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}  {obj.shape}  {obj.dtype}")

        f.visititems(print_tree)

        # mesh group name can vary ("Mesh/mesh" vs "Mesh/Mesh") -- detect dynamically
        mesh_group_name = list(f["Mesh"].keys())[0]
        points = f[f"Mesh/{mesh_group_name}/geometry"][:]
        cells = f[f"Mesh/{mesh_group_name}/topology"][:]

        for field_name in f["Function"].keys():
            steps = list(f[f"Function/{field_name}"].keys())
            last = steps[-1]
            data = f[f"Function/{field_name}/{last}"][:]
            fields[field_name] = data
            print(f"  campo '{field_name}' -- steps={steps} -- shape={data.shape}")

    print(f"\npoints : {points.shape}")
    print(f"cells  : {cells.shape}")

    nodes_per_cell = cells.shape[1]
    # tetra (4 nodes) for 3D LV mesh
    vtk_type_map = {3: 5, 4: 10}  # 3 -> triangle (2D), 4 -> tetra (3D)
    vtk_type = vtk_type_map.get(nodes_per_cell)
    if vtk_type is None:
        raise ValueError(f"Unexpected nodes_per_cell={nodes_per_cell}")

    cell_array = np.hstack([
        np.full((cells.shape[0], 1), nodes_per_cell, dtype=np.int64),
        cells,
    ]).ravel()
    celltypes = np.full(cells.shape[0], vtk_type, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_array, celltypes, points)

    for name, data in fields.items():
        grid.point_data[name] = data

    return grid, list(fields.keys())


def load_measured_points(csv_file: str) -> np.ndarray:
    """Reads the x,y,z columns written by save_selected_nodes()."""
    return np.loadtxt(csv_file, delimiter=",", skiprows=1)


# =============================================================================
# ITEM 1: interactive field viewer with rescale-on-switch
# =============================================================================

def item1_field_viewer(grid, field_names):
    print("\n[Item 1] Interactive field viewer")
    print("Available fields:", field_names)

    # build a vector-magnitude version of displacement/lambda if they
    # are vector fields (shape (N, 3)), useful as default scalar
    scalar_fields = []
    for name in field_names:
        data = grid.point_data[name]
        if data.ndim == 2 and data.shape[1] == 3:
            mag_name = f"{name}_mag"
            grid.point_data[mag_name] = np.linalg.norm(data, axis=1)
            scalar_fields.append(mag_name)
        else:
            scalar_fields.append(name)

    p = pv.Plotter(title="ex04 UQ results -- field viewer")
    actor = p.add_mesh(grid, scalars=scalar_fields[0], cmap="viridis", show_edges=False)
    p.add_scalar_bar(scalar_fields[0])

    def make_callback(name):
        def callback(state):
            if state:
                data = grid.point_data[name]
                vmin, vmax = float(data.min()), float(data.max())
                grid.set_active_scalars(name)
                actor.mapper.scalar_range = (vmin, vmax)
                p.render()
        return callback

    for i, name in enumerate(scalar_fields):
        p.add_checkbox_button_widget(
            make_callback(name), position=(10, 10 + i * 35), size=25, color_on="steelblue"
        )
        p.add_text(name, position=(45, 12 + i * 35), font_size=9)

    p.show()


# =============================================================================
# ITEM 2: posterior std-dev map + measurement points overlay
# =============================================================================

def item2_uncertainty_with_points(grid, measured_points):
    print("\n[Item 2] Posterior std-dev + measurement points overlay")

    if "posterior_stddev" not in grid.point_data:
        print("  posterior_stddev not found in file -- skipping item 2.")
        return

    p = pv.Plotter(title="Posterior std-dev + measurement points")

    data = grid.point_data["posterior_stddev"]
    p.add_mesh(grid, scalars="posterior_stddev", cmap="inferno",
               clim=(float(data.min()), float(data.max())),
               opacity=0.85, show_edges=False)
    p.add_scalar_bar("posterior_stddev (sqrt of variance)")

    # overlay measurement points as spheres
    points_poly = pv.PolyData(measured_points)
    p.add_mesh(points_poly, color="cyan", point_size=14,
               render_points_as_spheres=True, label="measurement points")

    p.add_legend()
    p.view_isometric()
    p.show()

    # -----------------------------------------------------------------
    # quantitative check: stddev vs distance to nearest measurement point
    # -----------------------------------------------------------------
    print("\n  Quantitative check: stddev vs. distance to nearest sensor")

    mesh_points = grid.points
    # cap to avoid an O(N*M) blowup on very fine meshes
    n_check = min(len(mesh_points), 5000)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(mesh_points), size=n_check, replace=False)

    dists = np.zeros(n_check)
    for k, i in enumerate(idx):
        d = np.linalg.norm(measured_points - mesh_points[i], axis=1)
        dists[k] = d.min()

    stddev_sample = data[idx]

    corr = np.corrcoef(dists, stddev_sample)[0, 1]
    print(f"  Pearson correlation (distance to nearest sensor vs stddev): {corr:.4f}")
    print("  (positive correlation supports the hypothesis: farther from")
    print("   sensors -> higher posterior uncertainty)")

    plt.figure(figsize=(6, 5))
    plt.scatter(dists, stddev_sample, s=8, alpha=0.4)
    plt.xlabel("distance to nearest measurement point")
    plt.ylabel("posterior stddev")
    plt.title(f"stddev vs. distance to sensor (r = {corr:.3f})")
    plt.tight_layout()
    plt.savefig("item2_stddev_vs_distance.png", dpi=150)
    print("  Saved plot: item2_stddev_vs_distance.png")
    plt.show()


# =============================================================================
# ITEM 3: c_error vs posterior_stddev correlation
# =============================================================================

def item3_error_vs_uncertainty(grid):
    print("\n[Item 3] c_error vs posterior_stddev correlation")

    if "c_error" not in grid.point_data or "posterior_stddev" not in grid.point_data:
        print("  c_error or posterior_stddev not found -- skipping item 3.")
        return

    c_error = grid.point_data["c_error"]
    stddev = grid.point_data["posterior_stddev"]

    # NOTE: c_error in ex04 is normalized (divided by den, the max
    # abs error value) -- still monotonically related to the raw
    # |C_est - C_true|, so the correlation analysis below is valid
    # in relative terms.

    corr = np.corrcoef(c_error, stddev)[0, 1]
    print(f"  Pearson correlation (c_error vs posterior_stddev): {corr:.4f}")
    print("  (positive correlation supports: high-uncertainty regions")
    print("   coincide with high actual estimation error)")

    # spatial side-by-side comparison
    p = pv.Plotter(shape=(1, 2), title="c_error vs posterior_stddev")

    p.subplot(0, 0)
    p.add_mesh(grid, scalars="c_error", cmap="inferno",
               clim=(float(c_error.min()), float(c_error.max())), show_edges=False)
    p.add_scalar_bar("c_error (normalized)")
    p.add_text("c_error", font_size=10)
    p.view_isometric()

    p.subplot(0, 1)
    p.add_mesh(grid, scalars="posterior_stddev", cmap="inferno",
               clim=(float(stddev.min()), float(stddev.max())), show_edges=False)
    p.add_scalar_bar("posterior_stddev")
    p.add_text("posterior_stddev", font_size=10)
    p.view_isometric()

    p.link_views()
    p.show()

    # scatter plot
    plt.figure(figsize=(6, 5))
    plt.scatter(stddev, c_error, s=8, alpha=0.4)
    plt.xlabel("posterior stddev")
    plt.ylabel("c_error (normalized)")
    plt.title(f"c_error vs posterior_stddev (r = {corr:.3f})")
    plt.tight_layout()
    plt.savefig("item3_error_vs_stddev.png", dpi=150)
    print("  Saved plot: item3_error_vs_stddev.png")
    plt.show()


# =============================================================================
# main
# =============================================================================

def main():
    grid, field_names = build_grid_from_h5(H5_FILE)

    print("\n=== available fields ===")
    for i, name in enumerate(field_names):
        print(f"  [{i}] {name}")

    measured_points = load_measured_points(MEASURED_POINTS_CSV)
    print(f"\nLoaded {len(measured_points)} measurement points from {MEASURED_POINTS_CSV}")

    item1_field_viewer(grid, field_names)
    item2_uncertainty_with_points(grid, measured_points)
    item3_error_vs_uncertainty(grid)


if __name__ == "__main__":
    main()
