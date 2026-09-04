import numpy as np
import matplotlib.pyplot as plt
import re
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "log_linear_alpha1e-3_nodes32.txt"

with open(log_file) as f:
    lines = f.readlines()

# --- parse L-BFGS-B gradient norms (from callback) ---
lbfgs_gnorm = []
in_lbfgs = True
for line in lines:
    if "Newton-CG outer iteration 0" in line:
        in_lbfgs = False
    if in_lbfgs and "||grad||_inf" in line and "gtol = 1e-08" in line:
        val = float(re.search(r":\s+([0-9eE+\-.]+)", line).group(1))
        lbfgs_gnorm.append(val)

# --- parse Newton-CG gradient norms ---
newton_gnorm = []
for line in lines:
    if "||grad||_inf" in line and "gtol = 1.0e-08" in line:
        val = float(re.search(r":\s+([0-9eE+\-.]+)", line).group(1))
        newton_gnorm.append(val)

# --- parse stopping reasons ---
lbfgs_msg  = ""
newton_stop = ""
lbfgs_final_gnorm = None
for line in lines:
    if "L-BFGS-B message" in line:
        lbfgs_msg = line.split(":")[-1].strip()
    if "L-BFGS-B ||grad||_inf" in line:
        lbfgs_final_gnorm = float(re.search(r":\s+([0-9eE+\-.]+)", line).group(1))
    if "Converged (criterion" in line:
        newton_stop = line.strip()

gtol = 1e-8

print(f"L-BFGS-B:  {len(lbfgs_gnorm)} iterations, "
      f"final ||grad||_inf = {lbfgs_gnorm[-1]:.3e}, "
      f"stopping: {lbfgs_msg}")
print(f"Newton-CG: {len(newton_gnorm)} iterations, "
      f"final ||grad||_inf = {newton_gnorm[-1]:.3e}, "
      f"stopping: {newton_stop}")

# --- plot ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(range(1, len(lbfgs_gnorm)+1), lbfgs_gnorm,
            "o-", color="orange", markersize=4, label="L-BFGS-B")
ax.semilogy(range(1, len(newton_gnorm)+1), newton_gnorm,
            "o-", color="steelblue", markersize=6, label="Newton-CG")

ax.axhline(gtol, color="red", linestyle="--", linewidth=1.5,
           label=f"gtol = {gtol:.0e}")

# mark final gradient of each method
ax.plot(len(lbfgs_gnorm),  lbfgs_gnorm[-1],  "*", color="orange",
        markersize=14, zorder=5)
ax.plot(len(newton_gnorm), newton_gnorm[-1], "*", color="steelblue",
        markersize=14, zorder=5)

ax.set_xlabel("Outer iteration", fontsize=12)
ax.set_ylabel(r"$\|\nabla J\|_\infty$", fontsize=12)
ax.set_title("Gradient norm convergence\n"
             f"L-BFGS-B: {lbfgs_msg}\n"
             f"Newton-CG: {newton_stop}",
             fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
outname = log_file.replace(".txt", "_gnorm.png").replace("log_", "plot_gnorm_")
plt.savefig(outname, dpi=150)
print(f"Saved: {outname}")
plt.show()
