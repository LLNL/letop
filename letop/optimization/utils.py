import firedrake_adjoint as fda
from pyadjoint import no_annotations
import re
import glob
import firedrake as fd


@no_annotations
def is_checkpoint(output_dir):
    checkpoints = glob.glob(f"{output_dir}/checkpoint*")
    checkpoints_sorted = sorted(
        checkpoints,
        key=lambda L: list(map(int, re.findall(r"iter_(\d+)\.h5", L)))[0],
    )
    return checkpoints_sorted


@no_annotations
def read_checkpoint(checkpoints, phi):
    last_file = checkpoints[-1]
    current_iter = int(re.findall(r"iter_(\d+)\.h5", last_file)[0])
    with fd.HDF5File(last_file, "r") as checkpoint:
        checkpoint.read(phi, "/checkpoint")
    print(f"Restarting simulation at {current_iter}")
    return current_iter
