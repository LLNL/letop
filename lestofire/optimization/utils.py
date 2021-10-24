import firedrake_adjoint as fda
import re
import glob
import firedrake as fd


def read_checkpoint(output_dir, phi):
    with fda.stop_annotating():
        checkpoints = glob.glob(f"{output_dir}/checkpoint*")
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda L: list(map(int, re.findall(r"iter_(\d+)\.h5", L)))[0],
        )
        if checkpoints_sorted:
            last_file = checkpoints_sorted[-1]
            current_iter = int(re.findall(r"iter_(\d+)\.h5", last_file)[0])
            with fd.HDF5File(
                f"{output_dir}/checkpoint_iter_{current_iter}.h5", "r"
            ) as checkpoint:
                checkpoint.read(phi, "/checkpoint")
            print(f"Restarting simulation at {current_iter}")
            return True
        else:
            return False
