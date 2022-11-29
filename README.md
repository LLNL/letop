![CircleCI](https://img.shields.io/circleci/build/gh/LLNL/letop)
[![codecov](https://codecov.io/gh/LLNL/lestofire/branch/main/graph/badge.svg)](https://codecov.io/gh/LLNL/letop)



<div align="center">
  <img width="300px" src="https://user-images.githubusercontent.com/7770764/139541514-7938159f-bfb9-41f4-8109-4f4388ca17b5.png">
</div>

## Level set topology optimization in Firedrake 🚧 🚧 (in progress)

LeToP implements the level set method in topology optimization.
It combines the automated calculation of shape derivatives in [Firedrake](https://gitlab.com/florian.feppon/null-space-optimizer) and [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint) with the [Null space optimizer](https://gitlab.com/florian.feppon/null-space-optimizer) to find the optimal design. The level set in advected with a Hamilton-Jacobi equation and properly reinitialized to maintain the signed distance property.

The user interface is very close to pyadjoint to allow for easy compatibility.

# Installation

Install with

```python
pip3 install .
```
at the project's root directory and with the Firedrake's virtual environment activated.
LeTop depends on the [10.5281/zenodo.7017917](https://zenodo.org/record/7017917#.Y4YXbLLMKK0), which can be simply installed by passing the flag `--doi 10.5281/zenodo.7017917` to `firedrake-install`.

## Installation issues
```
src/C/umfpack.c:23:10: fatal error: 'umfpack.h' file not found
```
The package `cvxopt` cannot find the suitesparse library, which should be within PETSc (check the option `--download-suitesparse` was passed). Create the following environment variables before installing `letop`:
```
export CVXOPT_SUITESPARSE_LIB_DIR=$PETSC_DIR/$PETSC_ARCH/lib 
export CVXOPT_SUITESPARSE_INC_DIR=$PETSC_DIR/$PETSC_ARCH/include
```

# Examples
## Cantilever

![cantilever](https://media.giphy.com/media/eWze54pzWhoBiiJDmK/giphy.gif)

## Heat exchanger

![heat_exchanger_3D](https://user-images.githubusercontent.com/7770764/139540221-8195f162-3850-4939-b116-e466fdb9d8b5.gif)

## Bridge

![bridge](https://user-images.githubusercontent.com/7770764/139540289-b7daff65-5c98-4828-8a07-445aab79b7bb.gif)



LLNL Release Number: LLNL-CODE-817098

## Considerations when using LeToP

Make use of the pyadjoint context manager `stop_annotating()` and the decorator `no_annotations` for:

- When using `interpolate()` or `project` from Firedrake as they might annotate unnecessary operations and result in wrong the shape derivatives.
- Similarly, when extending LeToP routines, make sure you are not annotating additional operations as Firedrake annotates everything by default when importing `firedrake_adjoint`.
- The shape velocities should be zeroed on the Dirichlet and Neumann boundaries. Use the `RegularizationSolver` boundary conditions to this effect.
- Add fd.parameters["form_compiler"]["quadrature_degree"] = 4 to ensure quadrature does not go too high due to the heaviside function (used to mark the subdomains)
- The isocontours of the level set must have enough mesh resolution, otherwise the reinitialization solver might fail.
