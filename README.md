[![salazardetroya](https://circleci.com/gh/LLNL/lestofire.svg?style=svg)](https://app.circleci.com/pipelines/github/LLNL/lestofire)
[![codecov](https://codecov.io/gh/LLNL/lestofire/branch/main/graph/badge.svg)](https://codecov.io/gh/LLNL/lestofire)



# Lestofire

Install with

```python
pip3 install .
```

at the project's root directory and with the Firedrake's virtual environment activated ([instructions](https://www.firedrakeproject.org/download.html))

## Level set topology optimization in Firedrake

Lestofire implements the level set method in topology optimization.
It combines the automated calculation of shape derivatives in [Firedrake](https://gitlab.com/florian.feppon/null-space-optimizer) and [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint) with the [Null space optimizer](https://gitlab.com/florian.feppon/null-space-optimizer) to find the optimal design.

Heat exchanger

![heat_exchanger](https://media.giphy.com/media/YhgqJt24PCXJgUdmLu/giphy.gif)

Cantilever

![cantilever](https://media.giphy.com/media/eWze54pzWhoBiiJDmK/giphy.gif)

LLNL Release Number: LLNL-CODE- 817098

## Considerations when using Lestofire

Make use of the pyadjoint context manager `stop_annotating()` and the decorator `no_annotations` for:

- When using `interpolate()` or `project` from Firedrake as they might annotate unnecessary operations and render the shape derivatives wrong.
- Similarly, when extending Lestofire routines, make sure you are not annotating additional operations as Firedrake annotates everything by default when importing `firedrake_adjoint`.

## Heuristics to keep in mind

- The isocontours of the level set must have enough mesh resolution, otherwise the reinitialization solver might fail.
- If the level set is unstable during the optimization, tune the accuracy of the Hamilton-Jacobi solver by settings the options `hj_ts_atol` and `hj_ts_rtol` in `InfDimProblem`'s `solver_parameters` argument. It could also be that the optimization "time step" `dt` is too high.
