
[![salazardetroya](https://circleci.com/gh/salazardetroya/lestofire.svg?style=svg)](https://app.circleci.com/pipelines/github/salazardetroya/lestofire)
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
