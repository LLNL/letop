# Lestofire

## Level set topology optimization in Firedrake
Lestofire implements the level set method in topology optimization.
It combines the automated calculation of shape derivatives in [Firedrake](https://gitlab.com/florian.feppon/null-space-optimizer) and (pyadjoint)[https://github.com/dolfin-adjoint/pyadjoint] with the (Null space optimizer)[https://gitlab.com/florian.feppon/null-space-optimizer] to find the optimal design.

Cantilever

![Cantilever](/uploads/0d1f160bf113ae98b8ba0eaaedb8ca44/cantilever.gif)

Heat exchanger

![heat_exchanger](/uploads/00c4561421266ec316903804f0198c1e/heat_exchanger.gif)
LLNL Release Number: LLNL-CODE- 817098
