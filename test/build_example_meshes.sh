#!/usr/bin/env bash


source $1/bin/activate
cd ../
lestofire=$(pwd)


for d in $lestofire/lestofire_examples/*/; do
	if [ "$lestofire/lestofire_examples/heat_exchanger/" = "$d" ]; then
		echo "Running ${d}"
		cd $d
		python3 2D_mesh.py
		gmsh -2 -option 2D_mesh.msh.opt 2D_mesh.geo
	fi
	if [ "$lestofire/lestofire_examples/cantilever/" = "$d" ]; then
		echo "Running ${d}"
		cd $d
		gmsh -2 -option mesh_cantilever.msh.opt mesh_cantilever.geo
	fi
done
