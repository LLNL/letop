#!/usr/bin/env bash


source $1/bin/activate
cd ../
letop=$(pwd)


for d in $letop/letop_examples/*/; do
	if [ "$letop/letop_examples/heat_exchanger/" = "$d" ]; then
		echo "Running ${d}"
		cd $d
		python3 2D_mesh.py
		gmsh -2 -option 2D_mesh.geo.opt 2D_mesh.geo
	fi
	if [ "$letop/letop_examples/cantilever/" = "$d" ]; then
		echo "Running ${d}"
		cd $d
		gmsh -2 -option mesh_cantilever.geo.opt mesh_cantilever.geo
	fi
done
