#!/usr/bin/env bash


source $1/bin/activate
cd ../
lestofire=$(pwd)


for d in $lestofire/examples/*/; do
	if [ "$lestofire/examples/heat_exchanger/" = "$d" ]; then
		echo "Running ${d}"
		cd $d
		python3 2D_mesh.py
		gmsh -2 -option 2D_mesh.msh.opt 2D_mesh.geo
		python3 heat_exchanger_nls.py --n_iters 12 | tee $lestofire/test/output.txt
		cd $lestofire/test/
		if ! python3 check_examples_output.py output.txt heat_exchanger 2> stderr.txt; then
			echo "problems"
			exit 1
		else
			echo "no problems"
		fi
	fi
	if [ "$lestofire/examples/cantilever/" = "$d" ]; then
		echo "Running ${d}"
		cd $d
		gmsh -2 -option mesh_cantilever.msh.opt mesh_cantilever.geo
		python3 cantilever.py --n_iters 12 | tee $lestofire/test/output.txt
		cd $lestofire/test/
		if ! python3 check_examples_output.py output.txt cantilever 2> stderr.txt; then
			echo "problems"
			exit 1
		else
			echo "no problems"
		fi
	fi
#	if [ "$lestofire/examples/stokes/" = "$d" ]; then
#		echo "Running ${d}"
#		cd $d
#		python3 mesh_stokes_flow.py
#		gmsh -2 mesh_stokes.geo
#		python3 stokes.py | tee $lestofire/test/output.txt
#		cd $lestofire/test/
#		if ! python3 check_examples_output.py output.txt stokes 2> stderr.txt; then
#			echo "problems"
#			exit 1
#		else
#			echo "no problems"
#		fi
#	fi
done
