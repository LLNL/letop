#!/usr/bin/env bash


source $1/bin/activate
for d in ../examples/*/; do
	if [ "../examples/heat_exchanger/" = "$d" ]; then
		cd $d
		python3 heat_exchanger_al.py > ../../test/output.txt
		cd ../../test/
		python3 check_examples_output.py output.txt heat_exchanger
	fi
done
