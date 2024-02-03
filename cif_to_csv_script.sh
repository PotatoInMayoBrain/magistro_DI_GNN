#!/bin/bash

#Ciklas prasukti per visus cif failus.
#Failai įvestyje bus .cif formato.
#Failai išvesties srautas rašomas į stdout ir bus .csv formato.

set -ue

for FILE in adjacency_matrices/*.csv
do
	python3 gnn.py "${FILE}" output/
done
