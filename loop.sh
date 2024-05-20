#!/bin/bash

#Ciklas prasukti per visus cif failus.
#Failai įvestyje bus .cif formato.
#Išvesties srautas rašomas į stdout ir bus .csv formato.

set -ue

total=$(ls data/*.cif | wc -l)

for FILE in $(ls data/*.cif | tqdm --total $total)
do
	python pymatgen_test.py "${FILE}" output/test_temp.csv
done 
