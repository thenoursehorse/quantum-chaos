#!/bin/bash

njobs=10
source ${HOME}/venv/kicked-boson_qist/bin/activate
export exec_folder=${HOME}/GitHub/quantum-chaos/kicked_bosons
export root_folder=/scratch/NemotoU/henry/quantum-chaos/kicked_bosons

M_arr=(300)
num_ensembles_arr=(100)
thetaOmega_arr=$(seq 0 0.2 20)
WOmega_arr=$(seq 1 0.1 9)
    
bosons() {
  M=$1
  num_ensembles=$2
  thetaOmega=$3
  WOmega=$4
    
  outfile=bosons_M${M}_num_ensembles${g}_thetaOmega${thetaOmega}_WOmega${WOmega}.out

  python3 -u ${exec_folder}/make_data.py \
    -M ${M} \
    -num_ensembles ${num_ensembles} \
    -thetaOmega ${thetaOmega} \
    -WOmega ${WOmega} \
    -root_folder ${root_folder} \
    -save_plots 0 -show_plots 0 \
    -save_data 1 \
	  &> ${outfile}
}
export -f bosons

# Run in parallel using GNU parallel
#parallel -j${njobs} --memsuspend 2G bosons ::: "${M_arr[@]}" ::: "${num_ensembles_arr[@]}" ::: "${thetaOmega_arr[@]}" ::: "${WOmega_arr[@]}"
parallel -j${njobs} bosons ::: "${M_arr[@]}" ::: "${num_ensembles_arr[@]}" ::: "${thetaOmega_arr[@]}" ::: "${WOmega_arr[@]}"