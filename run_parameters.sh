#!/bin/bash

njobs=10
source ${HOME}/venv/qutip_qist/bin/activate
export exec_folder=${HOME}/GitHub/kicked-boson/
export root_folder=/scratch/NemotoU/henry/kicked-boson/

N_arr=(300)
num_ensembles_arr=(10)
KChi_arr=$(seq 0 0.5 50)
phi_noise_arr=$(seq 0.005 0.001 0.05)

bosons() {
  N=$1
  num_ensembles=$2
  KChi=$3
  phi_noise=$4
  
  outfile=bosons_N${N}_num_ensembles${g}_KChi${KChi}_phi_noise${phi_noise}.out

  python3 -u ${exec_folder}/quantum_bosons.py \
    -N ${N} \
    -num_ensembles ${num_ensembles} \
    -KChi ${KChi} \
    -phi_noise ${phi_noise} \
    -root_folder ${root_folder} \
    -save_plots 1 -show_plots 0 \
    -save_data 1 \
	  &> ${outfile}
}
export -f bosons

# Run in parallel (indexed as alpha, N, g) using GNU parallel
#parallel -j${njobs} --memsuspend 2G bosons ::: "${N_arr[@]}" ::: "${num_ensembles_arr[@]}" ::: "${KChi_arr[@]}" ::: "${phi_noise_arr[@]}"
parallel -j${njobs} bosons ::: "${N_arr[@]}" ::: "${num_ensembles_arr[@]}" ::: "${KChi_arr[@]}" ::: "${phi_noise_arr[@]}"