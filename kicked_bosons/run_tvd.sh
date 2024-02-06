#!/bin/bash

njobs=10
# Source environment
#source /etc/profile.d/conda.sh
#conda activate quantum-chaos
source $HOME/miniforge3-qist/etc/profile.d/conda.sh
conda activate quantum-chaos-miniforge

export exec_folder=${HOME}/GitHub/quantum-chaos/kicked_bosons
export root_folder=/scratch/NemotoU/henry/quantum-chaos/kicked_bosons

# Parameters
export N=2
export num_ensembles=100000
export num_repeats=100

# For truncated haar distribution
M_haar_arr=(2 5 10 15 20 30 40 50 75 100)

# For kicked bosons
M_boson_arr=(10 20 50 100 200 300)
#M_boson_arr=(10 20 50 100 200 300 400 500)
theta_W_pairs=("7.4 7" "7.4 3.5" "7.4 2" "18 3" "20 0.5")
time_arr=($((10**12)) "heisenberg")

normal() {
  outfile=tvd_normal.out

  python3 -u ${exec_folder}/tvd.py \
    -N ${N} \
    -num_ensembles ${num_ensembles} \
    -num_repeats ${num_repeats} \
    -sample_type normal \
    -root_folder ${root_folder} \
    -random_ensemble_size \
    -save_data \
	  &> ${outfile}
}
export -f normal

truncated_haar() {
  M=$1
  outfile=tvd_haar_M${M}.out

  python3 -u ${exec_folder}/tvd.py \
    -M ${M} \
    -N ${N} \
    -num_ensembles ${num_ensembles} \
    -num_repeats ${num_repeats} \
    -sample_type haar \
    -root_folder ${root_folder} \
    -random_ensemble_size \
    -save_data \
	  &> ${outfile}
}
export -f truncated_haar

truncated_boson() {
  M=$1
  timee=$2

  # Split pairs
  set -- $3
  thetaOmega=$1
  WOmega=$2
  
  outfile=tvd_boson_M${M}_time${timee}_thetaOmega${thetaOmega}_WOmega${WOmega}.out
  
  python3 -u ${exec_folder}/tvd.py \
    -M ${M} \
    -N ${N} \
    -num_ensembles ${num_ensembles} \
    -num_repeats ${num_repeats} \
    -sample_type kicked-boson \
    -thetaOmega ${thetaOmega} \
    -WOmega ${WOmega} \
    -time ${timee} \
    -root_folder ${root_folder} \
    -random_ensemble_size \
    -save_data \
	  &> ${outfile}
}
export -f truncated_boson

#normal
#parallel -j${njobs} truncated_haar ::: "${M_haar_arr[@]}"
parallel -j${njobs} truncated_boson ::: "${M_boson_arr[@]}" ::: "${time_arr[@]}" ::: "${theta_W_pairs[@]}"