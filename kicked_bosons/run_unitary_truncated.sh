#!/bin/bash

njobs=10
# Source any environment stuff you need
#source /etc/profile.d/conda.sh
#conda activate quantum-chaos
source $HOME/miniforge3-qist/etc/profile.d/conda.sh
conda activate quantum-chaos-miniforge

# Export any variables that you want the bash function to be able to see
export exec_folder=${HOME}/GitHub/quantum-chaos/kicked_bosons
export root_folder=/scratch/NemotoU/henry/quantum-chaos/kicked_bosons

# The parameters you want to parallelize over (can also get this from an external
# parameter file)
M_arr=(10 20 50 100 200 300 400 500)
N_arr=(2)
num_ensembles_arr=(100)
num_repeats_arr=(1000)
time_arr=($((10**12)) "heisenberg")
theta_W_pairs=("7.4 7" "7.4 3.5" "7.4 2" "18 3" "20 0.5")

# The shell script function GNU-parallel calls  
unitary_truncated() {
  # $1,$2,$3, etc are the parameters parallel passes, in the order you give them
  M=$1
  N=$2
  num_ensembles=$3
  num_repeats=$4
  timee=$5

  # Split pairs
  set -- $6
  thetaOmega=$1
  WOmega=$2
  
  # stdout and stderr get redirected to this file
  outfile=unitary_truncated_M${M}_N${N}_num_ensembles${g}_num_repeats${num_repeats}_time_${timee}_thetaOmega${thetaOmega}_WOmega${WOmega}.out

  # This is basically the program I call, with the parameters I set
  python3 -u ${exec_folder}/save_unitary_truncated.py \
    -M ${M} \
    -N ${N} \
    -num_ensembles ${num_ensembles} \
    -num_repeats ${num_repeats} \
    -thetaOmega ${thetaOmega} \
    -WOmega ${WOmega} \
    -time ${timee} \
    -root_folder ${root_folder} \
    -save_data \
	  &> ${outfile}
}
# You must export the function for the shell script to know about it
export -f unitary_truncated

# Run in parallel using GNU parallel
parallel -j${njobs} unitary_truncated ::: "${M_arr[@]}" ::: "${N_arr[@]}" ::: "${num_ensembles_arr[@]}" ::: "${num_repeats_arr[@]}" ::: "${time_arr[@]}" ::: "${theta_W_pairs[@]}"