#!/bin/bash

njobs=10
# Source any environment stuff you need
source ${HOME}/environment_qist/quantum-chaos/bin/activate

# Export any variables that you want the bash function to be able to see
export exec_folder=${HOME}/GitHub/quantum-chaos/kicked_bosons
export root_folder=/scratch/NemotoU/henry/quantum-chaos/kicked_bosons
export exec_file=level_spacing_ratio.py

# The parameters you want to parallelize over (can also get this from an external
# parameter file)
M_arr=(50 100 150 200 250 300 350 400 450 500)
num_ensembles_arr=(100)
thetaOmega_arr=$(seq 0 0.2 20)
WOmega_arr=$(seq 1 0.1 9)

# The shell script function GNU-parallel calls  
bosons() {
  # $1,$2,$3, etc are the parameters parallel passes, in the order you give them
  M=$1
  num_ensembles=$2
  thetaOmega=$3
  WOmega=$4
  
  # stdout and stderr get redirected to this file
  outfile=bosons_M${M}_num_ensembles${g}_thetaOmega${thetaOmega}_WOmega${WOmega}.out

  # This is basically the program I call, with the parameters I set
  python3 -u ${exec_folder}/${exec_file} \
    -M ${M} \
    -num_ensembles ${num_ensembles} \
    -thetaOmega ${thetaOmega} \
    -WOmega ${WOmega} \
    -root_folder ${root_folder} \
    -save_plots 0 -show_plots 0 \
    -save_data 1 \
	  &> ${outfile}
}
# You must export the function for the shell script to know about it
export -f bosons

# Run in parallel using GNU parallel
#parallel -j${njobs} --memsuspend 2G bosons ::: "${M_arr[@]}" ::: "${num_ensembles_arr[@]}" ::: "${thetaOmega_arr[@]}" ::: "${WOmega_arr[@]}"
parallel -j${njobs} bosons ::: "${M_arr[@]}" ::: "${num_ensembles_arr[@]}" ::: "${thetaOmega_arr[@]}" ::: "${WOmega_arr[@]}"