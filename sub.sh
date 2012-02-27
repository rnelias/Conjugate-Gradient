#!/bin/bash
#PBS -S /bin/bash
#PBS -N cg

#PBS -l procs=1
#PBS -l walltime=01:00:00
#PBS -l mem=4000mb
#PBS -l pmem=4000mb

mpiexec /home/dpucsek/cg/cg /home/dpucsek/cg/input/Ab.txt 30
