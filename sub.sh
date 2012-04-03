#!/bin/bash
#PBS -S /bin/bash
#PBS -N cg

#PBS -M dean@lightbulbone.com
#PBS -m bea
#PBS -q nestor
#PBS -l procs=1
#PBS -l walltime=04:00:00
#PBS -l mem=16gb
#PBS -l pmem=12gb

mpiexec /home/dpucsek/cg/cg /home/dpucsek/cg/input/Ab.txt 30 30
