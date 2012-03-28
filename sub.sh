#!/bin/bash
#PBS -S /bin/bash
#PBS -N cg

#PBS -M dean@lightbulbone.com
#PBS -m bea
#PBS -l procs=1
#PBS -l walltime=01:00:00
#PBS -l mem=4000mb
#PBS -l pmem=4000mb

/home/dpucsek/cg/cg /home/dpucsek/cg/input/Ab.txt 30 10
