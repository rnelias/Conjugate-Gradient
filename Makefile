EXECUTABLE := cg
CUFILES	   := mv_ops.cu
CCFILES	   := cg.c

include ./common.mk

sync:
	rsync -Ccvzr -rsh=ssh . dpucsek@checkers.westgrid.ca:cuda/cg/