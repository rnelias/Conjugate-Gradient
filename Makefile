EXECUTABLE     := cg
CUFILES_sm_13  := cg.cu
CUDEPS	       := mv_ops.cu cu_ops.cu cuPrintf.cu
CCFILES	       := 

include ./common.mk

sync:
	rsync -Ccvzr -rsh=ssh . dpucsek@checkers.westgrid.ca:cuda/cg/