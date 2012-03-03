EXECUTABLE := cg
CUFILES	   := cg.cu cu_ops.cu mv_ops.cu
CCFILES	   := 

include ./common.mk

sync:
	rsync -Ccvzr -rsh=ssh . dpucsek@checkers.westgrid.ca:cuda/cg/