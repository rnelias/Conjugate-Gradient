CC=mpicc
CFLAGS=-Wall -g
LIBS=
OBJS=mv_ops.c cg.c
PROG_NAME=cg
NUM_PROCS=5

all:
	${CC} ${CFLAGS} ${LIBS} -o ${PROG_NAME} ${OBJS}

clean:
	rm -rf ${PROG_NAME}
	rm -rf ${PROG_NAME}.dSYM
	rm -rf ${PROG_NAME}.e*
	rm -rf ${PROG_NAME}.o*
	rm -rf *~

sync:
	rsync -Ccvzr -rsh=ssh . dpucsek@checkers.westgrid.ca:~/p2

run-test_mv_ops:
	mpiexec -n ${NUM_PROCS} ./cg input/mv_ops.txt 30

run-test:
	mpiexec -n ${NUM_PROCS} ./cg input/test.txt 30

run-test-suppressed:
	mpiexec -n ${NUM_PROCS} ./cg input/test.txt 30 n

run-full:
	mpiexec -n ${NUM_PROCS} ./cg input/Ab.txt 30

run-full-suppressed:
	mpiexec -n ${NUM_PROCS} ./cg input/Ab.txt 30 n