CC=gcc
CFLAGS=-Wall -g
LIBS=
OBJS=mv_ops.c cg.c
PROG_NAME=cg

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

run-test:
	mpiexec -n 5 ./cg 30 < input/test.txt

run-test-suppressed:
	mpiexec -n 5 ./cg 30 n < input/test.txt

run-full:
	mpiexec -n 100 ./cg 30 < input/Ab.txt

run-full-suppressed:
	mpiexec -n 100 ./cg 30 n < input/Ab.txt