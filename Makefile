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
	./cg input/test.txt 30

run-test-suppressed:
	./cg input/test.txt 30 n

run-full:
	./cg input/Ab.txt 30

run-full-suppressed:
	./cg input/Ab.txt 30 n