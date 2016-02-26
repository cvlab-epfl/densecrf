# update the path variables 

CC	= g++
CFLAGS	= -W -Wall -O2
#CFLAGS	= -W -Wall -g

all:
	make clean
	make prog_test_densecrf
	make prog_refine_pascal_nat

clean:
	rm -f *.a
	rm -f *.o
	rm -f prog_test_densecrf
	rm -f prog_refine_pascal_nat

libDenseCRF.a: libDenseCRF/bipartitedensecrf.cpp libDenseCRF/densecrf.cpp libDenseCRF/filter.cpp libDenseCRF/permutohedral.cpp libDenseCRF/util.cpp libDenseCRF/densecrf.h libDenseCRF/fastmath.h libDenseCRF/permutohedral.h libDenseCRF/sse_defs.h libDenseCRF/util.h
	$(CC) -w libDenseCRF/bipartitedensecrf.cpp libDenseCRF/densecrf.cpp libDenseCRF/filter.cpp libDenseCRF/permutohedral.cpp libDenseCRF/util.cpp -c $(CFLAGS)
	ar rcs libDenseCRF.a bipartitedensecrf.o densecrf.o filter.o permutohedral.o util.o

prog_test_densecrf: test_densecrf/simple_dense_inference.cpp libDenseCRF.a
	$(CC) test_densecrf/simple_dense_inference.cpp -o prog_test_densecrf $(CFLAGS) -L. -lDenseCRF

prog_refine_pascal_nat: refine_pascal_nat/dense_inference.cpp util/Timer.h libDenseCRF.a
	$(CC) -w refine_pascal_nat/dense_inference.cpp -o prog_refine_pascal_nat $(CFLAGS) -L. -lDenseCRF -lmatio -lhdf5 -fopenmp -lpthread -I./util/

