CXX = mpic++
CC = mpicc
CFLAGS = -fopenmp -Wall -Wconversion -O3 -fPIC -pipe 
LIBS = blas/blas.a
#LIBS = -lblas

all: train predict split

split: split.o
	${CXX} ${CFLAGS} -o split split.o 

train: tron.o selectiontree.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c tron.o selectiontree.o linear.o $(LIBS)

predict: tron.o selectiontree.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o selectiontree.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

selectiontree.o: selectiontree.cpp selectiontree.h
	$(CXX) $(CFLAGS) -c -o selectiontree.o selectiontree.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

split.o: split.cpp split.h
	${CXX} ${CFLAGS} -c -o split.o split.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ selectiontree.o tron*.o linear*.o split.o train predict split
