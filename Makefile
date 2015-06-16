CC=g++
CFLAGS= -std=c++11


all: main.o sptw.o worker_debug
	$(CC) -o main $< 

main.o: main.cc
	$(CC) -c -o $@ main.cc

sptw.o: sptw/sptw.cc sptw/sptw.h sptw/utils.h
	$(CC) -c -o $@ sptw/sptw.cc

worker_debug: worker.cpp
	$(CC) $(CFLAGS) -I armadillo-5.200.1/include -o $@ $^ -lblas -llapack

clean:
	rm worker_debug
