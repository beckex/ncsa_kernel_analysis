CC=g++
CFLAGS=-I.


all: main.o sptw.o
	$(CC) -o main $< 

main.o: main.cc
	$(CC) -c -o $@ main.cc

sptw.o: sptw/sptw.cc sptw/sptw.h sptw/utils.h
	$(CC) -c -o $@ sptw/sptw.cc
