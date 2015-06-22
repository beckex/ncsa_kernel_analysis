CC=g++
CFLAGS= -std=c++0x


all: main.o sptw.o worker.o
	$(CC) -o main $< 

main.o: main.cc
	$(CC) -c -o $@ main.cc

sptw.o: sptw/sptw.cc sptw/sptw.h sptw/utils.h
	$(CC) -c -o $@ sptw/sptw.cc

worker.o: worker.cpp worker.h
	$(CC) $(CFLAGS) -I armadillo-5.200.1/include -c $^

clean:
	rm *.o *.h.gch
