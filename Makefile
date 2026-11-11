# Build the sequential baseline and the hybrid MPI + OpenMP program.
#   make          build everything into ./build
#   make clean    remove build artifacts
#   make test     build and run the correctness test-suite
CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

all: build/wc_serial build/wc_mpi

build/wc_serial: src/wc_serial.cpp src/common.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ src/wc_serial.cpp

build/wc_mpi: src/wc_mpi.cpp src/common.hpp
	@mkdir -p build
	$(MPICXX) $(CXXFLAGS) -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -fopenmp -o $@ src/wc_mpi.cpp

test: all
	bash tests/run_tests.sh

clean:
	rm -rf build

.PHONY: all test clean
