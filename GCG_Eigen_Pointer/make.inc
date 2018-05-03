CC = gcc
CPPFLAGS = -I util/lapack-3.4.2/lapacke/include 

CFLAGS = -g -w -O3 -fexceptions -fPIC -DNCHOLMOD -DNCHOLMOD -DNPARTITION 
LIBS =  util/lapack-3.4.2/liblapacke.a util/lapack-3.4.2/liblapack.a util/blas_LINUX.a  -lm -u \
       -lfblas -lgfortran -ldl -lgcc_s

.SUFFIXES: .c.o
vpath %.h ./
vpath %.c ./
SOURCE = $(wildcard ./*.c) $(wildcard ./$(MAIN).c)
OBJECTS = $(SOURCE:.c=.o)
