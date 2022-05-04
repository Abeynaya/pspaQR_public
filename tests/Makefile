# MAKE SURE: ttor library should already be compiled 
include ../Makefile.conf
INCDIR = ../include
EXTDIR = ../external
SRCDIR = ../src
OBJDIR = ../obj
LIBDIR = ../lib

# spaQR lib
LDFLAGS += -L$(LIBDIR) -lspaQR

#Necessary
INCLUDE = -I$(INCDIR) -I$(EXTDIR) -I$(EIGENDIR) -I$(BLASDIR) -I$(TTORDIR)
LDFLAGS +=  -L$(BLASLIB)

CFLAGS = -std=c++14 -Wall -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE 
CFLAGS += -DTTOR_MPI
ifdef DEBUG
    CFLAGS += -g -Og -D_GLIBCXX_DEBUG -fsanitize=address
else
    CFLAGS += -g -O3 -DEIGEN_NO_DEBUG -DNDEBUG -Wno-unused-variable 
    CFLAGS += -mtune=native -march=native -fomit-frame-pointer -funroll-loops -ffast-math 
    #-funsafe-math-optimizations
endif

#Metis or Patoh
ifeq ($(USE_METIS),1)
	CFLAGS += -DUSE_METIS
	INCLUDE += -I$(METISDIR)
	LDFLAGS +=  -L$(METISLIB) -lmetis
else
	INCLUDE += -I$(PATOHDIR) 
	LDFLAGS += -L$(PATOHLIB) -lpatoh
endif

#MKL or openblas
ifeq ($(USE_MKL),1)
    CFLAGS += -DUSE_MKL -DEIGEN_USE_MKL_ALL
endif

#if HSL_MC64 routine is available
ifeq ($(HSL_AVAIL),1)
	CFLAGS += -DHSL_AVAIL
	INCLUDE += -I$(HSLDIR) 
	LDFLAGS +=  -L$(FORTRANLIB) -lgfortran -lquadmath -lm -L$(HSLLIB) -lhsl_mc64
endif

ifeq ($(USE_HWLOC),1)
	INCLUDE += -I$(HWLOC_INC)
	LDFLAGS += -L$(HWLOC_LIBS)
	CFLAGS += -DUSE_HWLOC
endif

DEPS := $(wildcard $(INCDIR)/*.h)
OBJ := $(patsubst $(INCDIR)/%.h,$(OBJDIR)/%.o,$(DEPS))

TTOROBJ := $(wildcard $(TTORLIB)/*.o)
all: test_pspaQR

# Executables
test_pspaQR: test_pspaQR.cpp $(TTOROBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(BLAS_PAR_LIBS) $(INCLUDE)  
.PHONY: clean

clean:
	rm -f test_pspaQR

