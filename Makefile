# The compiler to use.
CC = mpicxx

# Directories for Includes and Common clases
INCDIR= ./include/
SRCDIR= ./src/
BINDIR= ./bin/

# Directories and files for documentation
DOCDIR= ./doc/
DOXYFILE= $(DOCDIR)Doxyfile

# Compiler options -Weffc++
CFLAGS= -Wall -O2 -fopenmp -I$(INCDIR) -std=c++11 -lz -lboost_iostreams

# Sources and Common clases sources
CSOURCES= $(SRCDIR)Functions.cpp $(SRCDIR)TimeLog.cpp
SOURCESC= $(SRCDIR)BuildFeatureSimilarityMatrixByClass.cpp

# Objects
COBJECTS=$(CSOURCES:.cpp=.o)
OBJECTSC=$(SOURCESC:.cpp=.o)

# Name of the executable
EXECUTABLEC=$(BINDIR)BuildFeatureSimilarityMatrixByClass

# Declare phony targets:
.PHONY: clean all doc

all: $(EXECUTABLEC)

$(EXECUTABLEC): $(OBJECTSC) $(COBJECTS)
	$(CC) $(CFLAGS) $(OBJECTSC) $(COBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@
	
$(DOXYFILE):
	cd $(DOCDIR)
	
doc:
	doxygen $(DOXYFILE)

clean:
	rm -f $(OBJECTS) $(COBJECTS) $(OBJECTSCONVERT) $(OBJECTSC) $(EXECUTABLE) $(EXECUTABLEC)
