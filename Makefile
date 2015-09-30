CXX = g++
CPPFLAGS = -g -Wall -O3 -I. -std=c++11 -fopenmp
CPPFLAGS += -MMD
DEPS = dbscan.h
OBJ = filter_dbscan.o dbscan.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

filter_dbscan : $(OBJ)
	$(CXX) -o $@ $^ $(CPPFLAGS)

.PHONY : all
all: filter_dbscan 

.PHONY : clean
clean:
	-rm filter_dbscan *.o *.d
