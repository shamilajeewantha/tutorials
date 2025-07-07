CC=gcc
CXX=g++
CFLAGS=-Wall
CXXFLAGS=-Wall
LDFLAGS=

OBJS=main.o utils.o
TARGET=my_app

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

main.o: main.c main.h
	$(CC) $(CFLAGS) -c main.c -o main.o

utils.o: utils.cpp main.h
	$(CXX) $(CXXFLAGS) -c utils.cpp -o utils.o

clean:
	rm -f *.o $(TARGET)
