CXX = g++
CC = gcc
CXXFLAGS = -std=c++17 -Wall -I./ -I  /usr/local/include/
CFLAGS = -Wall -I./

# CPR static lib path (adjust to your CPR clone path)
CPR_LIB = /usr/local/lib/libcpr.a

# Link flags: curl, pthread, ssl, crypto (often needed for curl)
LDFLAGS = -lcurl -lpthread -lssl -lcrypto

OBJS = main.o utils.o

TARGET = my_app

all: $(TARGET)

main.o: main.c main.h
	$(CC) $(CFLAGS) -c main.c -o main.o

utils.o: utils.cpp main.h
	$(CXX) $(CXXFLAGS) -c utils.cpp -o utils.o

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(CPR_LIB) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f *.o $(TARGET)
