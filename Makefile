CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

cmix-bit: cmix_bit.cpp
	$(CXX) $(CXXFLAGS) -o cmix-bit cmix_bit.cpp

clean:
	rm -f cmix-bit
