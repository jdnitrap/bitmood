CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
CPPFLAGS += -Isrc -MMD -MP

SRCS := $(shell find src -name '*.cpp')
OBJS := $(SRCS:src/%.cpp=build/%.o)

cmix-bit: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

test: cmix-bit
	tests/run_tests.sh

clean:
	rm -rf build cmix-bit

.PHONY: test clean

-include $(OBJS:.o=.d)
