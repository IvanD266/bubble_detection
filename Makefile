# Makefile for ncnn Bubble Detection
# Usage: make          (build)
#        make clean    (remove binary)
#        make run      (build + execute)

CXX       := g++
CXXFLAGS  := -O3 -std=c++11 -fopenmp -Wall

# ncnn paths (edit if you move ncnn)
NCNN_ROOT := /home/rpi3/ncnn/build/install
NCNN_INC  := -I$(NCNN_ROOT)/include
NCNN_LIB  := -L$(NCNN_ROOT)/lib

# OpenCV flags via pkg-config
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# Target & Source
TARGET := app_noGUI
SRC    := run_noGUI.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(NCNN_INC) $(OPENCV_CFLAGS) $< -o $@ $(NCNN_LIB) -lncnn $(OPENCV_LIBS)

clean:
	rm -f $(TARGET)

run: all
	LD_LIBRARY_PATH=$(NCNN_LIB) ./$(TARGET)
