RUN WITH: g++ -O3 -std=c++11 run_noGUI.cpp -o app_noGUI -I/home/rpi3/ncnn/build/install/include     -L/home/rpi3/ncnn/build/install/lib     `pkg-config --cflags --libs opencv4`     -lncnn -fopenmp
