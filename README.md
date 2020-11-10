# light

```bash
sudo apt install libboost-all-dev
sudo apt install libopencv-dev
mkdir build
cd build
cmake ../ -G Ninja
ninja
./light --outfile image.png
```