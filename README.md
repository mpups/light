# light

Monte Carlo Ray Tacer based on Smallpaint by Károly Zsolnai-Fehér: https://users.cg.tuwien.ac.at/zsolnai/gfx/smallpaint/

## Build instructions

```bash
sudo apt install libboost-all-dev libopencv-dev libopenexr-dev
mkdir build
cd build
cmake ../ -G Ninja
ninja
./light --outfile image.png
```

For a full description of options: `./light --help`

## Example Output

![Box Scene](examples/html/test_png_0_2.jpg)

 In addition to your chosen output format light always saves the raw output as an EXR file. You can use ![pfs tools](http://pfstools.sourceforge.net/). The EXR file can also be used to resume rendering.

### Render the LuxBlend Reference Scene for Comparison

Download the latest version of ![LuxBlend](https://luxcorerender.org/download/) (no need to unzip).
In Blender 2.90.1 goto Edit -> Preferences -> Add-ons -> Install and select the zip file. It will warn about incompatibility with Blender 2.9 but it works fine.
Open luxblend_reference/scene.blend and press F12.