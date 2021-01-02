#pragma once

#include <string>
#include <vector>
#include <map>

namespace exr {

struct Image {
  Image(std::size_t w, std::size_t h) : width(w), height(h) {}
  std::size_t width;
  std::size_t height;
  std::map<std::string, std::vector<float>> slices;
};

void write(std::string fileName, const exr::Image& image);

exr::Image read(std::string fileName);

void writeTiled(std::string fileName,
									 std::size_t imageWidth, std::size_t imageHeight,
									 std::size_t tileWidth, std::size_t tileHeight,
									 const std::vector<std::vector<float>>& tiledPixels);

} // end namespace exr
