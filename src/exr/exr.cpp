#include "exr.hpp"
#include "convert.hpp"

#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfTiledOutputFile.h>
#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImathBox.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfPixelType.h>
#include <OpenEXR/ImfPartType.h>

#include <spdlog/spdlog.h>

namespace exr {

std::shared_ptr<spdlog::logger> logger() {
    static auto logger = spdlog::stdout_logger_mt("exr_logger");
    return spdlog::get("exr_logger");
}

std::map<Imf::PixelType, std::size_t> bytesPerPixel = {
  {Imf::PixelType::UINT, sizeof(std::uint32_t)},
  {Imf::PixelType::HALF, sizeof(half)},
  {Imf::PixelType::FLOAT, sizeof(float)}
};

std::map<DataType, Imf::PixelType> toImfType = {
  {DataType::UI32, Imf::PixelType::UINT},
  {DataType::FP16, Imf::PixelType::HALF},
  {DataType::FP32, Imf::PixelType::FLOAT}
};

bool checkIsExr(std::string fileName) {
  std::ifstream f(fileName, std::ios_base::binary);
  char b[4];
  f.read(b, 4);
  return !!f && b[0] == 0x76 && b[1] == 0x2f && b[2] == 0x31 && b[3] == 0x01;
}

void write(std::string fileName, const Image& image) {
  using namespace Imf;
  logger()->info("Writing file '{}' {}x{}", fileName, image.width, image.height);
  Header header(image.width, image.height);

  FrameBuffer fb;
  for (const auto& c : image.slices) {
    const auto& name = c.first;
    const auto& slice = c.second;
    logger()->info("Inserting channel {}", name);
    header.channels().insert(name, Channel(PixelType::FLOAT));
    fb.insert(name, Slice(PixelType::FLOAT, (char*)slice.data(),
              sizeof(float), sizeof(float) * image.width));
  }

  OutputFile file(fileName.c_str(), header);
  file.setFrameBuffer(fb);
  file.writePixels(image.height);
}

Image read(std::string fileName) {
  using namespace Imf;

  logger()->info("Reading file '{}'", fileName);
  if (!checkIsExr(fileName)) {
    logger()->error("'{}' is not an EXR file.", fileName);
    throw std::runtime_error("Could not read input.");
  }

  InputFile file(fileName.c_str());
  auto& header = file.header();
  auto& dw = header.dataWindow();
  auto w = dw.max.x - dw.min.x + 1;
  auto h = dw.max.y - dw.min.y + 1;
  logger()->info("Dimensions: {}x{}", w, h);

  FrameBuffer fb;
  Image image(w, h);
  auto& channels = header.channels();
  for (auto itr = channels.begin(); itr != channels.end(); ++itr) {
    const auto& name = itr.name();
    logger()->info("Found channel {}", name);
    auto& c = itr.channel();
    if (c.type != PixelType::FLOAT) {
      logger()->error("Only FLOAT channels are currently supported.");
      throw std::runtime_error("Could not read input.");
    }
    image.slices[name] = std::vector<float>(w * h);
    fb.insert(name, Slice(FLOAT, (char*)image.slices[name].data(),
                          sizeof(float), sizeof(float) * image.width));
  }

  file.setFrameBuffer(fb);
  file.readPixels(dw.min.y, dw.max.y);
  return image;
}

void writeTiled(std::string fileName, DataType fileDataType,
                std::size_t imageWidth, std::size_t imageHeight,
                std::size_t tileWidth, std::size_t tileHeight,
                const std::vector<std::vector<float>>& tiledPixels) {
  using namespace Imf;
  const auto outputType = toImfType.at(fileDataType);

  Header header(imageWidth, imageHeight);
  header.channels().insert("R", Channel(outputType));
  header.channels().insert("G", Channel(outputType));
  header.channels().insert("B", Channel(outputType));
  header.setTileDescription(TileDescription(tileWidth, tileHeight, ONE_LEVEL));

  std::vector<FrameBuffer> framebuffers;
  framebuffers.resize(tiledPixels.size());

  const auto bpp = bytesPerPixel.at(outputType);
  const auto colStride = 3 * bpp;
  const auto rowStride = 3 * tileWidth * bpp;

  std::vector<std::vector<std::uint32_t>> uintTiles;
  if (outputType == PixelType::UINT) {
    uintTiles = floatToUint(tiledPixels);
  }

  std::vector<std::vector<half>> halfTiles;
  if (outputType == PixelType::HALF) {
    halfTiles = floatToHalf(tiledPixels);
  }

  #pragma omp parallel for schedule(dynamic)
  for (auto t = 0u; t < tiledPixels.size(); ++t) {
    auto& tile = tiledPixels[t];
    auto& fb = framebuffers[t];
    char* outData = (char*)tile.data();

    if (outputType == PixelType::UINT) {
      outData = (char*)uintTiles[t].data();
    }

    if (outputType == PixelType::HALF) {
      outData = (char*)halfTiles[t].data();
    }

    fb.insert("R", Slice(outputType, outData, colStride, rowStride, 1, 1, 0.0, true, true));
    fb.insert("G", Slice(outputType, outData + bpp, colStride, rowStride, 1, 1, 0.0, true, true));
    fb.insert("B", Slice(outputType, outData + 2*bpp, colStride, rowStride, 1, 1, 0.0, true, true));
  }

  TiledOutputFile out(fileName.c_str(), header);
  std::size_t tx = 0;
  std::size_t ty = 0;
  std::size_t tilesX = imageWidth / tileWidth;

  for (const auto& fb : framebuffers) {
    out.setFrameBuffer(fb);
    out.writeTiles(tx, tx, ty, ty);
    tx += 1;
    if (tx >= tilesX) {
      tx = 0;
      ty += 1;
    }
  }
}

} // end namespace exr
