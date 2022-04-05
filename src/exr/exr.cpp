#include "exr.hpp"

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

#include "logging.hpp"

namespace exr {

std::shared_ptr<spdlog::logger> logger() {
    static auto logger = spdlog::stdout_logger_mt("exr_logger");
    return spdlog::get("exr_logger");
}

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
    header.channels().insert(name, Channel(FLOAT));
    fb.insert(name, Slice(FLOAT, (char*)slice.data(),
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

void writeTiled(std::string fileName,
                   std::size_t imageWidth, std::size_t imageHeight,
                   std::size_t tileWidth, std::size_t tileHeight,
                   const std::vector<std::vector<float>>& tiledPixels) {
  using namespace Imf;
  Header header(imageWidth, imageHeight);
  header.channels().insert("R", Channel(FLOAT));
  header.channels().insert("G", Channel(FLOAT));
  header.channels().insert("B", Channel(FLOAT));
  header.setTileDescription(TileDescription(tileWidth, tileHeight, ONE_LEVEL));

  std::vector<FrameBuffer> framebuffers;
  framebuffers.resize(tiledPixels.size());

  #pragma omp parallel for schedule(dynamic)
  for (std::size_t t = 0; t < tiledPixels.size(); ++t) {
    auto& tile = tiledPixels[t];
    auto& fb = framebuffers[t];
    auto colStride = 3 * sizeof(float);
    auto rowStride = 3 * tileWidth * sizeof(float);
    fb.insert("R", Slice(FLOAT, (char*)tile.data(), colStride, rowStride, 1, 1, 0.0, true, true));
    fb.insert("G", Slice(FLOAT, (char*)(tile.data() + 1), colStride, rowStride, 1, 1, 0.0, true, true));
    fb.insert("B", Slice(FLOAT, (char*)(tile.data() + 2), colStride, rowStride, 1, 1, 0.0, true, true));
  }

  TiledOutputFile out(fileName.c_str(), header);
  std::size_t tx = 0;
  std::size_t ty = 0;
  std::size_t tilesX = imageWidth / tileWidth;
  std::size_t tiles = framebuffers.size();

  for (std::size_t t = 0; t < tiles; ++t) {
    out.setFrameBuffer(framebuffers[t]);
    out.writeTiles(tx, tx, ty, ty);
    tx += 1;
    if (tx >= tilesX) {
      tx = 0;
      ty += 1;
    }
  }
}

} // end namespace exr
