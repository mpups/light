#pragma once

#include <string>
#include <vector>

#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfTiledOutputFile.h>
#include <OpenEXR/ImathVec.h>

void writeTiledExr(std::string fileName,
									 std::size_t imageWidth, std::size_t imageHeight,
									 std::size_t tileWidth, std::size_t tileHeight,
									 const std::vector<std::vector<float>>& tiledPixels);
