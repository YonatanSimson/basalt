// Loads basalt's `<name>_detected_corners.cereal` and dumps to JSON.
// Each entry: {frame_id, cam_id, corners:[[u,v],...], corner_ids:[...]}.

#include <fstream>
#include <iostream>
#include <string>

#include <basalt/calibration/calibration_helper.h>
#include <basalt/serialization/headers_serialization.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <detected_corners.cereal> <output.json>\n";
    return 1;
  }

  std::ifstream is(argv[1], std::ios::binary);
  if (!is.good()) {
    std::cerr << "Cannot open " << argv[1] << "\n";
    return 1;
  }

  basalt::CalibCornerMap calib_corners, calib_corners_rejected;
  {
    cereal::BinaryInputArchive archive(is);
    archive(calib_corners);
    archive(calib_corners_rejected);
  }

  std::ofstream os(argv[2]);
  os << "[";
  bool first = true;
  size_t total_corners = 0;
  for (const auto& kv : calib_corners) {
    const auto& tcid = kv.first;
    const auto& cd = kv.second;
    if (cd.corners.empty()) continue;
    if (!first) os << ",";
    first = false;
    os << "\n  {\"frame_id\":" << tcid.frame_id << ",\"cam_id\":" << tcid.cam_id
       << ",\"corner_ids\":[";
    for (size_t i = 0; i < cd.corner_ids.size(); ++i) {
      if (i) os << ",";
      os << cd.corner_ids[i];
    }
    os << "],\"corners\":[";
    for (size_t i = 0; i < cd.corners.size(); ++i) {
      if (i) os << ",";
      os << "[" << cd.corners[i].x() << "," << cd.corners[i].y() << "]";
    }
    os << "]}";
    total_corners += cd.corners.size();
  }
  os << "\n]\n";

  std::cerr << "Frames: " << calib_corners.size()
            << " corners: " << total_corners << "\n";
  return 0;
}
