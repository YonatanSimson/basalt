#include <basalt/io/dataset_io.h>
#include <basalt/io/dataset_io_mp4.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

namespace fs = std::filesystem;

// Generates a tiny MP4 with a moving square so each frame has unique content.
// Returns true on success. fps is honored exactly.
bool write_synthetic_mp4(const std::string& path, int width, int height,
                         int num_frames, double fps) {
  cv::VideoWriter writer;
  // mp4v fourcc + ".mp4" container is widely supported by FFmpeg builds.
  const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  if (!writer.open(path, fourcc, fps, cv::Size(width, height), true)) {
    return false;
  }
  for (int k = 0; k < num_frames; k++) {
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(20, 20, 20));
    const int x = (k * 5) % (width - 16);
    const int y = (k * 3) % (height - 16);
    cv::rectangle(frame, cv::Rect(x, y, 16, 16),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    writer.write(frame);
  }
  writer.release();
  return fs::exists(path) && fs::file_size(path) > 0;
}

class TmpDir {
 public:
  TmpDir() {
    char tmpl[] = "/tmp/basalt_mp4_test_XXXXXX";
    char* dir = mkdtemp(tmpl);
    if (!dir) std::abort();
    path_ = dir;
  }
  ~TmpDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

}  // namespace

TEST(DatasetIoMp4, MonoFromFile) {
  TmpDir tmp;
  const std::string mp4 = tmp.path() + "/mono.mp4";
  const int W = 96, H = 64, N = 24;
  const double FPS = 24.0;
  ASSERT_TRUE(write_synthetic_mp4(mp4, W, H, N, FPS))
      << "VideoWriter failed; this build of OpenCV is missing the FFmpeg "
         "backend.";

  auto io = basalt::DatasetIoFactory::getDatasetIo("mp4");
  io->read(mp4);
  auto ds = io->get_data();
  ASSERT_TRUE(ds.get() != nullptr);
  EXPECT_EQ(ds->get_num_cams(), 1u);
  ASSERT_EQ(static_cast<int>(ds->get_image_timestamps().size()), N);

  // Timestamps should be monotonically increasing.
  const auto& ts = ds->get_image_timestamps();
  for (size_t i = 1; i < ts.size(); i++) {
    EXPECT_GT(ts[i], ts[i - 1]);
  }

  // Frame stride should match 1/FPS within rounding.
  const int64_t expected_stride = static_cast<int64_t>(1e9 / FPS);
  for (size_t i = 1; i < ts.size(); i++) {
    EXPECT_NEAR(static_cast<double>(ts[i] - ts[i - 1]),
                static_cast<double>(expected_stride), 1.0);
  }

  // First frame: middle pixel should be the dark background; the white
  // square is at (0,0)..(15,15), so center (W/2,H/2) is dark.
  auto imgs = ds->get_image_data(ts[0]);
  ASSERT_EQ(imgs.size(), 1u);
  ASSERT_TRUE(imgs[0].img.get() != nullptr);
  EXPECT_EQ(imgs[0].img->w, W);
  EXPECT_EQ(imgs[0].img->h, H);
  const uint16_t center = (*imgs[0].img)(W / 2, H / 2);
  EXPECT_LT(center, static_cast<uint16_t>(100) << 8);

  // Random-access: read last frame, then jump back to first.
  auto last = ds->get_image_data(ts.back());
  ASSERT_TRUE(last[0].img.get() != nullptr);
  auto first_again = ds->get_image_data(ts.front());
  ASSERT_TRUE(first_again[0].img.get() != nullptr);
}

TEST(DatasetIoMp4, MultiCamFromDirectory) {
  TmpDir tmp;
  const int W = 80, H = 48;
  const std::vector<std::pair<int, double>> cams = {
      {18, 30.0}, {12, 24.0}, {24, 60.0}};  // (frame_count, fps) per cam

  for (size_t i = 0; i < cams.size(); i++) {
    const std::string p = tmp.path() + "/cam" + std::to_string(i) + ".mp4";
    ASSERT_TRUE(write_synthetic_mp4(p, W, H, cams[i].first, cams[i].second))
        << "VideoWriter failed for cam" << i;
  }

  auto io = basalt::DatasetIoFactory::getDatasetIo("mp4");
  io->read(tmp.path());
  auto ds = io->get_data();
  ASSERT_TRUE(ds.get() != nullptr);
  EXPECT_EQ(ds->get_num_cams(), cams.size());

  // Canonical timestamps come from cam0.
  const auto& ts = ds->get_image_timestamps();
  EXPECT_EQ(static_cast<int>(ts.size()), cams[0].first);

  // First canonical timestamp resolves on every camera (frame 0 for all).
  auto imgs = ds->get_image_data(ts.front());
  ASSERT_EQ(imgs.size(), cams.size());
  for (size_t i = 0; i < cams.size(); i++) {
    ASSERT_TRUE(imgs[i].img.get() != nullptr) << "cam" << i << " missing";
    EXPECT_EQ(imgs[i].img->w, W);
    EXPECT_EQ(imgs[i].img->h, H);
  }

  // A timestamp strictly past cam1's last frame should still load cam0/cam2
  // but not cam1, since their fps differ.
  const int64_t last_cam1_ns = static_cast<int64_t>(
      std::llround((cams[1].first - 1) / cams[1].second * 1e9));
  // pick a cam0 timestamp that is past last_cam1_ns
  int64_t probe = -1;
  for (auto t : ts)
    if (t > last_cam1_ns) {
      probe = t;
      break;
    }
  ASSERT_GT(probe, 0);
  auto imgs2 = ds->get_image_data(probe);
  ASSERT_EQ(imgs2.size(), cams.size());
  EXPECT_TRUE(imgs2[0].img.get() != nullptr);
  EXPECT_TRUE(imgs2[1].img.get() == nullptr);
}

TEST(DatasetIoMp4, RealInstaCameraFile) {
  // Skipped unless the file exists locally. This verifies that real-world
  // MP4s (HEVC/h264) decode through our reader.
  const std::string path =
      "/home/ysimson/data/insta_calibration/front_back_dining_room_tv/"
      "071_back_calib_output/VID_20260504_113358_00_071_rear.mp4";
  if (!fs::exists(path)) GTEST_SKIP() << "real MP4 not present at " << path;

  auto io = basalt::DatasetIoFactory::getDatasetIo("mp4");
  io->read(path);
  auto ds = io->get_data();
  ASSERT_TRUE(ds.get() != nullptr);
  EXPECT_EQ(ds->get_num_cams(), 1u);
  ASSERT_GT(ds->get_image_timestamps().size(), 0u);

  auto imgs = ds->get_image_data(ds->get_image_timestamps().front());
  ASSERT_EQ(imgs.size(), 1u);
  ASSERT_TRUE(imgs[0].img.get() != nullptr);
  EXPECT_GT(imgs[0].img->w, 0);
  EXPECT_GT(imgs[0].img->h, 0);
}
