/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef DATASET_IO_MP4_H
#define DATASET_IO_MP4_H

#include <basalt/io/dataset_io.h>
#include <basalt/utils/filesystem.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <regex>

namespace basalt {

// Dataset layout supported:
//   <path>.<ext>            single mono video
//   <path>/cam0.<ext>       mono (or first camera of a multi-cam rig)
//   <path>/cam1.<ext>       additional cameras (cam0..camN, contiguous indices)
//   <path>/imu.csv          optional EuRoC-format IMU: ts_ns,wx,wy,wz,ax,ay,az
//
// <ext> is any container FFmpeg can demux: mp4, mov, mkv, avi, webm, m4v.
// In single-file mode the extension is not checked (the path is handed to
// cv::VideoCapture as-is). In directory mode only the extensions above are
// auto-discovered.
//
// Per-camera FPS is read from the file (CAP_PROP_FPS); per-frame timestamps
// are computed independently for each camera. The number of fused image
// timestamps is min(frame_count_i) and we use cam0's per-frame timestamps as
// the canonical key. Each camera maintains its own t_ns -> frame_idx map so
// rigs with mismatched fps still resolve to a real frame on each cam.
class Mp4VioDataset : public VioDataset {
  size_t num_cams = 0;

  std::vector<std::string> video_paths;
  std::vector<double> fps;
  std::vector<int64_t> per_cam_frame_count;

  std::vector<int64_t> image_timestamps;  // canonical timestamps (cam0)
  std::vector<std::unordered_map<int64_t, int64_t>> ts_to_frame_idx;

  // VideoCapture is not thread-safe; readers are mutex-protected per cam.
  mutable std::vector<std::unique_ptr<std::mutex>> cap_mutex;
  mutable std::vector<cv::VideoCapture> caps;
  mutable std::vector<int64_t> last_frame_idx;  // -1 = unknown

  Eigen::aligned_vector<AccelData> accel_data;
  Eigen::aligned_vector<GyroData> gyro_data;

  std::vector<int64_t> gt_timestamps;
  Eigen::aligned_vector<Sophus::SE3d> gt_pose_data;

  int64_t mocap_to_imu_offset_ns = 0;

 public:
  ~Mp4VioDataset() {};

  size_t get_num_cams() const { return num_cams; }

  std::vector<int64_t>& get_image_timestamps() { return image_timestamps; }

  const Eigen::aligned_vector<AccelData>& get_accel_data() const {
    return accel_data;
  }
  const Eigen::aligned_vector<GyroData>& get_gyro_data() const {
    return gyro_data;
  }
  const std::vector<int64_t>& get_gt_timestamps() const {
    return gt_timestamps;
  }
  const Eigen::aligned_vector<Sophus::SE3d>& get_gt_pose_data() const {
    return gt_pose_data;
  }

  int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

  std::vector<ImageData> get_image_data(int64_t t_ns) {
    std::vector<ImageData> res(num_cams);

    for (size_t i = 0; i < num_cams; i++) {
      auto it = ts_to_frame_idx[i].find(t_ns);
      if (it == ts_to_frame_idx[i].end()) continue;

      const int64_t want_idx = it->second;

      std::lock_guard<std::mutex> lock(*cap_mutex[i]);

      cv::VideoCapture& cap = caps[i];
      if (!cap.isOpened()) continue;

      // Sequential next-frame is the fast path. Otherwise seek.
      if (last_frame_idx[i] < 0 || want_idx != last_frame_idx[i] + 1) {
        if (!cap.set(cv::CAP_PROP_POS_FRAMES,
                     static_cast<double>(want_idx))) {
          last_frame_idx[i] = -1;
          continue;
        }
      }

      cv::Mat frame;
      if (!cap.read(frame) || frame.empty()) {
        last_frame_idx[i] = -1;
        continue;
      }
      last_frame_idx[i] = want_idx;

      cv::Mat gray;
      if (frame.channels() == 1) {
        gray = frame;
      } else if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
      } else {
        continue;
      }

      res[i].img.reset(new ManagedImage<uint16_t>(gray.cols, gray.rows));
      const size_t full_size = static_cast<size_t>(gray.cols) * gray.rows;

      if (gray.depth() == CV_8U) {
        const uint8_t* in = gray.ptr<uint8_t>();
        uint16_t* out = res[i].img->ptr;
        for (size_t k = 0; k < full_size; k++) {
          out[k] = static_cast<uint16_t>(in[k]) << 8;
        }
      } else if (gray.depth() == CV_16U) {
        std::memcpy(res[i].img->ptr, gray.ptr(),
                    full_size * sizeof(uint16_t));
      } else {
        cv::Mat conv;
        gray.convertTo(conv, CV_8U);
        const uint8_t* in = conv.ptr<uint8_t>();
        uint16_t* out = res[i].img->ptr;
        for (size_t k = 0; k < full_size; k++) {
          out[k] = static_cast<uint16_t>(in[k]) << 8;
        }
      }
    }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class Mp4IO;
};

class Mp4IO : public DatasetIoInterface {
 public:
  Mp4IO() = default;

  void read(const std::string& path) {
    if (!fs::exists(path)) {
      std::cerr << "No mp4 dataset found at " << path << std::endl;
      std::abort();
    }

    data.reset(new Mp4VioDataset);

    std::vector<std::string> videos;
    if (fs::is_regular_file(path)) {
      videos.push_back(path);
    } else {
      videos = discover_cam_videos(path);
      if (videos.empty()) {
        std::cerr << "No camN.{mp4,mov,mkv,avi,webm,m4v} files found under "
                  << path << std::endl;
        std::abort();
      }
    }

    data->num_cams = videos.size();
    data->video_paths = videos;
    data->fps.resize(data->num_cams, 0.0);
    data->per_cam_frame_count.resize(data->num_cams, 0);
    data->ts_to_frame_idx.resize(data->num_cams);
    data->caps.resize(data->num_cams);
    data->last_frame_idx.assign(data->num_cams, -1);
    data->cap_mutex.clear();
    data->cap_mutex.reserve(data->num_cams);
    for (size_t i = 0; i < data->num_cams; i++) {
      data->cap_mutex.emplace_back(std::make_unique<std::mutex>());
    }

    for (size_t i = 0; i < data->num_cams; i++) {
      cv::VideoCapture& cap = data->caps[i];
      if (!cap.open(videos[i])) {
        std::cerr << "Failed to open " << videos[i] << std::endl;
        std::abort();
      }
      double f = cap.get(cv::CAP_PROP_FPS);
      if (!(f > 0.0)) {
        std::cerr << "Could not read FPS from " << videos[i]
                  << " (got " << f << ")" << std::endl;
        std::abort();
      }
      const int64_t n = static_cast<int64_t>(
          std::llround(cap.get(cv::CAP_PROP_FRAME_COUNT)));
      if (n <= 0) {
        std::cerr << "Could not read frame count from " << videos[i]
                  << std::endl;
        std::abort();
      }
      data->fps[i] = f;
      data->per_cam_frame_count[i] = n;

      data->ts_to_frame_idx[i].reserve(static_cast<size_t>(n));
      for (int64_t k = 0; k < n; k++) {
        const int64_t t_ns = frame_idx_to_ns(k, f);
        data->ts_to_frame_idx[i][t_ns] = k;
      }
    }

    // Canonical timestamps: cam0's full timeline.
    data->image_timestamps.clear();
    data->image_timestamps.reserve(
        static_cast<size_t>(data->per_cam_frame_count[0]));
    for (int64_t k = 0; k < data->per_cam_frame_count[0]; k++) {
      data->image_timestamps.push_back(frame_idx_to_ns(k, data->fps[0]));
    }

    // Optional IMU: <dataset_dir>/imu.csv (skipped if path is a file).
    if (fs::is_directory(path)) {
      const std::string imu_csv = path + "/imu.csv";
      if (fs::exists(imu_csv)) {
        read_imu_csv(imu_csv);
      }
    }
  }

  void reset() { data.reset(); }

  VioDatasetPtr get_data() { return data; }

 private:
  static int64_t frame_idx_to_ns(int64_t idx, double fps) {
    // Round to nearest ns for stable t_ns -> idx lookup.
    const double seconds = static_cast<double>(idx) / fps;
    return static_cast<int64_t>(std::llround(seconds * 1e9));
  }

  std::vector<std::string> discover_cam_videos(const std::string& dir) {
    std::vector<std::pair<int, std::string>> found;
    std::regex re(R"(^cam(\d+)\.(mp4|mov|mkv|avi|webm|m4v)$)",
                  std::regex::icase);
    for (const auto& entry : fs::directory_iterator(dir)) {
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      std::smatch m;
      if (std::regex_match(name, m, re)) {
        const int idx = std::stoi(m[1].str());
        found.emplace_back(idx, entry.path().string());
      }
    }
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Require contiguous indices starting at 0.
    std::vector<std::string> out;
    for (size_t i = 0; i < found.size(); i++) {
      if (found[i].first != static_cast<int>(i)) {
        std::cerr << "cam indices must be contiguous starting at 0; "
                  << "got cam" << found[i].first << " at position " << i
                  << std::endl;
        std::abort();
      }
      out.push_back(found[i].second);
    }
    return out;
  }

  void read_imu_csv(const std::string& csv_path) {
    data->accel_data.clear();
    data->gyro_data.clear();
    std::ifstream f(csv_path);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::stringstream ss(line);
      char tmp;
      uint64_t timestamp;
      Eigen::Vector3d gyro, accel;
      ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2] >>
          tmp >> accel[0] >> tmp >> accel[1] >> tmp >> accel[2];
      if (!ss) continue;

      AccelData a;
      a.timestamp_ns = static_cast<int64_t>(timestamp);
      a.data = accel;
      data->accel_data.push_back(a);

      GyroData g;
      g.timestamp_ns = static_cast<int64_t>(timestamp);
      g.data = gyro;
      data->gyro_data.push_back(g);
    }
  }

  std::shared_ptr<Mp4VioDataset> data;
};

}  // namespace basalt

#endif  // DATASET_IO_MP4_H
