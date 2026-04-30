#include <iostream>
#include <set>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <basalt/utils/apriltag.h>
#include <basalt/image/image.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image> [num_tags=36]\n";
    return 1;
  }

  std::string img_path = argv[1];
  int num_tags = (argc >= 3) ? std::stoi(argv[2]) : 36;

  cv::Mat img_bgr = cv::imread(img_path);
  if (img_bgr.empty()) {
    std::cerr << "Failed to load: " << img_path << "\n";
    return 1;
  }

  cv::Mat gray;
  cv::cvtColor(img_bgr, gray, cv::COLOR_BGR2GRAY);

  // basalt detector expects uint16
  cv::Mat gray16;
  gray.convertTo(gray16, CV_16U, 256.0);

  basalt::ManagedImage<uint16_t> img_basalt(gray16.cols, gray16.rows);
  for (int r = 0; r < gray16.rows; r++)
    memcpy(img_basalt.RowPtr(r), gray16.ptr<uint16_t>(r),
           gray16.cols * sizeof(uint16_t));

  basalt::ApriltagDetector detector(num_tags);

  Eigen::aligned_vector<Eigen::Vector2d> corners, corners_rejected;
  std::vector<int> ids, ids_rejected;
  std::vector<double> radii, radii_rejected;

  detector.detectTags(img_basalt, corners, ids, radii,
                      corners_rejected, ids_rejected, radii_rejected);

  std::cout << "Image: " << img_path << " (" << gray.cols << "x" << gray.rows << ")\n";
  std::cout << "Detected: " << ids.size() / 4 << " tags (" << ids.size()
            << " corners)\n";
  std::cout << "Rejected: " << ids_rejected.size() / 4 << " tags\n\n";

  // group corners by tag id (each tag has 4 corners, ids are tag_id*4+corner)
  std::set<int> tag_ids;
  for (int id : ids) tag_ids.insert(id / 4);

  std::cout << "Tag IDs found: ";
  for (int id : tag_ids) std::cout << id << " ";
  std::cout << "\n\n";

  // draw and save debug image
  cv::Mat out = img_bgr.clone();
  for (size_t i = 0; i < ids.size(); i += 4) {
    int tag_id = ids[i] / 4;
    std::vector<cv::Point2f> pts(4);
    for (int c = 0; c < 4; c++)
      pts[c] = cv::Point2f(corners[i + c][0], corners[i + c][1]);

    for (int c = 0; c < 4; c++)
      cv::line(out, pts[c], pts[(c + 1) % 4], cv::Scalar(0, 220, 0), 2);

    cv::Point2f center(0, 0);
    for (auto& p : pts) center += p;
    center *= 0.25f;
    cv::putText(out, std::to_string(tag_id), center - cv::Point2f(12, -5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 220, 0), 1);
  }

  std::string out_path = img_path.substr(0, img_path.rfind('.')) + "_basalt_detection.jpg";
  cv::imwrite(out_path, out);
  std::cout << "Annotated image -> " << out_path << "\n";

  return 0;
}
