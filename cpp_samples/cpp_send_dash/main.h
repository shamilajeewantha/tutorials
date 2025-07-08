#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

// Functions callable from C
void say_hello_from_cpp(const char* msg);
void do_get_request();
void do_post_request();
void opencv_image_encode();
void send_dash();

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Pure C++ functions only visible to C++ compilation units

#include <string>
#include <vector>
#include <tuple>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::tuple<float, float, float, float> convert2relative(
    const cv::Mat& image,
    const std::vector<int>& bbox
);

json items_preparation(
    const cv::Mat& image,
    float confidence,
    const std::vector<int>& detection_bbox,
    const std::string& detection_cls
);

json create_detection_list(
    const cv::Mat& image,
    const std::vector<std::string>& detection_cls,
    const std::vector<float>& detection_conf,
    const std::vector<std::vector<int>>& detection_bbox,
    const std::string& detection_type
);

std::string binary_image_to_base64(const cv::Mat& binary_image);

std::pair<std::string, int> get_token();

std::tuple<std::string, std::string, int> send_update(
    const json& detection_list,
    int unit_id,
    const std::string& unit_name,
    const cv::Mat& image,
    const std::string& camera
);

#endif // __cplusplus

#endif // MAIN_H
