// utils.cpp
#include <iostream>
#include "main.h"
#include <cpr/cpr.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "base64.h"
#include <vector>
#include <spdlog/spdlog.h> 
#include <chrono>
#include <ctime>

using namespace cv;
using json = nlohmann::json;
using namespace std;

void say_hello_from_cpp(const char* msg) {
    std::cout << "Message from C++: " << msg << std::endl;
}

void do_get_request() {
    auto r = cpr::Get(cpr::Url{"https://httpbin.org/get"});
    std::cout << "Status code: " << r.status_code << std::endl;
    std::cout << r.text << std::endl;
}

void do_post_request() {
    cpr::Response r = cpr::Post(
        cpr::Url{"http://localhost:5000/receive"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{R"({"name": "Shamila", "message": "Hello from C++!"})"}
    );

    std::cout << "Status code: " << r.status_code << std::endl;
    std::cout << r.text << std::endl;
}



std::tuple<std::string, std::string, int> send_update(
    const json& detection_list,  // JSON array or object of detections
    int unit_id,
    const std::string& unit_name,
    const cv::Mat& image,
    const std::string& camera
) {
    // Encode image to base64 string
    std::string base64_string = binary_image_to_base64(image);

    // Get token and expiry
    auto [token, expires_in] = get_token();
    if (token.empty()) {
        std::cerr << "Failed to retrieve authentication token. Aborting request." << std::endl;
        return {"", "", 0}; // failure: empty payload and response, status 0
    }

    // Current timestamp minus 14400 seconds (4 hours)
    int64_t timestamp = std::time(nullptr) - 14400;

    // Build JSON payload
    json payload = {
        {"Robot", {
            {"Id", unit_id},
            {"Name", unit_name},
            {"Camera", camera},
            {"Image", base64_string}
        }},
        {"TimeStamp", timestamp},
        {"Type", "TD"},
        {"DETECTIONS", json::array({detection_list})}  // wrap detection_list in an array
    };

    // Build payload2 as in Python (with Image = "image" string)
    json payload2 = {
        {"Robot", {
            {"Id", unit_id},
            {"Name", unit_name},
            {"Camera", camera},
            {"Image", "image"}
        }},
        {"TimeStamp", timestamp},
        {"Type", "TD"},
        {"DETECTIONS", json::array({detection_list})}
    };

    std::string url = "https://rad-staging.azurewebsites.net//umbraco/api/RobotDetails/PostRobotCameraDetails";
    cpr::Header headers = {
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + token}
    };

    cpr::Response r = cpr::Post(cpr::Url{url}, headers, cpr::Body{payload.dump()});

    return {payload2.dump(), r.text, r.status_code};
}



std::pair<std::string, int> get_token() {
    std::string url = "https://rad-identity.azurewebsites.net/connect/token";
    std::string payload = "client_id=100004&client_secret=RAD.SOC.Api-Secret-5&grant_type=client_credentials&scopes=RAD.SOC.Api";
    cpr::Header headers{{"Content-Type", "application/x-www-form-urlencoded"}};

    cpr::Response r = cpr::Post(cpr::Url{url}, headers, cpr::Body{payload});

    if (r.status_code == 200) {
        try {
            auto response_data = json::parse(r.text);

            if (response_data.contains("access_token") && response_data.contains("expires_in")) {
                std::string access_token = response_data["access_token"];
                int expires_in = response_data["expires_in"];
                return {access_token, expires_in};
            }
        } catch (const std::exception& e) {
            spdlog::error("JSON parse error: {}", e.what());
        }
    }

    spdlog::error("Failed to get token: {}", r.text);
    return {"", 0};  // Return empty string and 0 if failed
}


void opencv_image_encode()
{
    std::string image_path = samples::findFile("frame001.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }

    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);
    auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
    std::string encoded = base64_encode(enc_msg, buf.size());
    std::cout << "image encode success";

    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window

    if(k == 's')
    {
        imwrite("frame001.png", img);
    }

}







// Keep the signature: returns tuple of floats
std::tuple<float, float, float, float> convert2relative(
    const cv::Mat& image,
    const std::vector<int>& bbox
) {
    int x_min = bbox[0];
    int y_min = bbox[1];
    int x_max = bbox[2];
    int y_max = bbox[3];

    int height = image.rows;
    int width = image.cols;

    float cx = static_cast<float>(x_max + x_min) / (2.0f * width);
    float cy = static_cast<float>(y_max + y_min) / (2.0f * height);
    float w  = static_cast<float>(x_max - x_min) / width;
    float h  = static_cast<float>(y_max - y_min) / height;

    return std::make_tuple(cx, cy, w, h);
}

// Keep the signature and return a JSON object
json items_preparation(
    const cv::Mat& image,
    float confidence,
    const std::vector<int>& detection_bbox,
    const std::string& detection_cls
) {
    auto [x, y, w, h] = convert2relative(image, detection_bbox);

    auto round4 = [](float val) {
        return std::round(val * 10000.0f) / 10000.0f;
    };

    json items = {
        {"CX", round4(x)},
        {"CY", round4(y)},
        {"Width", round4(w)},
        {"Height", round4(h)},
        {"Confidence", confidence},
        {"Type", detection_cls},
        {"DetectionNumber", 0}
    };

    return items;
}




json create_detection_list(
    const cv::Mat& image,
    const std::vector<std::string>& detection_cls,
    const std::vector<float>& detection_conf,
    const std::vector<std::vector<int>>& detection_bbox,
    const std::string& detection_type
) {
    std::vector<json> detection_list;

    for (size_t i = 0; i < detection_cls.size(); ++i) {
        float confidence = detection_conf[i];
        if (detection_type == "TD") {
            json item = items_preparation(image, confidence, detection_bbox[i], detection_cls[i]);
            detection_list.push_back(item);
        }
    }

    json json_string = {
        {"ITEMS", detection_list}
    };

    return json_string;
}



std::string binary_image_to_base64(const cv::Mat& binary_image) {
    std::vector<uchar> buf;
    // Encode the image to JPEG format in memory buffer
    cv::imencode(".jpg", binary_image, buf);

    // Cast buffer data to unsigned char pointer
    auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());

    // Call your base64 encode function
    std::string base64_encoded = base64_encode(enc_msg, buf.size());

    return base64_encoded;
}




void send_dash()
{
    std::string image_path = samples::findFile("frame001.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }


    std::vector<std::string> detection_cls = {"HD", "HD", "FAD", "LTD"};
    std::vector<float> detection_conf = {0.92f, 0.85f, 0.95f, 0.73f};
    std::vector<std::vector<int>> detection_bbox = {
        {100, 150, 200, 300},
        {250, 100, 350, 250},
        {400, 450, 200, 300},
        {650, 700, 350, 250}
    };
    std::string detection_type = "TD";

    int unit_id = 1798;
    std::string unit_name = "test_perception";
    std::string camera = "front";

    json detection_list = create_detection_list(img, detection_cls, detection_conf, detection_bbox, detection_type);

    auto [payload2, response_text, status_code] = send_update(detection_list, unit_id, unit_name, img, camera);


}

