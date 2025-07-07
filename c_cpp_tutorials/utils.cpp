// utils.cpp
#include <iostream>
#include "main.h"
#include <cpr/cpr.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "base64.h"

void say_hello_from_cpp() {
    std::cout << "Hello from C++!" << std::endl;
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

using namespace cv;

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