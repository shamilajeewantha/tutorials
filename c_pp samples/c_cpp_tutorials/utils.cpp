// utils.cpp
#include <iostream>
#include "main.h"
#include <cpr/cpr.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "base64.h"
#include <vector>

using namespace cv;


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


void send_dash()
{
    std::string image_path = samples::findFile("frame001.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }


    std::vector<std::string> detection_cls = {"HD", "VD", "FAD", "LTD"};
    std::vector<float> detection_conf = {0.92f, 0.85f, 0.95f, 0.73f};
    std::vector<std::vector<int>> detection_bbox = {
        {100, 150, 200, 300},
        {250, 100, 350, 250},
        {400, 450, 200, 300},
        {650, 700, 350, 250}
    };
    std::string detection_type = "TD";







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

