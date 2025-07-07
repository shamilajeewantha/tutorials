
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "main.h"
#include <iostream>

using namespace cv;

int main()
{
    std::string image_path = samples::findFile("frame001.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
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

    return 0;
}