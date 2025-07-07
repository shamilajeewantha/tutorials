// utils.cpp
#include <iostream>
#include "main.h"
#include <cpr/cpr.h>

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