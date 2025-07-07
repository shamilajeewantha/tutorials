#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::ifstream f("example.json");
    if (!f) {
        std::cerr << "Failed to open example.json" << std::endl;
        return 1;
    }

    json data = json::parse(f);

    std::cout << "Name: " << data["name"] << std::endl;
    std::cout << "Age: " << data["age"] << std::endl;
    std::cout << "Skills:" << std::endl;

    for (const auto& skill : data["skills"]) {
        std::cout << "  - " << skill << std::endl;
    }

    return 0;
}
