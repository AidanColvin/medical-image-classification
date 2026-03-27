#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./validator <filename>" << std::endl;
        return 1;
    }
    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Error: File not found." << std::endl;
        return 1;
    }
    std::string line;
    int count = 0;
    std::getline(file, line); // Skip header
    while (std::getline(file, line)) {
        if (!line.empty()) count++;
    }
    std::cout << "\n[C++ VALIDATOR] Report for: " << argv[1] << std::endl;
    if (count == 624) {
        std::cout << "[SUCCESS] 624 rows verified. Matches template." << std::endl;
        return 0;
    } else {
        std::cerr << "[FAILURE] Expected 624 rows, found " << count << std::endl;
        return 1;
    }
}
