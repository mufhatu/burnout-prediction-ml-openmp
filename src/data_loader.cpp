#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "encoder.h"

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));

}

void loadCSV(

    const std::string& filename,
    std::vector<double>& x,
    std::vector<double>& y){

        std::ifstream file(filename);
        std::string line;

        //skip header
        std::getline(file,line);

        while (std::getline(file,line)){

            std::stringstream ss(line);
            std::string value;

            //stress level
            std::getline(ss,value,',');
            value = trim(value);
            if (value.empty()) continue;
            double stress = std::stod(value);

            //burnout label (TEXT → encode)
            std::getline(ss, value, ',');
            double label = encodeLabel(value);

            x.push_back(stress);
            y.push_back(label);         
        }
    }