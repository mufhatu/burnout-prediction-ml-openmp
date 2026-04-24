#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include "encoder.h"
#include "data_loader.h"

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \r\n\t");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \r\n\t");
    return str.substr(first, (last - first + 1));
}

void loadCSV(

    const std::string& filename,
    std::vector<std::vector<double>>& X,
    std::vector<double>& y){

        std::ifstream file(filename);

        if (!file.is_open()) {
        std::cerr << "Error: could not open " << filename << "\n";
        return;
        }

        std::string line;

        //skip header
        std::getline(file,line);
        
        int lineNum = 1;

        while (std::getline(file,line)){
            lineNum++;

            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string val;
            std::vector<double> row;
            bool bad = false;

            // Read 10 numeric features:
            // age, experience_years, daily_work_hours, sleep_hours,
            // caffeine_intake, bugs_per_day, commits_per_day,
            // meetings_per_day, screen_time, exercise_hours
            for (int i = 0; i < 10; i++) {
                if (!std::getline(ss, val, ',')) { bad = true; break; }
                val = trim(val);
                try {
                    row.push_back(std::stod(val));
                } catch (...) {
                    bad = true; break;
                }
            }   
        

            if (bad) continue;  // ← add this line right here

            // stress_level (col 11)
            if (!std::getline(ss, val, ',')) continue;
            val = trim(val);
            try {
                row.push_back(std::stod(val));
            } catch (...) { continue; }

            // burnout_level (col 12)
            if (!std::getline(ss, val, ',')) continue;
            double label = encodeLabel(trim(val));

            X.push_back(row);
            y.push_back(label);
    
    }    
    

    std::cout << "Loaded " << X.size() << " rows, "
              << (X.empty() ? 0 : X[0].size()) << " features.\n";

}