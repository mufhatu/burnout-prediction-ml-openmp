#pragma once
#include <string>
#include <vector>

void loadCSV(const std::string& filename,
             std::vector<std::vector<double>>& X,
             std::vector<double>& y);