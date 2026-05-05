#pragma once
#include <string>
#include <vector>

/**
 * @brief Loads a CSV dataset and splits it into features and labels.
 *
 * @param filename Path to the CSV file.
 * @param X Output feature matrix (rows = samples, cols = features).
 * @param y Output label vector (burnout level per sample).
 */
void loadCSV(const std::string& filename,
             std::vector<std::vector<double>>& X,
             std::vector<double>& y);