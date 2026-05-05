#pragma once
#include <string>

/**
 * @brief Encodes a burnout label string into a numeric value.
 *
 * @param label Burnout level as a string ("Low", "Medium", "High").
 * @return 0.0 for Low, 1.0 for Medium, 2.0 for High, 0.0 for unknown.
 */
inline double encodeLabel(const std::string& label){
    if (label == "Low") return 0.0;
    if (label == "Medium") return 1.0;
    if (label == "High") return 2.0;
    return 0.0;
    
}
