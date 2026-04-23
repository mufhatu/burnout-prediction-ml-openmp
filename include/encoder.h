#pragma once
#include <string>

inline double encodeLabel(const std::string& label){
    if (label == "Low") return 0.0;
    if (label == "Medium") return 1.0;
    if (label == "High") return 2.0;
    return 0.0;
    
}
