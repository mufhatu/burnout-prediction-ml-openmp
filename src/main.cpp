#include <iostream>
#include <vector>
#include "sequential_trainer.h"

class sequential_trainer {
public:
    void train(const std::vector<double>& x,
               const std::vector<double>& y,
               double& w, double& b) {
        if (x.empty()) {
            w = 0;
            b = 0;
            return;
        }

        double sumx = 0;
        double sumy = 0;
        double sumxy = 0;
        double sumxx = 0;
        int n = static_cast<int>(x.size());

        for (int i = 0; i < n; ++i) {
            sumx += x[i];
            sumy += y[i];
            sumxy += x[i] * y[i];
            sumxx += x[i] * x[i];
        }

        double denom = n * sumxx - sumx * sumx;
        if (denom == 0) {
            w = 0;
            b = sumy / n;
        } else {
            w = (n * sumxy - sumx * sumy) / denom;
            b = (sumy - w * sumx) / n;
        }
    }
};

void loadCSV(const std::string& filename,
             std::vector<double>& x,
             std::vector<double>& y);

std::string decodeLabel(double value) {
    if (value < 0.5) return "Low";
    else if (value < 1.5) return "Medium";
    else return "High";
}

int main() {
    
    std::vector<double>x,y;
    
    loadCSV("../data/developer_burnout_dataset.csv", x, y);

    double w = 0;
    double b = 0;

    sequential_trainer trainer;
    trainer.train(x, y, w, b);

    std::cout << "Training complete\n";
    std::cout << "w = " << w << "\n";
    std::cout << "b = " << b << "\n\n";

    std::cout << "Prediction:\n";

    for (int i = 0; i < 5 && i < x.size(); i++) {

        double raw = w * x[i] + b;

        std::cout << "Stress: " << x[i]
                  << " | Actual: " << y[i]
                  << " | Predicted: " << decodeLabel(raw)
                  << "\n";
    }
    return 0;
}