#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sklearn_cpp {
namespace utils {

// CSV loader used by all three main programs
// Convention: every column except the last is a feature, the last column is the label
class CSVReader {
public:
    // Load a CSV file into a feature matrix X and label vector y
    // Set has_header to true if the first row contains column names
    static void load(const std::string& filepath,
                     bool has_header,
                     std::vector<std::vector<double>>& X,
                     std::vector<double>& y) {
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("CSVReader::load -- cannot open file: " + filepath);

        X.clear();
        y.clear();

        std::string line;
        bool first_line = true;

        while (std::getline(file, line)) {
            // Skip the header row if requested
            if (first_line && has_header) {
                first_line = false;
                continue;
            }
            first_line = false;

            // Skip blank lines
            if (line.empty()) continue;

            // Parse one row of the CSV into doubles
            std::vector<double> row;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                row.push_back(std::stod(token));
            }
            if (row.empty()) continue;

            // Last column is the label, everything before it is a feature
            y.push_back(row.back());
            row.pop_back();
            X.push_back(row);
        }
    }
};

}
}