#pragma once

/*
 * csv_reader.hpp
 * ==============
 * Lightweight CSV loader for sklearn_cpp.  Header-only -- just include and use.
 *
 * Convention used across the entire library:
 *   * Every column except the last is a feature  -> stored in X
 *   * The last column is the label               -> stored in y
 *
 * Per-dataset hints:
 *     concrete.csv     has_header = true   (first row is column names)
 *     ecg.csv          has_header = false
 *     mnist_micro.csv  has_header = false
 *
 * Example
 * -------
 *     #include "sklearn_cpp/utils/csv_reader.hpp"
 *
 *     std::vector<std::vector<double>> X;
 *     std::vector<double> y;
 *     sklearn_cpp::utils::CSVReader::load("data/concrete.csv", true, X, y);
 */

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sklearn_cpp {
namespace utils {

class CSVReader {
public:
    /**
     * Load a CSV file into a feature matrix X and label vector y.
     *
     * @param filepath    path to the .csv file
     * @param has_header  true  -> skip the first line (column names)
     *                    false -> treat every line as a data row
     * @param X           output feature matrix, shape [n_samples][n_features]
     * @param y           output label vector,   shape [n_samples]
     *
     * @throws std::runtime_error if the file cannot be opened.
     */
    static void load(const std::string&                filepath,
                     bool                              has_header,
                     std::vector<std::vector<double>>& X,
                     std::vector<double>&              y)
    {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error(
                "CSVReader::load -- cannot open file: " + filepath);
        }

        X.clear();
        y.clear();

        std::string line;
        bool        first_line = true;

        while (std::getline(file, line)) {

            // Skip the header row if requested
            if (first_line && has_header) { first_line = false; continue; }
            first_line = false;

            // Skip blank lines
            if (line.empty() ||
                line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

            std::vector<double> row = parse_line(line);
            if (row.empty()) continue;

            // Last column is the label; everything before it is a feature
            y.push_back(row.back());
            row.pop_back();
            X.push_back(std::move(row));
        }
    }

    /**
     * Convenience overload that returns (X, y) by value.
     * Useful with structured bindings:
     *     auto [X, y] = CSVReader::load("file.csv", false);
     */
    static std::pair<std::vector<std::vector<double>>, std::vector<double>>
    load(const std::string& filepath, bool has_header)
    {
        std::vector<std::vector<double>> X;
        std::vector<double>              y;
        load(filepath, has_header, X, y);
        return {std::move(X), std::move(y)};
    }

private:
    // Split a comma-separated line into doubles; trims whitespace per token.
    static std::vector<double> parse_line(const std::string& line)
    {
        std::vector<double> values;
        std::stringstream   ss(line);
        std::string         token;

        while (std::getline(ss, token, ',')) {
            const auto a = token.find_first_not_of(" \t\r\n");
            const auto b = token.find_last_not_of (" \t\r\n");
            if (a == std::string::npos) continue;

            token = token.substr(a, b - a + 1);
            values.push_back(std::stod(token));
        }
        return values;
    }
};

} // namespace utils
} // namespace sklearn_cpp
