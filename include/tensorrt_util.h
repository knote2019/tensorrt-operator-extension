#pragma once

#include <algorithm>
#include <chrono>
#include <string>
#include <tuple>
#include <vector>

class TensorRTUtil {
 public:
    /*****************************************************************
     * print_dims
     * **************************************************************/
    static void print_dims(const std::string& dims_name, const nvinfer1::Dims& d) {
        std::cout << dims_name << ": " << dims_to_string(d) << std::endl;
    }

    /*****************************************************************
     * read_file
     * **************************************************************/
    template <typename T>
    static void read_file(const std::string& fileName, std::vector<T>* pVector) {
        std::unique_ptr<std::ifstream, std::function<void(std::ifstream*)>> file(
            new std::ifstream{fileName, std::ios::in | std::ios::binary}, [](std::ifstream* pf) {
                pf->close();
                delete pf;
            });
        if (file->is_open()) {
            file->seekg(0, std::ifstream::end);
            long size = file->tellg();
            file->seekg(0, std::ifstream::beg);
            // adjust vector.
            pVector->resize(size / sizeof(T));
            file->read(reinterpret_cast<char*>(pVector->data()), size);
        } else {
            std::stringstream ss;
            ss << "read file " << fileName << " failed." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

    /*****************************************************************
     * write_file
     * **************************************************************/
    template <typename T>
    static void write_file(const std::string& fileName, T* pData, size_t size) {
        std::unique_ptr<std::ofstream, std::function<void(std::ofstream*)>> file(
            new std::ofstream{fileName, std::ios::out | std::ios::binary | std::ios::trunc}, [](std::ofstream* pf) {
                pf->close();
                delete pf;
            });
        if (file->is_open()) {
            file->write(reinterpret_cast<char*>(pData), size * sizeof(T));
        } else {
            std::stringstream ss;
            ss << "write file " << fileName << " failed." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

    /*****************************************************************
     * dims_to_string
     * **************************************************************/
    static std::string dims_to_string(const nvinfer1::Dims& d) {
        std::stringstream ss;
        ss << "(nbDims = " << d.nbDims << ") ";
        for (int i = 0; i < d.nbDims; ++i) {
            ss << d.d[i];
            if (i < d.nbDims - 1) {
                ss << " x ";
            }
        }
        return ss.str();
    }

    /*****************************************************************
     * check_file_exist
     * **************************************************************/
    static bool check_file_exist(const std::string& file_path) {
        std::ifstream file1(file_path);
        return file1.good();
    }
};
