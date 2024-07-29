#pragma once

#include <algorithm>
#include <chrono>
#include <string>
#include <tuple>
#include <vector>

class TensorRTUtil {
 public:
    /*****************************************************************
     * pre_process
     * **************************************************************/
    static cv::Mat pre_process(cv::Mat image, cv::Size size, cv::Scalar mean, cv::Scalar std) {
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, size, mean, true, false);
        cv::divide(blob, std, blob);
        return blob;
    }

    /*****************************************************************
     * calculate_dims_length
     * **************************************************************/
    static size_t calculate_dims_length(nvinfer1::Dims dims) {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<>());
    }

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
     * get_max_value_index
     * **************************************************************/
    template <typename T>
    static unsigned int get_max_value_index(std::vector<T>* v) {
        return std::distance(v->begin(), std::max_element(v->begin(), v->end()));
    }

    /*****************************************************************
     * get_imagenet_class_name
     * **************************************************************/
    static std::string get_imagenet_class_name(unsigned int class_id) {
        // json file.
        std::string imagenet_class_index_file = TensorRTUtil::root_path() + "include/imagenet/imagenet_class_names.json";
        std::ifstream imagenet_class_index_fs(imagenet_class_index_file, std::ios::binary);
        nlohmann::json imagenet_class_index_json;
        imagenet_class_index_fs >> imagenet_class_index_json;
        imagenet_class_index_fs.close();
        // parse class name.
        std::string class_name{};
        imagenet_class_index_json.at(std::to_string(class_id)).at(1).get_to(class_name);
        return class_name;
    }

    /*****************************************************************
     * put_text
     * **************************************************************/
    static void put_text(cv::Mat& image, const std::string& text) {
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 2;
        int thickness = 2;
        int baseline;
        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
        cv::Point origin;
        origin.x = image.cols / 2 - text_size.width / 2;
        origin.y = image.rows / 2 + text_size.height / 2;
        cv::putText(image, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
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
     * permutation_to_string
     * **************************************************************/
    static std::string permutation_to_string(const nvinfer1::Permutation& p) {
        std::stringstream ss;
        for (int i = 0; i < nvinfer1::Dims::MAX_DIMS; ++i) {
            ss << p.order[i];
            if (i < nvinfer1::Dims::MAX_DIMS - 1) {
                ss << " , ";
            }
        }
        return ss.str();
    }

    /*****************************************************************
     * root_path
     * **************************************************************/
    static std::string root_path() {
        std::string current_path = __FILE__;
        return current_path.substr(0, current_path.find("include")) + "/";
    }

    /*****************************************************************
     * check_file_exist
     * **************************************************************/
    static bool check_file_exist(const std::string& file_path) {
        std::ifstream file1(file_path);
        return file1.good();
    }
};
