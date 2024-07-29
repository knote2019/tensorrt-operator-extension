#pragma once

#include <iostream>
#include <string>

#include "NvInfer.h"

class TensorRTLogger : public nvinfer1::ILogger {
 private:
    nvinfer1::ILogger::Severity mReportableSeverity;

 public:
    explicit TensorRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
        : mReportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity <= mReportableSeverity) {
            std::cout << severityPrefix(mReportableSeverity) << msg << std::endl;
        }
    }

 private:
    static std::string severityPrefix(nvinfer1::ILogger::Severity severity) {
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                return "[INTERNAL ERROR] ";
            case nvinfer1::ILogger::Severity::kERROR:
                return "[ERROR] ";
            case nvinfer1::ILogger::Severity::kWARNING:
                return "[WARNING] ";
            case nvinfer1::ILogger::Severity::kINFO:
                return "[INFO] ";
            case nvinfer1::ILogger::Severity::kVERBOSE:
                return "[VERBOSE] ";
            default:
                return "[NA]";
        }
    }
};
