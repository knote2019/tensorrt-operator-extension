#pragma once

// std.
#include <array>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <csignal>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <mutex>
#include <random>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

// gtest.
#include <gtest/gtest.h>

// cuda.
#include "cuda/cuda_util.h"

// tensorrt.
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

// opencv.
#include <opencv2/opencv.hpp>

// headers.
#include "json.hpp"
#include "tensorrt_engine.h"
#include "tensorrt_logger.h"
#include "tensorrt_util.h"
