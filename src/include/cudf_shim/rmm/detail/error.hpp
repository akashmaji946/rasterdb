/* RasterDB rmm compatibility shim */
#pragma once
#include "../../types.hpp"
#include "../../column.hpp"
#include <stdexcept>

#define RMM_EXPECTS(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)
#define RMM_CUDA_TRY(expr) (expr)
#define RMM_CUDA_TRY_ALLOC(expr) (expr)
#define CUDF_CUDA_TRY(expr) (expr)
#define CUDF_CHECK_CUDA(expr) (expr)
#define CUDF_EXPECTS(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)
