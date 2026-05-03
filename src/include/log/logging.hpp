/*
 * Copyright 2025, RasterDB Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#else
#warning "SPDLOG_ACTIVE_LEVEL is overridden, output may be lost"
#endif

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/ansicolor_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>

#ifndef RASTERDB_DEFAULT_LOG_LEVEL
#define RASTERDB_DEFAULT_LOG_LEVEL "info"
#endif

#ifndef RASTERDB_DEFAULT_LOG_DIR
#define RASTERDB_DEFAULT_LOG_DIR "."
#endif

#define RASTERDB_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::default_logger_raw(), __VA_ARGS__)
#define RASTERDB_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::default_logger_raw(), __VA_ARGS__)
#define RASTERDB_LOG_INFO(...)  SPDLOG_LOGGER_INFO(spdlog::default_logger_raw(), __VA_ARGS__)
#define RASTERDB_LOG_WARN(...)  SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), __VA_ARGS__)
#define RASTERDB_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), __VA_ARGS__)
#define RASTERDB_LOG_FATAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::default_logger_raw(), __VA_ARGS__)
#define SIRIUS_LOG_FATAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::default_logger_raw(), __VA_ARGS__)

namespace duckdb {

inline constexpr int SIRIUS_LOG_FLUSH_SEC         = 3;
inline constexpr const char* SIRIUS_LOG_LEVEL_ENV  = "SIRIUS_LOG_LEVEL";
inline constexpr const char* RASTERDB_LOG_LEVEL_ENV = "RASTERDB_LOG_LEVEL";
inline constexpr const char* SIRIUS_LOG_DIR_ENV     = "SIRIUS_LOG_DIR";
inline constexpr const char* RASTERDB_LOG_DIR_ENV   = "RASTERDB_LOG_DIR";

inline std::optional<std::string> GetEnvVar(const std::string& name)
{
  const char* val = std::getenv(name.c_str());
  if (val) {
    return std::string(val);
  } else {
    return std::nullopt;
  }
}

inline spdlog::level::level_enum ParseLogLevel(const std::string& s)
{
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "trace") return spdlog::level::trace;
  if (lower == "debug") return spdlog::level::debug;
  if (lower == "info")  return spdlog::level::info;
  if (lower == "warn" || lower == "warning") return spdlog::level::warn;
  if (lower == "error" || lower == "err") return spdlog::level::err;
  if (lower == "critical" || lower == "fatal") return spdlog::level::critical;
  if (lower == "off" || lower == "none") return spdlog::level::off;
  return spdlog::level::info;
}

inline spdlog::level::level_enum GetLogLevel()
{
  // RASTERDB_LOG_LEVEL takes precedence over SIRIUS_LOG_LEVEL
  auto rdb_level = GetEnvVar(RASTERDB_LOG_LEVEL_ENV);
  if (rdb_level.has_value()) return ParseLogLevel(*rdb_level);
  auto sir_level = GetEnvVar(SIRIUS_LOG_LEVEL_ENV);
  if (sir_level.has_value()) return ParseLogLevel(*sir_level);
  return ParseLogLevel(RASTERDB_DEFAULT_LOG_LEVEL);
}

inline std::string GetLogDir()
{
  auto rasterdb_log_dir_str = GetEnvVar(RASTERDB_LOG_DIR_ENV);
  if (rasterdb_log_dir_str.has_value()) { return *rasterdb_log_dir_str; }
  auto log_dir_str = GetEnvVar(SIRIUS_LOG_DIR_ENV);
  if (log_dir_str.has_value()) { return *log_dir_str; }
  return RASTERDB_DEFAULT_LOG_DIR;
}

inline bool RasterDBShouldLog(spdlog::level::level_enum level)
{
  auto* logger = spdlog::default_logger_raw();
  return logger && logger->should_log(level);
}

inline void InitGlobalLogger(std::string log_file = "")
{
  // Log file
  if (log_file.empty()) {
    auto log_dir = GetLogDir();
    log_file     = log_dir + "/sirius.log";
  }
  auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false);
  file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [%s:%#] %v");

  // Logger
  auto logger    = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{file_sink});
  auto log_level = GetLogLevel();
  logger->set_level(log_level);
  spdlog::set_default_logger(logger);
  spdlog::set_level(log_level);  // Also set the global level
  auto rasterdb_log_level_str = GetEnvVar(RASTERDB_LOG_LEVEL_ENV);
  auto sirius_log_level_str = GetEnvVar(SIRIUS_LOG_LEVEL_ENV);
  if (rasterdb_log_level_str.has_value() || sirius_log_level_str.has_value()) {
    spdlog::flush_on(log_level);
  } else {
    spdlog::flush_every(std::chrono::seconds(SIRIUS_LOG_FLUSH_SEC));
  }
}

inline void InitGPULogger()
{
  // stderr color sink for RasterDB GPU (Vulkan/rasterdf) extension path.
  // Colors: DEBUG=Blue, INFO=Yellow, WARN=Orange, ERROR=Red
  auto stderr_sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
  stderr_sink->set_pattern("[%H:%M:%S] [%^%l%$] %v");
  stderr_sink->set_color(spdlog::level::trace,    stderr_sink->cyan);
  stderr_sink->set_color(spdlog::level::debug,    stderr_sink->blue);
  stderr_sink->set_color(spdlog::level::info,     stderr_sink->yellow);
  stderr_sink->set_color(spdlog::level::warn,     "\033[38;5;208m");  // Orange (256-color)
  stderr_sink->set_color(spdlog::level::err,      stderr_sink->red);
  stderr_sink->set_color(spdlog::level::critical, stderr_sink->red_bold);

  auto logger = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{stderr_sink});
  auto log_level = GetLogLevel();
  logger->set_level(log_level);
  logger->flush_on(spdlog::level::debug);
  spdlog::set_default_logger(logger);
  spdlog::set_level(log_level);
}

}  // namespace duckdb
