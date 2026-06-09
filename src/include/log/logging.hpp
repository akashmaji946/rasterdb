/*
 * Copyright 2026, RasterDB Contributors.
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

#ifndef RASTERDB_LOG_LOGGING_HPP
#define RASTERDB_LOG_LOGGING_HPP

#include "fmt/format.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#ifndef RASTERDB_DEFAULT_LOG_LEVEL
#define RASTERDB_DEFAULT_LOG_LEVEL "info"
#endif

#ifndef RASTERDB_DEFAULT_LOG_DIR
#define RASTERDB_DEFAULT_LOG_DIR "."
#endif

namespace duckdb {
namespace logging {

enum class level_enum : int {
  trace = 0,
  debug = 1,
  info = 2,
  warn = 3,
  err = 4,
  critical = 5,
  off = 6
};

inline constexpr int SIRIUS_LOG_FLUSH_SEC            = 3;
inline constexpr const char *SIRIUS_LOG_LEVEL_ENV    = "SIRIUS_LOG_LEVEL";
inline constexpr const char *RASTERDB_LOG_LEVEL_ENV  = "RASTERDB_LOG_LEVEL";
inline constexpr const char *SIRIUS_LOG_DIR_ENV      = "SIRIUS_LOG_DIR";
inline constexpr const char *RASTERDB_LOG_DIR_ENV    = "RASTERDB_LOG_DIR";
inline constexpr const char *ANSI_RESET              = "\033[0m";
inline constexpr const char *ANSI_TRACE              = "\033[36m";
inline constexpr const char *ANSI_DEBUG              = "\033[34m";
inline constexpr const char *ANSI_INFO               = "\033[33m";
inline constexpr const char *ANSI_WARN               = "\033[38;5;208m";
inline constexpr const char *ANSI_ERROR              = "\033[31m";
inline constexpr const char *ANSI_CRITICAL           = "\033[1;31m";

struct logger_state {
  level_enum level = level_enum::info;
  bool use_stderr = true;
  bool use_color = false;
  std::string log_file_path;
  std::FILE *file = nullptr;
  std::mutex mutex;

  ~logger_state() {
    if (file && file != stderr) {
      std::fclose(file);
    }
  }
};

inline logger_state &global_logger() {
  static logger_state state;
  return state;
}

inline std::optional<std::string> GetEnvVar(const std::string &name) {
  const char *val = std::getenv(name.c_str());
  if (val) {
    return std::string(val);
  }
  return std::nullopt;
}

inline level_enum ParseLogLevel(const std::string &s) {
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "trace") {
    return level_enum::trace;
  }
  if (lower == "debug") {
    return level_enum::debug;
  }
  if (lower == "info") {
    return level_enum::info;
  }
  if (lower == "warn" || lower == "warning") {
    return level_enum::warn;
  }
  if (lower == "error" || lower == "err") {
    return level_enum::err;
  }
  if (lower == "critical" || lower == "fatal") {
    return level_enum::critical;
  }
  if (lower == "off" || lower == "none") {
    return level_enum::off;
  }
  return level_enum::info;
}

inline level_enum GetLogLevel() {
  auto rdb_level = GetEnvVar(RASTERDB_LOG_LEVEL_ENV);
  if (rdb_level.has_value()) {
    return ParseLogLevel(*rdb_level);
  }
  auto sir_level = GetEnvVar(SIRIUS_LOG_LEVEL_ENV);
  if (sir_level.has_value()) {
    return ParseLogLevel(*sir_level);
  }
  return ParseLogLevel(RASTERDB_DEFAULT_LOG_LEVEL);
}

inline std::string GetLogDir() {
  auto rasterdb_log_dir_str = GetEnvVar(RASTERDB_LOG_DIR_ENV);
  if (rasterdb_log_dir_str.has_value()) {
    return *rasterdb_log_dir_str;
  }
  auto log_dir_str = GetEnvVar(SIRIUS_LOG_DIR_ENV);
  if (log_dir_str.has_value()) {
    return *log_dir_str;
  }
  return RASTERDB_DEFAULT_LOG_DIR;
}

inline const char *level_name(level_enum level) {
  switch (level) {
  case level_enum::trace:
    return "trace";
  case level_enum::debug:
    return "debug";
  case level_enum::info:
    return "info";
  case level_enum::warn:
    return "warn";
  case level_enum::err:
    return "error";
  case level_enum::critical:
    return "critical";
  case level_enum::off:
    return "off";
  }
  return "info";
}

inline const char *level_color(level_enum level) {
  switch (level) {
  case level_enum::trace:
    return ANSI_TRACE;
  case level_enum::debug:
    return ANSI_DEBUG;
  case level_enum::info:
    return ANSI_INFO;
  case level_enum::warn:
    return ANSI_WARN;
  case level_enum::err:
    return ANSI_ERROR;
  case level_enum::critical:
    return ANSI_CRITICAL;
  case level_enum::off:
    return "";
  }
  return "";
}

inline bool RasterDBShouldLog(level_enum level) {
  auto &state = global_logger();
  return state.level != level_enum::off && static_cast<int>(level) >= static_cast<int>(state.level);
}

inline std::string time_string(bool include_date) {
  auto now = std::chrono::system_clock::now();
  auto secs = std::chrono::system_clock::to_time_t(now);
  std::tm tm_value{};
#if defined(_WIN32)
  localtime_s(&tm_value, &secs);
#else
  localtime_r(&secs, &tm_value);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), include_date ? "%Y-%m-%d %H:%M:%S" : "%H:%M:%S", &tm_value);
  return std::string(buffer);
}

inline void close_file_if_needed(logger_state &state) {
  if (state.file && state.file != stderr) {
    std::fclose(state.file);
  }
  state.file = nullptr;
}

inline std::FILE *open_log_file(logger_state &state) {
  if (state.use_stderr) {
    return stderr;
  }
  if (state.file) {
    return state.file;
  }
  if (!state.log_file_path.empty()) {
    std::filesystem::path path(state.log_file_path);
    auto parent = path.parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
    }
    state.file = std::fopen(state.log_file_path.c_str(), "a");
  }
  if (!state.file) {
    state.use_stderr = true;
    state.use_color = true;
    return stderr;
  }
  return state.file;
}

inline void write_log_line(level_enum level, const std::string &message) {
  auto &state = global_logger();
  if (!RasterDBShouldLog(level)) {
    return;
  }

  std::lock_guard<std::mutex> guard(state.mutex);
  std::FILE *out = open_log_file(state);
  if (state.use_stderr) {
    if (state.use_color) {
      std::fprintf(out, "[%s] [%s%s%s] %s\n",
                   time_string(false).c_str(),
                   level_color(level),
                   level_name(level),
                   ANSI_RESET,
                   message.c_str());
    } else {
      std::fprintf(out, "[%s] [%s] %s\n",
                   time_string(false).c_str(),
                   level_name(level),
                   message.c_str());
    }
    std::fflush(out);
    return;
  }

  std::fprintf(out, "[%s] [%s] %s\n",
               time_string(true).c_str(),
               level_name(level),
               message.c_str());
  std::fflush(out);
}

template <typename Format, typename... Args>
inline void log_message(level_enum level, const Format &format, Args&&... args) {
  if (!RasterDBShouldLog(level)) {
    return;
  }
  write_log_line(level, duckdb_fmt::format(format, std::forward<Args>(args)...));
}

inline void InitGlobalLogger(std::string log_file = "") {
  auto &state = global_logger();
  std::lock_guard<std::mutex> guard(state.mutex);
  close_file_if_needed(state);
  if (log_file.empty()) {
    log_file = GetLogDir() + "/sirius.log";
  }
  state.level = GetLogLevel();
  state.use_stderr = false;
  state.use_color = false;
  state.log_file_path = std::move(log_file);
}

inline void InitGPULogger() {
  auto &state = global_logger();
  std::lock_guard<std::mutex> guard(state.mutex);
  close_file_if_needed(state);
  state.level = GetLogLevel();
  state.use_stderr = true;
  state.use_color = true;
  state.log_file_path.clear();
  state.file = stderr;
}

} // namespace logging

inline bool RasterDBShouldLog(logging::level_enum level) {
  return logging::RasterDBShouldLog(level);
}

inline void InitGlobalLogger(std::string log_file = "") {
  logging::InitGlobalLogger(std::move(log_file));
}

inline void InitGPULogger() {
  logging::InitGPULogger();
}

} // namespace duckdb

namespace spdlog {

namespace level {
using level_enum = duckdb::logging::level_enum;
inline constexpr level_enum trace = level_enum::trace;
inline constexpr level_enum debug = level_enum::debug;
inline constexpr level_enum info = level_enum::info;
inline constexpr level_enum warn = level_enum::warn;
inline constexpr level_enum err = level_enum::err;
inline constexpr level_enum critical = level_enum::critical;
inline constexpr level_enum off = level_enum::off;
} // namespace level

template <typename Format, typename... Args>
inline void info(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::info, format, std::forward<Args>(args)...);
}

template <typename Format, typename... Args>
inline void warn(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::warn, format, std::forward<Args>(args)...);
}

template <typename Format, typename... Args>
inline void error(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::err, format, std::forward<Args>(args)...);
}

template <typename Format, typename... Args>
inline void critical(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::critical, format, std::forward<Args>(args)...);
}

template <typename Format, typename... Args>
inline void debug(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::debug, format, std::forward<Args>(args)...);
}

template <typename Format, typename... Args>
inline void trace(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::trace, format, std::forward<Args>(args)...);
}

} // namespace spdlog

#define RASTERDB_LOG_TRACE(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::trace, __VA_ARGS__)
#define RASTERDB_LOG_DEBUG(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::debug, __VA_ARGS__)
#define RASTERDB_LOG_INFO(...)  ::duckdb::logging::log_message(::duckdb::logging::level_enum::info, __VA_ARGS__)
#define RASTERDB_LOG_WARN(...)  ::duckdb::logging::log_message(::duckdb::logging::level_enum::warn, __VA_ARGS__)
#define RASTERDB_LOG_ERROR(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::err, __VA_ARGS__)
#define RASTERDB_LOG_FATAL(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::critical, __VA_ARGS__)
#define SIRIUS_LOG_FATAL(...)   ::duckdb::logging::log_message(::duckdb::logging::level_enum::critical, __VA_ARGS__)

#endif // RASTERDB_LOG_LOGGING_HPP
