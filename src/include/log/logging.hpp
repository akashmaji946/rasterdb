/*
 * Copyright 2025, Sirius Contributors.
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

#ifndef SIRIUS_LOG_LOGGING_HPP
#define SIRIUS_LOG_LOGGING_HPP

#include "fmt/format.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#ifndef SIRIUS_DEFAULT_LOG_DIR
#define SIRIUS_DEFAULT_LOG_DIR "."
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

inline constexpr int SIRIUS_LOG_FLUSH_SEC         = 3;
inline constexpr const char *SIRIUS_LOG_LEVEL_ENV = "SIRIUS_LOG_LEVEL";
inline constexpr const char *SIRIUS_LOG_DIR_ENV   = "SIRIUS_LOG_DIR";

struct logger_state {
  level_enum level = level_enum::info;
  std::string log_file_path;
  std::FILE *file = nullptr;
  std::mutex mutex;

  ~logger_state() {
    if (file) {
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

inline level_enum GetLogLevel() {
  auto log_level_str = GetEnvVar(SIRIUS_LOG_LEVEL_ENV);
  if (log_level_str.has_value()) {
    if (*log_level_str == "trace") return level_enum::trace;
    if (*log_level_str == "debug") return level_enum::debug;
    if (*log_level_str == "info") return level_enum::info;
    if (*log_level_str == "warn") return level_enum::warn;
    if (*log_level_str == "error") return level_enum::err;
    if (*log_level_str == "critical") return level_enum::critical;
    if (*log_level_str == "off") return level_enum::off;
  }
  return level_enum::info;
}

inline std::string GetLogDir() {
  auto log_dir_str = GetEnvVar(SIRIUS_LOG_DIR_ENV);
  if (log_dir_str.has_value()) {
    return *log_dir_str;
  }
  return SIRIUS_DEFAULT_LOG_DIR;
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

inline bool should_log(level_enum level) {
  auto &state = global_logger();
  return state.level != level_enum::off && static_cast<int>(level) >= static_cast<int>(state.level);
}

inline std::string time_string() {
  auto now = std::chrono::system_clock::now();
  auto secs = std::chrono::system_clock::to_time_t(now);
  std::tm tm_value{};
#if defined(_WIN32)
  localtime_s(&tm_value, &secs);
#else
  localtime_r(&secs, &tm_value);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_value);
  return std::string(buffer);
}

inline std::FILE *open_log_file() {
  auto &state = global_logger();
  if (state.file) {
    return state.file;
  }
  if (state.log_file_path.empty()) {
    state.log_file_path = GetLogDir() + "/sirius.log";
  }
  std::filesystem::path path(state.log_file_path);
  auto parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }
  state.file = std::fopen(state.log_file_path.c_str(), "a");
  return state.file ? state.file : stderr;
}

inline void write_log_line(level_enum level, const std::string &message) {
  if (!should_log(level)) {
    return;
  }
  auto &state = global_logger();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto *out = open_log_file();
  std::fprintf(out, "[%s] [%s] %s\n",
               time_string().c_str(),
               level_name(level),
               message.c_str());
  std::fflush(out);
}

template <typename Format, typename... Args>
inline void log_message(level_enum level, const Format &format, Args&&... args) {
  if (!should_log(level)) {
    return;
  }
  write_log_line(level, duckdb_fmt::format(format, std::forward<Args>(args)...));
}

inline void InitGlobalLogger(std::string log_file = "") {
  auto &state = global_logger();
  std::lock_guard<std::mutex> guard(state.mutex);
  if (state.file) {
    std::fclose(state.file);
    state.file = nullptr;
  }
  state.level = GetLogLevel();
  state.log_file_path = log_file.empty() ? (GetLogDir() + "/sirius.log") : std::move(log_file);
}

} // namespace logging

inline void InitGlobalLogger(std::string log_file = "") {
  logging::InitGlobalLogger(std::move(log_file));
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
inline void trace(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::trace, format, std::forward<Args>(args)...);
}

template <typename Format, typename... Args>
inline void debug(const Format &format, Args&&... args) {
  duckdb::logging::log_message(level::debug, format, std::forward<Args>(args)...);
}

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

} // namespace spdlog

#define SIRIUS_LOG_TRACE(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::trace, __VA_ARGS__)
#define SIRIUS_LOG_DEBUG(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::debug, __VA_ARGS__)
#define SIRIUS_LOG_INFO(...)  ::duckdb::logging::log_message(::duckdb::logging::level_enum::info, __VA_ARGS__)
#define SIRIUS_LOG_WARN(...)  ::duckdb::logging::log_message(::duckdb::logging::level_enum::warn, __VA_ARGS__)
#define SIRIUS_LOG_ERROR(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::err, __VA_ARGS__)
#define SIRIUS_LOG_FATAL(...) ::duckdb::logging::log_message(::duckdb::logging::level_enum::critical, __VA_ARGS__)

#endif // SIRIUS_LOG_LOGGING_HPP
