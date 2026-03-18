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

#include "creator/task_creator.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "extension_lock.hpp"
#include "memory/rasterdb_memory_reservation_manager.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "planner/query.hpp"
#include "rasterdb_config.hpp"
#include "rasterdb_pipeline_hashmap.hpp"

#include <rmm/resource_ref.hpp>

#include <duckdb/main/client_context.hpp>
#include <duckdb/main/client_context_state.hpp>
#include <duckdb/planner/extension_callback.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace cucascade::memory {
class small_pinned_host_memory_resource;
}  // namespace cucascade::memory

namespace duckdb {

/// \brief Manages the lifetime of the rasterdb_context within a DuckDB ClientContext.
class RasterDBContext : public ClientContextState {
 public:
  RasterDBContext();
  ~RasterDBContext() noexcept override;

  // Non-copyable and non-movable
  RasterDBContext(const RasterDBContext&)            = delete;
  RasterDBContext& operator=(const RasterDBContext&) = delete;
  RasterDBContext(RasterDBContext&&)                 = delete;
  RasterDBContext& operator=(RasterDBContext&&)      = delete;

  /// \brief Called at the beginning of a query execution.
  /// \param context The client context.
  void QueryBegin(ClientContext& context) final;

  /// \brief Called at the end of a query execution.
  void QueryEnd() final;

  /// \brief Called at the end of a query execution with context.
  /// \param context The client context.
  void QueryEnd(ClientContext& context) final;

  /// \brief Called at the end of a query execution with context and error data.
  /// \param context The client context.
  /// \param error Optional error data.
  void QueryEnd(ClientContext& context, optional_ptr<ErrorData> error) final;

  /// \brief Initialize the Sirius context with the given configuration.
  void initialize(const rasterdb::rasterdb_config& config);

  /// \brief Terminate the Sirius context, releasing all resources.
  void terminate();

  [[nodiscard]] const cucascade::memory::system_topology_info& get_hw_topology() const noexcept
  {
    return config_.get_hw_topology();
  }

  /// \brief Get the memory reservation manager.
  [[nodiscard]] rasterdb::memory::rasterdb_memory_reservation_manager& get_memory_manager();
  [[nodiscard]] const rasterdb::memory::rasterdb_memory_reservation_manager& get_memory_manager() const;

  [[nodiscard]] cucascade::shared_data_repository_manager& get_data_repository_manager();
  [[nodiscard]] const cucascade::shared_data_repository_manager& get_data_repository_manager()
    const;

  [[nodiscard]] rasterdb::pipeline::pipeline_executor& get_pipeline_executor();
  [[nodiscard]] const rasterdb::pipeline::pipeline_executor& get_pipeline_executor() const;

  /// \brief Get the downgrade executor for a specific memory space.
  [[nodiscard]] rasterdb::parallel::downgrade_executor& get_downgrade_executor(
    cucascade::memory::memory_space_id space_id);
  [[nodiscard]] const rasterdb::parallel::downgrade_executor& get_downgrade_executor(
    cucascade::memory::memory_space_id space_id) const;

  /// \brief Get all downgrade executors.
  [[nodiscard]] const std::vector<std::unique_ptr<rasterdb::parallel::downgrade_executor>>&
  get_downgrade_executors() const;

  [[nodiscard]] rasterdb::creator::task_creator& get_task_creator();
  [[nodiscard]] const rasterdb::creator::task_creator& get_task_creator() const;

  /// \brief Start a query with its pipeline hashmap.
  /// \param pipeline_hashmap The pipeline hashmap for the query.
  void create_query(rasterdb::rasterdb_pipeline_hashmap pipeline_hashmap);

  /// \brief Get the current query.
  [[nodiscard]] duckdb::shared_ptr<rasterdb::planner::query> get_query();
  [[nodiscard]] duckdb::shared_ptr<const rasterdb::planner::query> get_query() const;

  /// \brief Get the current Sirius configuration (const).
  [[nodiscard]] const rasterdb::rasterdb_config& get_config() const noexcept { return config_; }

  /// \brief Get the current Sirius configuration (mutable, e.g. for SET command callbacks).
  [[nodiscard]] rasterdb::rasterdb_config& get_config() noexcept { return config_; }

 private:
  void throw_if_not_initialized() const;

  mutable std::mutex mutex_;
  bool is_initialized_ = false;
  rasterdb::rasterdb_config config_;
  std::unique_ptr<rasterdb::memory::rasterdb_memory_reservation_manager> memory_manager_;
  // Destroyed before memory_manager_ (declared after it — reverse destruction order).
  std::unique_ptr<cucascade::memory::small_pinned_host_memory_resource> small_pinned_allocator_;
  // Previous cuDF pinned resource and threshold — restored in terminate() before
  // small_pinned_allocator_ is destroyed to prevent dangling references.
  std::optional<rmm::host_device_async_resource_ref> prev_pinned_mr_{};
  std::size_t prev_pinned_threshold_{0};
  std::unique_ptr<cucascade::shared_data_repository_manager> data_repository_manager_;
  std::unique_ptr<rasterdb::pipeline::pipeline_executor> pipeline_executor_;
  std::vector<std::unique_ptr<rasterdb::parallel::downgrade_executor>> downgrade_executors_;
  std::unique_ptr<rasterdb::creator::task_creator> task_creator_;
  duckdb::shared_ptr<rasterdb::planner::query> query_;
};

/// todo(amin): when duckdb is updated, we need to enable OnExtensionLoaded to support sirius
/// extensions
class RasterDBContextExtensionCallback : public ExtensionCallback {
 public:
  RasterDBContextExtensionCallback();

  /// \brief Called when a new connection is opened.
  /// \param context The client context.
  void OnConnectionOpened(ClientContext& context) final;

  /// \brief Called when a connection is closed.
  /// \param context The client context.
  void OnConnectionClosed(ClientContext& context) final;

  /// \brief Called when an extension is loaded.
  /// \param db The database instance.
  /// \param name The name of the loaded extension.
  void OnExtensionLoaded(DatabaseInstance& db, const string& name) final;

  void OnBeginExtensionLoad(DatabaseInstance& db, const string& name) final;

  //! Called after an extension fails to load loading
  void OnExtensionLoadFail(DatabaseInstance& db, const string& name, const ErrorData& error) final;

 private:
  void read_config_file_if_exists();

  std::unique_ptr<rasterdb::extension_lock> extension_lock_;
  rasterdb::rasterdb_config config_;
  duckdb::shared_ptr<RasterDBContext> context_;
};

}  // namespace duckdb
