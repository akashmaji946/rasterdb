#pragma once

#include <data/data_repository.hpp>
#include <parallel/task_executor.hpp>
#include <scan/duckdb_scan_task_new.hpp>

namespace sirius::parallel
{

/// KEVIN:
/// Inheritance from ItaskExecutor exposes
/// -- Start(), Stop(), to launch the worker loop over threads.
/// So, the Task must know the data repository to push to.
class DuckDBScanExecutor : public ITaskExecutor
{
public:
  DuckDBScanExecutor(TaskExecutorConfig config, DataRepository& data_repository)
      : ITaskExecutor(sirius::make_shared<DuckDBScanTaskQueue>(), config)
      , data_repository_(data_repository)
  {}

  // Add a task to the scan queue  
  void AddScanTask(sirius::unique_ptr<DuckDBScanTask> scan_task) {
    Schedule(std::move(scan_task));
  }

private:
  DataRepository& data_repository_;
};

} // namespace sirius::parallel