# =============================================================================
# Sirius scan tests
# =============================================================================

set(TEST_SOURCES
  ${TEST_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/test_duckdb_scan_task.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_duckdb_scan_tpch.cpp
  PARENT_SCOPE)
