/*
 * Copyright 2025, RasterDB Contributors.
 * String Dictionary — CPU-side dictionary encoder / decoder for VARCHAR columns.
 *
 * During scan, low-cardinality VARCHAR columns are dictionary-encoded to INT32 codes.
 * This lets ALL existing INT32 GPU shaders (compare, hash, sort, join, groupby)
 * work on string columns with zero new shader development.
 *
 * Layout mirrors cuDF's DICTIONARY32 type concept:
 *   GPU column:  int32_t codes[N]        (one code per row)
 *   CPU side:    vector<string> dictionary (code → string mapping)
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rasterdb {
namespace gpu {

/// Dictionary encoder for a single VARCHAR column.
/// Thread-safe for single-writer (scan thread), lock-free readers.
struct string_dictionary {
  /// String → code mapping (for encoding during scan)
  std::unordered_map<std::string, int32_t> str_to_code;

  /// Code → string mapping (for decoding results back to VARCHAR)
  std::vector<std::string> code_to_str;

  /// Encode a string to its INT32 code.  Inserts new entry if unseen.
  int32_t encode(std::string_view s) {
    auto it = str_to_code.find(std::string(s));
    if (it != str_to_code.end()) return it->second;
    int32_t code = static_cast<int32_t>(code_to_str.size());
    code_to_str.emplace_back(s);
    str_to_code[std::string(s)] = code;
    return code;
  }

  /// Decode an INT32 code back to its string.
  const std::string& decode(int32_t code) const {
    return code_to_str[static_cast<size_t>(code)];
  }

  /// Number of unique strings in the dictionary.
  size_t cardinality() const { return code_to_str.size(); }

  /// Clear the dictionary.
  void clear() {
    str_to_code.clear();
    code_to_str.clear();
  }

  /// Build a sorted dictionary where codes are assigned in lexicographic order.
  /// This allows ORDER BY on dictionary-encoded columns to produce correct results
  /// using the existing INT32 radix sort (since code order = lexicographic order).
  void rebuild_sorted() {
    std::vector<std::string> sorted_strs(code_to_str.begin(), code_to_str.end());
    std::sort(sorted_strs.begin(), sorted_strs.end());
    str_to_code.clear();
    code_to_str = sorted_strs;
    for (int32_t i = 0; i < static_cast<int32_t>(sorted_strs.size()); i++) {
      str_to_code[sorted_strs[i]] = i;
    }
  }
};

/// Per-table dictionary metadata.  Stored alongside gpu_table.
struct table_dictionaries {
  /// col_index → dictionary  (only for dictionary-encoded VARCHAR columns)
  std::unordered_map<size_t, string_dictionary> col_dicts;

  bool has_dict(size_t col_idx) const {
    return col_dicts.count(col_idx) > 0;
  }
  string_dictionary& get(size_t col_idx) {
    return col_dicts[col_idx];
  }
  const string_dictionary& get(size_t col_idx) const {
    return col_dicts.at(col_idx);
  }
};

} // namespace gpu
} // namespace rasterdb
