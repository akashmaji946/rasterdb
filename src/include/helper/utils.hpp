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

// standard library
#include <bit>
#include <cstdint>
#include <type_traits>

namespace sirius::utils {

template <typename T>
inline constexpr T ceil_div(T a, T b)
{
  static_assert(std::is_integral_v<T>, "ceil_div requires an integral type");
  return (a + b - 1) / b;
}

template <typename T>
inline constexpr T ceil_div_8(T a)
{
  static_assert(std::is_integral_v<T>, "ceil_div_8 requires an integral type");
  static_assert(std::is_integral_v<T>, "ceil_div_8 requires an unsigned type");
  return (a + 7) >> 3;
}

template <typename T>
inline constexpr T div_8(T a)
{
  static_assert(std::is_integral_v<T>, "div_8 requires an integral type");
  static_assert(std::is_integral_v<T>, "div_8 requires an unsigned type");
  return a >> 3;
}

template <uint64_t Divisor, typename T>
inline constexpr T div(T a)
{
  static_assert(std::is_integral_v<T>, "div requires an integral type");
  static_assert(std::is_integral_v<T>, "div requires an unsigned type");
  static_assert(std::has_single_bit(Divisor), "div requires Divisor to be a power of two");
  uint64_t constexpr shift = std::countr_zero(Divisor);
  return a >> shift;
}

template <typename T>
inline constexpr T mul_8(T a)
{
  static_assert(std::is_integral_v<T>, "mul_8 requires an integral type");
  static_assert(std::is_integral_v<T>, "mul_8 requires an unsigned type");
  return a << 3;
}

template <typename T>
inline constexpr T mod_8(T a)
{
  static_assert(std::is_integral_v<T>, "mod_8 requires an integral type");
  static_assert(std::is_integral_v<T>, "mod_8 requires an unsigned type");
  return a & 7;
}

template <uint64_t Divisor, typename T>
inline constexpr T mod(T a)
{
  static_assert(std::is_integral_v<T>, "mod requires an integral type");
  static_assert(std::is_integral_v<T>, "mod requires an unsigned type");
  static_assert(std::has_single_bit(Divisor), "mod requires Divisor to be a power of two");
  return a & static_cast<T>(Divisor - 1);
}

template <typename T>
inline constexpr T align_8(T a)
{
  static_assert(std::is_integral_v<T>, "align_8 requires an integral type");
  static_assert(std::is_unsigned_v<T>, "align_8 requires an unsigned type");
  return (a + 7) & ~static_cast<T>(7);
}

template <typename S, typename T>
inline constexpr S make_mask(T num_bits)
{
  static_assert(std::is_integral_v<T>, "make_mask requires an integral type for num_bits");
  static_assert(std::is_integral_v<T>, "make_mask requires an unsigned type for num_bits");
  static_assert(std::is_integral_v<S>, "make_mask requires an integral type for return");
  static_assert(std::is_integral_v<S>, "make_mask requires an unsigned type for return");
  return static_cast<S>((static_cast<S>(1) << num_bits) - 1);
}

}  // namespace sirius::utils
