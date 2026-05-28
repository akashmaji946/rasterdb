# Decimal GPU Support

RasterDB must execute DuckDB `DECIMAL(width, scale)` as fixed-point integers,
not as `FLOAT64`. A stored value is the unscaled integer payload together with
its logical scale. For example, `DECIMAL(9,2)` value `12.34` is compared as
integer `1234` with scale `2`.

## Implemented Foundation

The first exact path supports transport and constant filtering for decimals
that fit in 32 or 64 bits:

| DuckDB type | DuckDB physical storage | RasterDB GPU storage |
| --- | --- | --- |
| `DECIMAL(1..4, s)` | `INT16` | Widened `INT32` payload plus scale |
| `DECIMAL(5..9, s)` | `INT32` | `INT32` payload plus scale |
| `DECIMAL(10..18, s)` | `INT64` | `INT64` payload plus scale |
| `DECIMAL(19..38, s)` | `INT128` | Not implemented; CPU fallback |

Implemented behavior:

- Scan/upload preserves the decimal scale and widens DuckDB `INT16` decimal
  storage before GPU execution.
- Output materialization writes `INT32` and `INT64` payloads back into the
  correct DuckDB decimal physical representation, including checked narrowing
  back to `INT16`.
- Column-to-constant filters encode decimal literals as scaled integer
  payloads, so values such as `12.34` are not truncated or converted through
  floating point.
- Decimal scale-changing casts and mixed-scale column comparisons currently
  throw from the GPU executor and fall back to DuckDB instead of returning an
  inexact result.
- Decimal `ORDER BY` keys use a signed fixed-point radix transform for
  `DECIMAL32` and `DECIMAL64`.
- Decimal arithmetic, decimal value aggregation, and decimal `TOP N` keys
  currently fall back until their rescale, accumulator, and top-n key paths are
  implemented.
- Equality joins on decimal keys may use the GPU only when both key types have
  identical width and scale.

Smoke coverage is in `test/test_decimal_gpu.sql`.

## Operation Roadmap

### Phase 2: Same-Scale Fixed-Point Operators

- Validate equi-join and group-key paths for equal decimal types, where raw
  payload equality is already the required operation.
- Extend the signed sortable transform route to `TOP N`.
- Add tests with negative, zero, maximum-width, and duplicate-key values.

### Phase 3: Rescale And Arithmetic

- Add a decimal rescale operator using checked multiply/divide by powers of
  ten with DuckDB-compatible rounding rules.
- Enable decimal casts and mixed-scale comparison only through that operator.
- Implement addition and subtraction after operand scales are aligned.
- Implement multiplication with widened intermediate payloads and derived
  output scale.
- Implement division/modulo only after precision, rounding, and overflow
  behavior match DuckDB.

### Phase 4: Aggregation

- Keep `MIN` and `MAX` at the input decimal scale.
- Widen `SUM(DECIMAL32)` to a decimal 64-bit accumulator where its result type
  allows it.
- Support DuckDB-compatible `SUM(DECIMAL64)` output only after `DECIMAL128`
  storage and reduction are implemented.
- Produce `AVG` using the bound DuckDB result type and an explicit decimal to
  floating-point conversion, never by reducing unscaled integers as doubles.

### Phase 5: DECIMAL128

- Add a 16-byte two-limb payload representation for Vulkan shaders.
- Implement compare, equality/hash, gather, materialization, rescale, and
  overflow checks before enabling arithmetic or aggregates.
- Keep all width greater than 18 queries on CPU fallback until these operators
  pass differential tests against DuckDB.

## Validation Rule

Every enabled decimal GPU operator should be tested against DuckDB CPU results
for positive values, negative values, zeros, boundaries, differing scales, and
overflow cases. Unsupported operators must fall back rather than use a
floating-point approximation.
