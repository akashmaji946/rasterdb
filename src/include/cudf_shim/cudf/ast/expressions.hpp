/*
 * RasterDB cudf::ast compatibility shim.
 * Provides AST expression types so existing code compiles.
 */
#pragma once
#include "../../types.hpp"
#include "../../column.hpp"

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <stdexcept>

namespace cudf {
namespace ast {

enum class ast_operator : int32_t {
    ADD, SUB, MUL, DIV, TRUE_DIV, FLOOR_DIV, MOD, PYMOD, POW,
    EQUAL, NOT_EQUAL, LESS, GREATER, LESS_EQUAL, GREATER_EQUAL,
    BITWISE_AND, BITWISE_OR, BITWISE_XOR,
    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
    IDENTITY, SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN,
    SINH, COSH, TANH, EXP, LOG, SQRT, CBRT, CEIL, FLOOR,
    ABS, RINT, BIT_INVERT, NOT,
    CAST_TO_INT64, CAST_TO_UINT64, CAST_TO_FLOAT64,
    NULL_EQUAL, NULL_NOT_EQUAL, NULL_LOGICAL_AND, NULL_LOGICAL_OR,
    NULL_MAX, NULL_MIN
};

class expression {
public:
    virtual ~expression() = default;
};

class column_reference : public expression {
public:
    enum class table_reference { LEFT, RIGHT, OUTPUT };
    column_reference(cudf::size_type col_idx, table_reference tbl = table_reference::LEFT)
        : _col_idx(col_idx), _table(tbl) {}
    cudf::size_type get_column_index() const { return _col_idx; }
    table_reference get_table_source() const { return _table; }
private:
    cudf::size_type _col_idx;
    table_reference _table;
};

class literal : public expression {
public:
    template <typename T>
    literal(T) {}
};

class operation : public expression {
public:
    operation(ast_operator op, expression const& l)
        : _op(op), _operands{&l, nullptr} {}
    operation(ast_operator op, expression const& l, expression const& r)
        : _op(op), _operands{&l, &r} {}
    ast_operator get_operator() const { return _op; }
private:
    ast_operator _op;
    expression const* _operands[2];
};

class column_name_reference : public expression {
public:
    column_name_reference(std::string name) : _name(std::move(name)) {}
    const std::string& get_column_name() const { return _name; }
private:
    std::string _name;
};

// AST tree — container for expressions
class tree {
public:
    tree() = default;
    tree(tree&&) = default;
    tree& operator=(tree&&) = default;
    tree(tree const&) = default;
    tree& operator=(tree const&) = default;

    template <typename ExprT, typename... Args>
    ExprT& emplace(Args&&... args) {
        auto ptr = std::make_unique<ExprT>(std::forward<Args>(args)...);
        auto& ref = *ptr;
        _exprs.push_back(std::move(ptr));
        return ref;
    }

    expression const& back() const { return *_exprs.back(); }
    expression const& front() const { return *_exprs.front(); }
    std::size_t size() const { return _exprs.size(); }
    bool empty() const { return _exprs.empty(); }

private:
    std::vector<std::unique_ptr<expression>> _exprs;
};

} // namespace ast

// Bring table_reference to cudf::ast namespace level for backward compat
namespace ast {
using table_reference = column_reference::table_reference;
} // namespace ast

// cudf::compute_column stub
inline std::unique_ptr<column> compute_column(table_view const&, ast::expression const&) {
    throw std::runtime_error("cudf::compute_column not implemented in RasterDB — use rasterdf");
}

// cudf::transform stub
inline std::unique_ptr<column> transform(column_view const&, std::string const&, data_type, bool) {
    throw std::runtime_error("cudf::transform not implemented in RasterDB — use rasterdf");
}

// cudf null_aware / null policy helpers
enum class null_aware : bool { NO, YES };

} // namespace cudf
