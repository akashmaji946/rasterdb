-- 00_simple.sql
-- 10 Simple SPJ queries for RasterDB (librasterdf + DuckDB)
-- Tables: users (id, age, score, dept), departments (dept, budget)

-- Q1: Full table scan
SELECT * FROM users;

-- Q2: Point lookup (equality filter)
SELECT * FROM users WHERE id = 42;

-- Q3: Projection + range filter
SELECT id, age FROM users WHERE age > 60;

-- Q4: Compound AND predicate
SELECT * FROM users WHERE age >= 25 AND score > 90;

-- Q5: Compound OR predicate
SELECT * FROM users WHERE age < 20 OR score > 95;

-- Q6: NOT predicate
SELECT * FROM users WHERE NOT (dept = 1);

-- Q7: Column-column comparison
SELECT * FROM users WHERE score > age;

-- Q8: Inner join
SELECT * FROM users u JOIN departments d ON u.dept = d.dept;

-- Q9: GroupBy with SUM aggregation
SELECT dept, SUM(score) FROM users GROUP BY dept;

-- Q10: Filter + GroupBy with COUNT aggregation
SELECT dept, COUNT(score) FROM users WHERE age > 30 GROUP BY dept;
