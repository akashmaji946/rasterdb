-- simple.sql
-- Table creation, data insertion, and 10 SPJ queries using RasterDB's gpu_execution


-- start as:
-- assuming rduckdb is installed and available in PATH (refer installation)
-- rduckdb simple.db


-- 1. Create tables
CREATE TABLE users (
    id    INTEGER NOT NULL,
    age   INTEGER NOT NULL,
    score INTEGER NOT NULL,
    dept  INTEGER NOT NULL
);

CREATE TABLE departments (
    dept   INTEGER NOT NULL,
    budget INTEGER NOT NULL
);

-- 2. Insert sample data
INSERT INTO users VALUES 
    (1, 20, 85, 1), 
    (2, 35, 95, 2), 
    (3, 65, 40, 1), 
    (4, 42, 100, 3), 
    (42, 25, 75, 2);

INSERT INTO departments VALUES 
    (1, 100), 
    (2, 200), 
    (3, 150);

-- 3. Execute queries using gpu_execution

-- Q1: Full table scan
SELECT * FROM gpu_execution('SELECT * FROM users;');

-- Q2: Point lookup (equality filter)
SELECT * FROM gpu_execution('SELECT * FROM users WHERE id = 42;');

-- Q3: Projection + range filter
SELECT * FROM gpu_execution('SELECT id, age FROM users WHERE age > 60;');

-- Q4: Compound AND predicate
SELECT * FROM gpu_execution('SELECT * FROM users WHERE age >= 25 AND score > 90;');

-- Q5: Compound OR predicate
SELECT * FROM gpu_execution('SELECT * FROM users WHERE age < 20 OR score > 95;');

-- Q6: NOT predicate
SELECT * FROM gpu_execution('SELECT * FROM users WHERE NOT (dept = 1);');

-- Q7: Column-column comparison
SELECT * FROM gpu_execution('SELECT * FROM users WHERE score > age;');

-- Q8: Inner join
SELECT * FROM gpu_execution(
  'SELECT u.id, u.age, u.score, u.dept AS user_dept, d.dept AS department_dept, d.budget
   FROM users u
   JOIN departments d ON u.dept = d.dept'
);

-- Q9: GroupBy with SUM aggregation
SELECT * FROM gpu_execution('SELECT dept, SUM(score) FROM users GROUP BY dept;');

-- Q10: Filter + GroupBy with COUNT aggregation
SELECT * FROM gpu_execution('SELECT dept, COUNT(score) FROM users WHERE age > 30 GROUP BY dept;');
