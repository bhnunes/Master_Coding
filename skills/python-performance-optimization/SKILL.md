---
name: python-performance-optimization
description: >
  Use when optimizing Python runtime performance, identifying bottlenecks,
  profiling slow code, or improving performance in scientific workloads,
  pandas pipelines, loops, or parallel execution.
compatibility: opencode
---

# Python Performance Optimization

This skill guides the agent to systematically optimize Python code for runtime
performance while preserving correctness, readability, and maintainability.

Use this skill when tasks involve:

- speeding up Python code
- reducing runtime or latency
- identifying performance bottlenecks
- profiling functions or scripts
- optimizing pandas, NumPy, loops, or parallel workloads
- improving performance in scientific computing pipelines

Do NOT use this skill when:

- the problem is unrelated to performance
- the workload is primarily I/O-bound without I/O optimization requests
- the optimization would require unsafe behavior changes

---

# Core Principles

Always follow this order:

1.  Measure first
2.  Identify the real bottleneck
3.  Prefer algorithmic improvements over micro-optimizations
4.  Reduce Python interpreter overhead
5.  Use vectorization or compiled paths when appropriate
6.  Re-measure after changes
7.  Validate correctness

Never assume a bottleneck without evidence.

---

# Performance Investigation Workflow

The agent must follow this workflow when optimizing code:

1.  Measure runtime using `time.perf_counter`
2.  If the code is slow, run `cProfile`
3.  Identify the top time-consuming functions
4.  If needed, run `line_profiler` for line-level analysis
5.  Optimize algorithm or loops
6.  Re-measure runtime
7.  Confirm output correctness

------------------------------------------------------------------------

# Profiling Tools

## Quick timing (first step)

Use `time.perf_counter()` for quick measurements.

Example:

``` python
from time import perf_counter

start = perf_counter()
result = my_function(data)
end = perf_counter()

print(f"Execution time: {end - start:.4f} seconds")
```

Use when: - comparing two implementations - measuring overall runtime

------------------------------------------------------------------------

## cProfile (function-level profiling)

Use when identifying which functions consume the most time.

CLI usage:

    python -m cProfile -s tottime script.py

Python usage:

``` python
import cProfile
import pstats

with cProfile.Profile() as pr:
    my_function()

stats = pstats.Stats(pr)
stats.sort_stats("tottime").print_stats(20)
```

Interpretation: - `tottime` = time spent inside the function itself -
`cumtime` = time including subcalls

------------------------------------------------------------------------

## line_profiler (line-level profiling)

Use when a specific function appears slow.

Install:

    pip install line_profiler

Example:

``` python
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(my_function)

lp.run('my_function(data)')
lp.print_stats()
```

------------------------------------------------------------------------

## py-spy (sampling profiler)

Useful for profiling a running process without modifying code.

Install:

    pip install py-spy

Run:

    py-spy top -- python script.py

Flame graph:

    py-spy record -o profile.svg -- python script.py

Open the SVG to visualize hotspots.

------------------------------------------------------------------------

## Scalene profiler

Modern profiler that identifies CPU, memory, and GPU bottlenecks.

Install:

    pip install scalene

Run:

    scalene script.py

Scalene displays: - CPU time - memory allocation - Python vs native
execution time

------------------------------------------------------------------------

# Optimization Heuristics

## Algorithm and Data Structures

Prefer: 
- sets or dictionaries instead of lists for membership checks 
- indexing or grouping instead of repeated scanning 
- one-pass aggregation instead of repeated filtering 
- improved asymptotic complexity

------------------------------------------------------------------------

## Reduce Python Interpreter Overhead

Prefer: 
- built-in functions over manual loops 
- comprehensions when appropriate 
- hoisting invariant computations outside loops 
- local variables inside hot loops 
- batching operations instead of repeated calls

------------------------------------------------------------------------

## Numeric and Scientific Workloads

Prefer: 
- NumPy vectorization for numerical arrays 
- avoiding Python loops over arrays 
- in-place operations when safe 
- compact data types 
- Numba for hot numeric loops when vectorization is insufficient

------------------------------------------------------------------------

## pandas Optimization

Prefer: 
- vectorized column operations 
- `merge`, `map`, `groupby`, and boolean masks 
- avoiding `iterrows()` 
- minimizing DataFrame copies 
- categorical dtypes when appropriate

Avoid:

    df.apply(axis=1)

unless absolutely necessary.

------------------------------------------------------------------------

## Parallelism

Use:

-   multiprocessing for CPU-bound independent tasks
-   threading for I/O-bound workloads
-   joblib for embarrassingly parallel workloads

Avoid:

-   asyncio for CPU acceleration
-   multiprocessing for small tasks
-   parallelism when serialization dominates execution time

------------------------------------------------------------------------

# Memory-Speed Tradeoffs

Performance may improve by:

-   reducing object creation
-   minimizing memory copies
-   preallocating arrays
-   using views instead of copies
-   improving cache locality

------------------------------------------------------------------------

# Anti-Patterns

Do not:

-   optimize without measuring
-   claim speed improvements without benchmarks
-   introduce obscure tricks for negligible gains
-   silently change numerical behavior
-   use vectorization that causes memory explosion
-   introduce heavy dependencies without significant benefit

------------------------------------------------------------------------

# Benchmark Template

Use this template to compare implementations:

``` python
from time import perf_counter

def benchmark(fn, *args, repeats=5, **kwargs):
    times = []
    result = None

    for _ in range(repeats):
        start = perf_counter()
        result = fn(*args, **kwargs)
        end = perf_counter()

        times.append(end - start)

    avg = sum(times) / len(times)

    print(f"Average runtime: {avg:.6f} seconds")

    return result
```

Always ensure both baseline and optimized versions are run using the
same inputs.

------------------------------------------------------------------------

# Correctness Validation

After optimization, verify:

-   identical outputs
-   acceptable floating point tolerance
-   preserved ordering when required
-   preserved side effects

Benchmark improvements are invalid if correctness changes.

------------------------------------------------------------------------

# Preferred Libraries

Use existing tools whenever possible:

-   time
-   timeit
-   cProfile
-   pstats
-   functools
-   itertools
-   collections
-   heapq
-   numpy
-   pandas
-   numba
-   joblib
-   multiprocessing

Avoid adding dependencies unless the performance gain is significant.
