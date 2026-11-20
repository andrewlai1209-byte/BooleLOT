"""New Python transcriptions of mathematical procedures from period sources."""

# These examples teach syntax and semantics. Source IDs refer to the mathematics,
# not to a claim that the historical authors wrote Python.
PROGRAMS = {
    "gcd": '''def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a
''',
    "lcm": '''def lcm(a, b):
    if not a or not b:
        return 0
    x, y = abs(a), abs(b)
    while y:
        x, y = y, x % y
    return abs(a * b) // x
''',
    "intersection": '''def intersection(a, b):
    return set(a) & set(b)
''',
    "union": '''def union(a, b):
    return set(a) | set(b)
''',
    "complement": '''def complement(universe, subset):
    return set(universe) - set(subset)
''',
    "fibonacci": '''def fibonacci(n):
    if n < 0:
        raise ValueError("n must be nonnegative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
    "binomial": '''def binomial(n, k):
    if n < 0 or k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result
''',
    "primes": '''def primes(n):
    if n < 2:
        return []
    marked = [True] * (n + 1)
    p = 2
    while p * p <= n:
        if marked[p]:
            for multiple in range(p * p, n + 1, p):
                marked[multiple] = False
        p += 1
    return [i for i in range(2, n + 1) if marked[i]]
''',
    "triangular": '''def triangular(n):
    if n < 0:
        raise ValueError("n must be nonnegative")
    return n * (n + 1) // 2
''',
    "perfect_number": '''def perfect_number(n):
    if n < 2:
        return False
    total = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            total += d
            if d != n // d:
                total += n // d
        d += 1
    return total == n
''',
    "divisors": '''def divisors(n):
    if n < 1:
        raise ValueError("n must be positive")
    small = []
    large = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small += [d]
            if d != n // d:
                large += [n // d]
        d += 1
    return sorted(small + large)
''',
    "integer_power": '''def integer_power(base, exponent):
    if exponent < 0:
        raise ValueError("exponent must be nonnegative")
    value = 1
    for _ in range(exponent):
        value *= base
    return value
''',
    "median": '''def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("at least one value is required")
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
''',
    "matrix_product": '''def matrix_product(a, b):
    if not a or not b or not b[0]:
        raise ValueError("incompatible matrices")
    inner, width = len(b), len(b[0])
    if any(len(row) != inner for row in a) or any(len(row) != width for row in b):
        raise ValueError("incompatible matrices")
    return [[sum(row[k] * b[k][j] for k in range(inner)) for j in range(width)] for row in a]
''',
    "determinant_2x2": '''def determinant_2x2(matrix):
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("expected a two-by-two matrix")
    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c
''',
    "pascal_row": '''def pascal_row(n):
    if n < 0:
        raise ValueError("n must be nonnegative")
    row = [1]
    for i in range(1, n + 1):
        row += [row[i - 1] * (n - i + 1) // i]
    return row
''',
}
