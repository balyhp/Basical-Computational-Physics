from typing import List, Tuple

def gaussian_elimination_pp_solve_multi(A: List[List[float]],
                                        B: List[List[float]],
                                        tol: float = 1e-12) -> List[List[float]]:
    """
    Solve A X = B for X using Gaussian elimination with partial pivoting.
    - A: n x n matrix
    - B: n x m matrix (m right-hand sides). For a single RHS, shape is n x 1.
    Returns X as n x m matrix.
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be square and non-empty")
    if len(B) != n:
        raise ValueError("B must have the same number of rows as A")
    m = len(B[0]) if B else 0
    if any(len(row) != m for row in B):
        raise ValueError("All rows of B must have the same length")

    # Build augmented matrix [A | B]
    M = [list(map(float, A[i])) + list(map(float, B[i])) for i in range(n)]

    # Forward elimination with partial pivoting
    for k in range(n):
        # Select pivot row with largest absolute value in current column k
        pivot_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[pivot_row][k]) < tol:
            raise ValueError("Matrix is singular or nearly singular at column {}".format(k))
        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]  #exchange two rows

        # Eliminate below
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            if factor == 0.0:
                continue
            # Update the rest of the row, including RHS blocks
            for j in range(k, n + m):
                M[i][j] -= factor * M[k][j]

    # Back substitution for all RHS columns simultaneously
    X = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        diag = M[i][i]
        if abs(diag) < tol:
            raise ValueError("Zero diagonal encountered during back substitution")
        for r in range(m):
            s = 0.0
            for k in range(i + 1, n):
                s += M[i][k] * X[k][r]
            rhs = M[i][n + r] - s
            X[i][r] = rhs / diag
    return X

def gaussian_elimination_pp_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b using Gaussian elimination with partial pivoting."""
    B = [[bi] for bi in b]  # shape.B = n x 1 
    X = gaussian_elimination_pp_solve_multi(A, B)
    return [row[0] for row in X]

def inverse_via_ge_pp(A: List[List[float]]) -> List[List[float]]:
    """Compute A^{-1} by solving A X = I using one elimination with partial pivoting."""
    n = len(A)
    I = [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]   # identity matrix
    return gaussian_elimination_pp_solve_multi(A, I)

def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiply (for quick verification)."""
    n, p, m = len(A), len(A[0]), len(B[0])
    if any(len(row) != p for row in A) or any(len(row) != m for row in B) or len(B) != p:
        raise ValueError("Incompatible shapes for matmul")
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            aik = A[i][k]
            for j in range(m):
                C[i][j] += aik * B[k][j]
    return C

def main():
    # Coefficient matrix and RHS from the problem
    A = [
        [2, 3, 5],
        [3, 4, 8],
        [1, 3, 3],
    ]
    b = [5, 6, 5]
    #A = [[2, 3, 5],
    #    [3, 5, 8],
    #    [2, 3, 3],]
    #b = [3,4,2]
    
    # Solve A x = b
    x = gaussian_elimination_pp_solve(A, b)
    print("Solution x:")
    print(["{:.8f}".format(v) for v in x])

    # Optional: compute inverse
    Ainv = inverse_via_ge_pp(A)
    print("\nA^{-1}:")
    for row in Ainv:
        print(" ".join("{: .8f}".format(v) for v in row))

    # Quick verification
    Ax = [sum(A[i][j] * x[j] for j in range(3)) for i in range(3)]
    print("\nCheck A x:")
    print(["{:.8f}".format(v) for v in Ax])

    # Check A * A^{-1} ≈ I
    AAinv = matmul(A, Ainv)
    print("\nCheck A * A^{-1}:")
    for row in AAinv:
        print(" ".join("{: .8f}".format(v) for v in row))

if __name__ == "__main__":
    main()