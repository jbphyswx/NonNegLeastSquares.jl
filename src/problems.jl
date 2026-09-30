"""Supertype for problem representations accepted by [`solve_nnls`](@ref)."""
abstract type AbstractNNLSProblem end

"""
    NNLSData(A, B)

Represent `minimize norm(A*X-B)` subject to `X >= 0`. `A` is a matrix;
`B` is a vector or matrix with the same number of rows. Stores the inputs
without copying, converting, or factoring them.
"""
struct NNLSData{A,B} <: AbstractNNLSProblem
    A::A
    B::B
end

"""
    NNLSGram(G, C)

Represent an NNLS problem by `G = A'*A` and `C = A'*B`. `G` is a square
matrix; `C` is a vector or matrix with matching rows. Stores the inputs
without copying or factoring them. Use `FastNNLS()` or `Pivot(CachedPivot())`.
The original data and absolute residual norm cannot be recovered from `G, C`.
"""
struct NNLSGram{G,C} <: AbstractNNLSProblem
    G::G
    C::C
end
