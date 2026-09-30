"""
    solve_nnls(problem::AbstractNNLSProblem, algorithm::AbstractNNLSAlgorithm; kwargs...)

Solve an NNLS problem with an explicit algorithm. Built-in methods return a
matrix, including an `n×1` matrix for a vector RHS. Solver-specific keywords
are forwarded to the numerical kernel. The problem type determines whether
inputs are data or Gram matrices; it does not select a different algorithm.

Extend this function with positional methods for algorithm or problem types
owned by your package. For pivot policies, extend [`solve_pivot`](@ref).
"""
function solve_nnls(problem::AbstractNNLSProblem, algorithm::AbstractNNLSAlgorithm; kwargs...)
    throw(ArgumentError("unsupported problem/algorithm: $(typeof(problem)), $(typeof(algorithm))"))
end

"""
    solve_pivot(problem::AbstractNNLSProblem, variant::AbstractPivotVariant; kwargs...)

Positional extension point for `Pivot(variant)`. Built-in policies support
`NNLSData`; `CachedPivot` also supports `NNLSGram`. Return an `n×k` solution
matrix, where `k=1` for a vector RHS. Built-in methods leave the input data
unchanged; `GroupedPivot` mutates an explicitly supplied `P!` warm-start mask.
"""
function solve_pivot(problem::AbstractNNLSProblem, variant::AbstractPivotVariant; kwargs...)
    throw(ArgumentError("unsupported problem/pivot policy: $(typeof(problem)), $(typeof(variant))"))
end

_as_matrix(b::AbstractVector) = reshape(b, length(b), 1)
_as_matrix(B::AbstractMatrix) = B

solve_nnls(p::AbstractNNLSProblem, alg::Pivot; kwargs...) =
    solve_pivot(p, alg.variant; kwargs...)

solve_pivot(p::NNLSData, ::DirectPivot; kwargs...) =
    pivot(p.A, _as_matrix(p.B); kwargs...)
solve_pivot(p::NNLSData, ::CachedPivot; kwargs...) =
    pivot_cache(p.A, _as_matrix(p.B); kwargs..., gram=false)
solve_pivot(p::NNLSGram, ::CachedPivot; kwargs...) =
    pivot_cache(p.G, _as_matrix(p.C); kwargs..., gram=true)
solve_pivot(p::NNLSData, ::GroupedPivot; kwargs...) =
    pivot_comb(p.A, _as_matrix(p.B); kwargs...)

solve_nnls(p::NNLSData, ::LawsonHanson; kwargs...) =
    nnls(p.A, _as_matrix(p.B); kwargs...)
solve_nnls(p::NNLSData, ::FastNNLS; kwargs...) =
    fnnls(p.A, _as_matrix(p.B); kwargs..., gram=false)
solve_nnls(p::NNLSGram, ::FastNNLS; kwargs...) =
    fnnls(p.G, _as_matrix(p.C); kwargs..., gram=true)
solve_nnls(p::NNLSData, ::DeviationMaximization; kwargs...) =
    lhdm(p.A, _as_matrix(p.B); kwargs...)
