"""
    nonneg_lsq(A, B; gram=false, alg=Pivot(), variant=:none, kwargs...)
    nonneg_lsq(A, B, algorithm::AbstractNNLSAlgorithm; gram=false, kwargs...)
    nonneg_lsq(A, B, algorithm::AbstractNNLSAlgorithm, gram::Bool; kwargs...)

Compute the matrix `X` minimizing `norm(A*X-B)` subject to `X >= 0`.
For an `m×n` matrix `A` and an `m×k` matrix `B`, the result is `n×k`.
A vector RHS produces an `n×1` matrix.

Algorithms may be specified by objects or legacy symbols:

- `Pivot()`: direct block pivoting (default).
- `Pivot(CachedPivot())`: cached Gram matrices.
- `Pivot(GroupedPivot())`: grouped passive-set solves.
- `LawsonHanson()`: classic Lawson–Hanson NNLS.
- `FastNNLS()`: Bro–De Jong fast NNLS.
- `DeviationMaximization()`: Lawson–Hanson with deviation maximization.

With `gram=true`, supply `A'*A` and `A'*B` instead of the data.
If `alg` is omitted, the default is `Pivot(CachedPivot())` for Gram input and
`Pivot()` otherwise. Explicit typed algorithms are used as specified.
For typed algorithms, `variant` may be omitted or match the contained policy:
`:none` for direct pivoting, `:cache` for cached pivoting, and `:comb` for grouped
pivoting. Other algorithm types use `:none`.

Solver keywords are forwarded unchanged. Pivot variants accept `tol` (absolute
feasibility allowance), `rtol` (relative allowance with separate primal/dual
scales), and `max_iter` (maximum pivot passes; exhaustion throws). `rtol=0`
selects absolute-only tolerances. `max_iter=0` permits an initially feasible
solution. `use_parallel` controls threading over RHS columns for algorithms
other than `GroupedPivot`, which does not accept that keyword.

For downstream extensions, implement [`solve_nnls`](@ref) or [`solve_pivot`](@ref)
on types owned by your package. No symbol registration is required.

The future of this interface is to unify the various NNLS algorithms under this single method
"""
function nonneg_lsq(A, B, alg::AbstractNNLSAlgorithm, gram::Bool; kwargs...)
    problem = gram ? NNLSGram(A, B) : NNLSData(A, B)
    return solve_nnls(problem, alg; kwargs...)
end

@inline nonneg_lsq(A, B, alg::AbstractNNLSAlgorithm; gram::Bool=false, kwargs...) =
    nonneg_lsq(A, B, alg, gram; kwargs...)

_legacy_default_variant(::Symbol) = :none
_legacy_default_variant(::AbstractNNLSAlgorithm) = :none
_legacy_default_variant(alg::Pivot) = _legacy_default_variant(alg.variant)
_legacy_default_variant(::AbstractPivotVariant) = :none
_legacy_default_variant(::CachedPivot) = :cache
_legacy_default_variant(::GroupedPivot) = :comb

"""
    Backwards compatible interface method
"""
function nonneg_lsq(A, B; gram::Bool=false, alg::Union{Symbol, AbstractNNLSAlgorithm} = (gram ? Pivot(CachedPivot()) : Pivot()), variant::Symbol = _legacy_default_variant(alg), kwargs...)
    if (alg isa AbstractNNLSAlgorithm)
        if !(variant === _legacy_default_variant(alg))
            throw(ArgumentError("mismatched variant $variant for algorithm $alg, pass matching default or use a symbol alg for legacy interface"))
        end
        return nonneg_lsq(A, B, alg, gram; kwargs...)
    end

    variant in (:none, :cache, :comb) || throw(ArgumentError("unknown variant: $variant"))
    algorithm = if (alg === :pivot)
        if gram
            (variant === :comb) && throw(ArgumentError("grouped pivoting does not support Gram input"))
            Pivot(CachedPivot())
        else
            (variant === :none) ? Pivot() : ((variant === :cache) ? Pivot(CachedPivot()) : Pivot(GroupedPivot()))
        end
    else
        (variant === :none) || throw(ArgumentError("variant only applies to alg=:pivot"))
        if (alg === :nnls)
            (gram ? FastNNLS() : LawsonHanson())
        elseif alg === :fnnls
            FastNNLS()
        elseif alg === :lhdm
            DeviationMaximization()
        else
            throw(ArgumentError("unknown algorithm: $alg"))
        end
    end

    return nonneg_lsq(A, B, algorithm, gram; kwargs...)
end
