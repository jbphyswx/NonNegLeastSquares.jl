"""Supertype for NNLS algorithms; extend [`solve_nnls`](@ref) for new algorithms."""
abstract type AbstractNNLSAlgorithm end

"""Supertype for pivot policies; extend [`solve_pivot`](@ref) for new policies."""
abstract type AbstractPivotVariant end

"""Direct block pivoting using restricted least-squares solves on the data matrix."""
struct DirectPivot <: AbstractPivotVariant end

"""Block pivoting using cached Gram matrices and restricted pseudoinverse solves."""
struct CachedPivot <: AbstractPivotVariant end

"""Block pivoting that groups RHS columns sharing a passive set."""
struct GroupedPivot <: AbstractPivotVariant end

"""
    Pivot(variant=DirectPivot())

Block principal pivoting with a policy of type `AbstractPivotVariant`.
Built-in policies are `DirectPivot()`, `CachedPivot()`, and `GroupedPivot()`.
"""
struct Pivot{V<:AbstractPivotVariant} <: AbstractNNLSAlgorithm
    variant::V
end
Pivot() = Pivot(DirectPivot())

"""Lawson–Hanson active-set NNLS using incremental orthogonal transformations."""
struct LawsonHanson <: AbstractNNLSAlgorithm end

"""Bro–De Jong fast NNLS using Gram matrices."""
struct FastNNLS <: AbstractNNLSAlgorithm end

"""Lawson–Hanson NNLS with deviation maximization (LHDM)."""
struct DeviationMaximization <: AbstractNNLSAlgorithm end
