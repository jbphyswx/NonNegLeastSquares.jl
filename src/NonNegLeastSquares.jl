"""
    NonNegLeastSquares
Nonnegative least squares
"""
module NonNegLeastSquares

using LinearAlgebra
import SparseArrays

export nonneg_lsq
export AbstractNNLSAlgorithm, AbstractPivotVariant, AbstractNNLSProblem
export Pivot, DirectPivot, CachedPivot, GroupedPivot
export LawsonHanson, FastNNLS, DeviationMaximization
export NNLSData, NNLSGram, solve_nnls, solve_pivot

include("algorithms.jl")
include("problems.jl")

## Algorithms
include("nnls.jl")
using .NNLS: nnls
include("fnnls.jl")
include("pivot.jl")
include("pivot_comb.jl")
include("pivot_cache.jl")
include("admm.jl")
include("lhdm.jl")
using .LHDM: lhdm
## Common interface to algorithms
include("dispatch.jl")
include("interface.jl")

## Helper functions
include("cssls.jl") # combinatorial subspace least squares (CSSLS)

end
