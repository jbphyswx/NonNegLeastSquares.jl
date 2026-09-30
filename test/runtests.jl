using Test # @test, @testset
using LinearAlgebra: norm, I
using NonNegLeastSquares: nonneg_lsq
using SparseArrays: sprand

#test specific
using Random #: seed!
include("test_helpers.jl")
using .NNLSTestUtils: assert_kkt, exhaustive_nnls

"""
Measure memory allocation within a function to avoid issues
with global variables.
"""
macro wrappedallocs(expr)
    argnames = [gensym() for a in expr.args]
    quote
        function g($(argnames...))
            @allocated $(Expr(expr.head, argnames...))
        end
        $(Expr(:call, :g, [esc(a) for a in expr.args]...))
    end
end

function test_case1()
    A = [ 0.53879488  0.65816267
          0.12873446  0.98669198
          0.24555042  0.00598804
          0.80491791  0.32793762 ]

    b = [0.888, 0.562, 0.255, 0.077]
    x = [0.15512102, 0.69328985] # rounded reference, checked by the native oracle
    return A, b, x
end

function test_case2()
    A = [ -0.24  -0.82   1.35   0.36   0.35
          -0.53  -0.20  -0.76   0.98  -0.54
           0.22   1.25  -1.60  -1.37  -1.94
          -0.51  -0.56  -0.08   0.96   0.46
           0.48  -2.25   0.38   0.06  -1.29 ]
    b = [-1.6, 0.19, 0.17, 0.31, -1.27]
    x = [2.2010416, 1.19009924, 0.0, 1.55001345, 0.0]
    return A, b, x
end

function test_case3() # non-float
    A = ones(Int, 4, 3)
    b = 2*ones(Int, 4)
    x = [2,0,0]
    return A, b, x
end

function test_algorithm(fh::Function, cases, ε::Real=1e-5; use_parallel=false)
    # Solve A*x = b for x, subject to x >=0
    A, b, x = test_case1()
    @test norm(fh(A,b) - x) < ε

    A, b, x = test_case2()
    @test norm(fh(A,b) - x) < ε

    for (A,B,objectives) in cases
        # Exercise vector and matrix entry points on the same random inputs.
        x = vec(fh(A,B[:,1];use_parallel))
        X = fh(A,B[:,2:3];use_parallel)
        @test size(X) == (size(A,2),2)
        for (j,solution) in enumerate((x,X[:,1],X[:,2]))
            assert_kkt(A,B[:,j],solution)
            if objectives !== nothing
                @test sum(abs2,A*solution-B[:,j]) ≈ objectives[j] atol=1e-9 rtol=1e-8
            end
        end
    end
end

nnls(A,b; use_parallel=false) = nonneg_lsq(A, b; alg=:nnls, use_parallel)
nnls_gram(A,b; use_parallel=false) = nonneg_lsq(A'*A, A'*b; alg=:nnls, gram=true, use_parallel)
fnnls(A,b; use_parallel=false) = nonneg_lsq(A, b; alg=:fnnls, use_parallel)
fnnls_gram(A,b; use_parallel=false) = nonneg_lsq(A'*A, A'*b; alg=:fnnls, gram=true, use_parallel)
pivot(A,b; use_parallel=false) = nonneg_lsq(A, b; alg=:pivot, use_parallel)
pivot_comb(A,b; use_parallel=nothing) = nonneg_lsq(A, b; alg=:pivot, variant=:comb)  # doesn't support `use_parallel`
pivot_cache(A,b; use_parallel=false) = nonneg_lsq(A, b; alg=:pivot, variant=:cache, use_parallel)
lhdm(A,b; use_parallel=false) = nonneg_lsq(A, b; alg=:lhdm, use_parallel)

algs = [nnls, nnls_gram, fnnls, fnnls_gram, pivot, pivot_comb, pivot_cache, lhdm]
errs = fill(1e-5, length(algs))

# Generate once so algorithms and threaded/nonthreaded calls see identical data.
rng = MersenneTwister(51)
cases = map(1:100) do _
    m,n = rand(rng,1:10),rand(rng,1:10)
    A,B = randn(rng,m,n),randn(rng,m,3)
    objectives = n <= 6 ? [exhaustive_nnls(A,B[:,j])[2] for j in 1:3] : nothing
    (A,B,objectives)
end

for use_parallel in (false, true)
    @show use_parallel
    for (f, ε) in zip(algs, errs)
        print("testing ")
        @show f
        @testset "$(nameof(f)), parallel=$use_parallel" begin
            test_algorithm(f, cases, ε; use_parallel)
        end
        println("done")
    end
end

#= non-float test fails, so revisit later
@testset "pivot_cache-non-float" begin
    A, b, x = test_case3()
    xi = pivot_cache(A, b)
    xf = pivot_cache(Float32.(A), Float32.(b))
    @test xi ≈ xf
end
=#

@testset "comb" begin
    A, b, x = test_case2()
    x0 = pivot_comb(A, b)
    P! = falses(size(A,2),1)
    x1 = nonneg_lsq(A, b; alg=:pivot, variant=:comb, P! = P!)
    @test x0 == x1
    @test P! != falses(size(A,2),1)
    @test all(x0[(!).(P!)] .== 0)
end

@testset "NNLS" begin include("nnls_test.jl") end
@testset "Native references" begin include("native_reference_test.jl") end
@testset "FNNLS" begin include("fnnls_test.jl") end
@testset "Pivot" begin include("pivot_test.jl") end
@testset "Pivot termination" begin include("pivot_termination_test.jl") end
@testset "Pivot buffers" begin include("pivot_buffers_test.jl") end
@testset "Sparse" begin include("sparse_test.jl") end
@testset "LHDM" begin include("lhdm_test.jl") end
