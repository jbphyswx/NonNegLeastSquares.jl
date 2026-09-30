module InterfaceTests

using Test, LinearAlgebra
using NonNegLeastSquares
import NonNegLeastSquares as NN
using ..NNLSTestUtils: assert_kkt

# These types belong to a downstream module, with no additions to symbol routing.
struct CustomAlgorithm <: AbstractNNLSAlgorithm end
struct CustomPivot{T} <: AbstractPivotVariant
    tol::T
end
struct WrappedProblem{P} <: AbstractNNLSProblem
    inner::P
end
struct WrappedMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::M
end
Base.size(A::WrappedMatrix) = size(A.data)
Base.getindex(A::WrappedMatrix, i::Int, j::Int) = A.data[i,j]

NN.solve_nnls(p::NNLSData, ::CustomAlgorithm; kwargs...) =
    solve_nnls(p, LawsonHanson(); kwargs...)
NN.solve_pivot(p::NNLSData, v::CustomPivot; kwargs...) =
    solve_pivot(p, DirectPivot(); tol=v.tol, kwargs...)
NN.solve_pivot(p::WrappedProblem, v::DirectPivot; kwargs...) =
    solve_pivot(p.inner, v; kwargs...)
NN.solve_pivot(p::NNLSData{<:WrappedMatrix}, v::DirectPivot; kwargs...) =
    solve_pivot(NNLSData(p.A.data, p.B), v; kwargs...)

struct MissingAlgorithm <: AbstractNNLSAlgorithm end
struct MissingPivot <: AbstractPivotVariant end
struct FailingAlgorithm <: AbstractNNLSAlgorithm end
const extension_error = ErrorException("downstream solver failure")
NN.solve_nnls(::NNLSData, ::FailingAlgorithm; kwargs...) = throw(extension_error)

@testset "Typed and legacy algorithms" begin
    choices = ((LawsonHanson(), :nnls, :none, NN.nnls),
               (FastNNLS(), :fnnls, :none, NN.fnnls),
               (DeviationMaximization(), :lhdm, :none, NN.lhdm),
               (Pivot(), :pivot, :none, NN.pivot),
               (Pivot(CachedPivot()), :pivot, :cache, NN.pivot_cache),
               (Pivot(GroupedPivot()), :pivot, :comb, NN.pivot_comb))
    for T in (Float32, Float64), matrix_rhs in (false, true)
        A = T[1 1; 0 1; 1 0]
        B = matrix_rhs ? T[-1 2; 2 0; 3 -1] : T[-1, 2, 3]
        A0, B0 = copy(A), copy(B)
        p = NNLSData(A,B)
        @test p.A === A && p.B === B
        for (alg, symbol, variant, kernel) in choices
            for parallel in (false, true)
                options = alg isa Pivot{GroupedPivot} ? (;) : (; use_parallel=parallel)
                X = solve_nnls(p, alg; options...)
                @test X == kernel(A, reshape(B,size(A,1),:); options...)
                @test X == nonneg_lsq(A,B; alg, options...)
                @test X == nonneg_lsq(A,B,alg; options...)
                @test X == nonneg_lsq(A,B; alg=symbol, variant, options...)
                @test size(X) == (2, matrix_rhs ? 2 : 1)
                @test eltype(X) === T
                for j in axes(X,2)
                    assert_kkt(A,reshape(B,size(A,1),:)[:,j],X[:,j];
                               rtol=T === Float32 ? 1e-5 : 1e-8)
                end
                @test A == A0 && B == B0
            end
        end
    end
end

@testset "Gram routing" begin
    A = [1.0 1; 0 1; 1 0]
    for B in ([1.0,2,-1], [1.0 2; 2 0; -1 3])
        G,C = A'*A,A'*B
        G0,C0 = copy(G),copy(C)
        p = NNLSGram(G,C)
        @test p.G === G && p.C === C
        for alg in (FastNNLS(), Pivot(CachedPivot())), parallel in (false,true)
            X = solve_nnls(p,alg;use_parallel=parallel)
            @test X ≈ nonneg_lsq(A,B;alg,use_parallel=parallel)
            @test X == nonneg_lsq(G,C;alg,gram=true,use_parallel=parallel)
            @test X == nonneg_lsq(G,C,alg;gram=true,use_parallel=parallel)
            @test X == nonneg_lsq(G,C,alg,true;use_parallel=parallel)
            @test size(X) == (2, size(B,2))
            for j in axes(X,2)
                assert_kkt(A,reshape(B,size(A,1),:)[:,j],X[:,j])
            end
        end
        expected = solve_nnls(p,Pivot(CachedPivot()))
        @test nonneg_lsq(G,C;gram=true) ≈ expected
        for variant in (:none,:cache)
            @test nonneg_lsq(G,C;alg=:pivot,variant,gram=true) ≈ expected
        end
        for alg in (:nnls,:fnnls)
            @test nonneg_lsq(G,C;alg,gram=true) ≈ expected
        end
        @test G == G0 && C == C0
    end
end

@testset "Unsupported selections" begin
    A,b = Matrix{Float64}(I,2,2),ones(2)
    for alg in (LawsonHanson(), DeviationMaximization(), Pivot(), Pivot(GroupedPivot()))
        @test_throws ArgumentError solve_nnls(NNLSGram(A,b),alg)
        @test_throws ArgumentError nonneg_lsq(A,b;alg,gram=true)
    end
    @test_throws ArgumentError nonneg_lsq(A,b;alg=:pivot,variant=:comb,gram=true)
    for alg in (:unknown,:admm,:pivot_cache)
        @test_throws ArgumentError nonneg_lsq(A,b;alg)
    end
    @test_throws ArgumentError nonneg_lsq(A,b;variant=:unknown)
    for alg in (:nnls,:fnnls,:lhdm), variant in (:cache,:comb)
        @test_throws ArgumentError nonneg_lsq(A,b;alg,variant)
    end
    for (alg, matching) in ((Pivot(), :none), (Pivot(CachedPivot()), :cache),
                            (Pivot(GroupedPivot()), :comb), (LawsonHanson(), :none),
                            (FastNNLS(), :none), (DeviationMaximization(), :none))
        @test nonneg_lsq(A,b;alg,variant=matching) == ones(2,1)
        for variant in (:none,:cache,:comb)
            variant === matching && continue
            @test_throws ArgumentError nonneg_lsq(A,b;alg,variant)
        end
    end
    @test_throws ArgumentError solve_nnls(NNLSData(A,b),MissingAlgorithm())
    @test_throws ArgumentError solve_nnls(NNLSData(A,b),Pivot(MissingPivot()))
end

@testset "Keyword forwarding" begin
    A,b = Matrix{Float64}(I,2,2),ones(2)
    for alg in (Pivot(),Pivot(CachedPivot()),Pivot(GroupedPivot()))
        @test nonneg_lsq(A,-b;alg,max_iter=0,tol=0.0,rtol=0.0) == zeros(2,1)
        @test_throws ErrorException nonneg_lsq(A,b;alg,max_iter=0)
        @test nonneg_lsq(A,b;alg,max_iter=1) == ones(2,1)
        @test nonneg_lsq(A,b;alg,max_iter=0,tol=2.0,rtol=0.0) == zeros(2,1)
        @test nonneg_lsq(A,b*1e-12;alg,tol=0.0,rtol=0.0) ≈ fill(1e-12,2,1)
    end
    # Only grouped pivoting currently accepts a warm-start mask.
    P = trues(2,1)
    alg = Pivot(GroupedPivot())
    @test nonneg_lsq(A,b;alg,max_iter=0,P! = P) == ones(2,1)
    fill!(P,false)
    @test nonneg_lsq(A,b;alg,P! = P) == ones(2,1)
    @test all(P)
end

@testset "Downstream dispatch" begin
    A,b = [1.0 1; 0 1],[1.0,2]
    expected = reshape([0.0,1.5],:,1)
    @test (@inferred solve_nnls(NNLSData(A,b),Pivot())) ≈ expected
    @test nonneg_lsq(A,b;alg=CustomAlgorithm()) ≈ expected
    @test nonneg_lsq(A,b,CustomAlgorithm()) ≈ expected
    @test nonneg_lsq(A,b;alg=Pivot(CustomPivot(0.0)),rtol=0.0) ≈ expected
    @test solve_nnls(WrappedProblem(NNLSData(A,b)),Pivot()) ≈ expected
    @test nonneg_lsq(WrappedMatrix(A),b;alg=Pivot()) ≈ expected
    @test_throws extension_error nonneg_lsq(A,b;alg=FailingAlgorithm())
    @test isempty(Test.detect_ambiguities(NN,InterfaceTests;recursive=false))
end

end
