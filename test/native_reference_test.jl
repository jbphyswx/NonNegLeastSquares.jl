using LinearAlgebra: LinearAlgebra as LA
using SparseArrays: sparse

@testset "Reference helper checks" begin
    A = Matrix{Float64}(LA.I,3,3)
    b = [-2.0,3,0]
    x,objective = exhaustive_nnls(A,b)
    @test x == [0,3,0]
    @test objective == 4
    @test_throws ArgumentError exhaustive_nnls(zeros(2,11),zeros(2))
    for fixture in (test_case1,test_case2)
        A,b,rounded = fixture()
        x,objective = exhaustive_nnls(A,b)
        @test x ≈ rounded atol=1e-7
        assert_kkt(A,b,x)
    end
end

@testset "Exact optimality certificates" begin
    Q = Rational{BigInt}
    Aq = Q.([2 1 0; 1 3 1; 0 1 2; 1 0 1])
    xq = Q.([3,0,2])
    for gq in (Q.([0,5,0]),zeros(Q,3))
        bq = Aq*xq - Aq*((Aq'*Aq)\gq)
        @test Aq'*(Aq*xq-bq) == gq
        @test all(>=(0),xq) && all(>=(0),gq) && iszero(LA.dot(xq,gq))
        for T in (Float32,Float64), scale in (0.01,1,1e6), f in algs
            A,b,expected = T.(Aq),T.(bq)*T(scale),T.(xq)*T(scale)
            x = vec(f(A,b))
            accuracy = T === Float32 ? 1e-4 : 1e-8
            @test x ≈ expected atol=accuracy*scale rtol=accuracy
            assert_kkt(A,b,x;rtol=accuracy,atol=accuracy*scale)
        end
    end
end

@testset "Nonunique and zero-column optima" begin
    matrices = ([1.0 1 0; 0 0 1; 1 1 1; 2 2 1],
                [1.0 0 0; 0 0 1; 1 0 1; 2 0 1])
    for A in matrices
        b = A*[1.0,0,2]
        _,objective = exhaustive_nnls(A,b)
        @test objective <= 1e-20
        for f in algs
            x = vec(f(A,b))
            assert_kkt(A,b,x)
            @test A*x ≈ b atol=1e-9
            perm = [3,1,2]
            xp = vec(f(A[:,perm],b))
            assert_kkt(A[:,perm],b,xp)
            @test A[:,perm]*xp ≈ b atol=1e-9
        end
    end
end

@testset "Analytic boundary and zero RHS" begin
    A = Matrix{Float64}(LA.I,3,3)
    B = [-2.0 0; 3 0; 0 0]
    expected = [0.0 0; 3 0; 0 0]
    for f in algs
        X = f(A,B)
        @test X ≈ expected
        for j in axes(B,2)
            assert_kkt(A,B[:,j],X[:,j])
        end
    end
end

@testset "Structured and sparse certificate" begin
    Q = Rational{BigInt}
    Aq = Q.([3 -1 0; -1 3 -1; 0 -1 3])
    xq,gq = Q.([1,0,2]),Q.([0,1,0])
    bq = Aq*xq - Aq'\gq
    @test Aq'*(Aq*xq-bq) == gq
    A = LA.SymTridiagonal(fill(3.0,3),fill(-1.0,2))
    b,expected = Float64.(bq),Float64.(xq)
    for data in (A,Matrix(A),sparse(A)), f in (nnls,fnnls,pivot,pivot_cache)
        x = vec(f(data,b))
        @test x ≈ expected atol=1e-9
        assert_kkt(data,b,x)
    end
end
