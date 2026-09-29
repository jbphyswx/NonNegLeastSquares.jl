using Test: Test
using LinearAlgebra: LinearAlgebra as LA
using SparseArrays: sparse
using NonNegLeastSquares: NonNegLeastSquares

Test.@testset "RHS views and input preservation" begin
    # Sparse Float32 solves already fail on this fixture in the baseline.
    for (T, sparse_input) in ((Float32,false), (Float64,false), (Float64,true)),
            variant in (:none, :cache)
        A = T[1 1; 0 1]
        data = sparse_input ? sparse(A) : A
        storage = T[1 -1 3; 99 99 99; 2 -1 1; 99 99 99]
        B = view(storage,1:2:4,:)
        expected = T[0 0 2; 1.5 0 1]
        for parallel in (false,true)
            saved_A, saved_storage = copy(data), copy(storage)
            X = NonNegLeastSquares.nonneg_lsq(data,B; alg=:pivot,variant,use_parallel=parallel)
            Test.@test X ≈ expected
            Test.@test data == saved_A
            Test.@test storage == saved_storage
        end
    end
end

Test.@testset "Partially converged batch" begin
    for T in (Float32, Float64)
        A = T[1 1; 0 1]
        # Middle column is feasible at initialization. Last is feasible with
        # the warm start. First requires removing a negative passive variable.
        B = T[1 -1 3; 2 -1 1]
        expected = T[0 0 2; 1.5 0 1]
        P = Bool[false false true; false false true]
        saved_A, saved_B = copy(A), copy(B)
        X = NonNegLeastSquares.nonneg_lsq(A,B; alg=:pivot,variant=:comb,P! = P,max_iter=2)
        Test.@test X ≈ expected
        Test.@test P == Bool[false false true; true false true]
        Test.@test A == saved_A && B == saved_B
        G = A'*(A*X-B)
        Test.@test minimum(G) >= -100*eps(T)
        Test.@test LA.norm(X.*G,Inf) <= 100*eps(T)
    end
end
