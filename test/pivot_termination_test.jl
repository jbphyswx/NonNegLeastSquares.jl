using Test: Test
using LinearAlgebra: LinearAlgebra as LA
using NonNegLeastSquares: NonNegLeastSquares

Test.@testset "Pivot convergence" begin
    for variant in (:none, :cache, :comb)
        Test.@testset "$variant" begin
            # Keep exception checks serial; task failures can wrap the cause.
            solve(A, B; kwargs...) = variant === :comb ?
                NonNegLeastSquares.nonneg_lsq(A, B; alg=:pivot, variant, kwargs...) :
                NonNegLeastSquares.nonneg_lsq(A, B; alg=:pivot, variant, use_parallel=false, kwargs...)

            Test.@testset "scaled exact solution" begin
                for T in (Float32, Float64), amplitude in (1, 1e5, 1e8, 1e12)
                    A = Matrix(LA.SymTridiagonal(fill(T(0.75),7), fill(T(0.125),6)))
                    truth = zeros(T,7)
                    truth[5] = T(amplitude)
                    b = A*truth
                    accuracy = T === Float32 ? 2e-5 : 1e-12
                    X = solve(A,b)
                    Test.@test LA.norm(vec(X)-truth, Inf) <= accuracy*amplitude
                    Test.@test LA.norm(A*vec(X)-b, Inf) <= accuracy*LA.norm(b,Inf)
                    for rhs in (b, reshape(b,:,1)), atol in (1e-8, 1e-12)
                        X = solve(A,rhs; tol=atol, max_iter=10)
                        Test.@test size(X) == (7,1)
                        Test.@test all(isfinite,X)
                        Test.@test LA.norm(vec(X)-truth, Inf) <= accuracy*amplitude
                        Test.@test LA.norm(A*vec(X)-b, Inf) <= accuracy*LA.norm(b,Inf)
                        Test.@test minimum(X) >= -accuracy*amplitude
                    end
                end
            end

            Test.@testset "exact pass budget" begin
                A = [1.0 1; 0 1]
                b = [1.0,2]
                # Both columns enter first: [-1,2]; one leaves next: [0,1.5].
                for budget in (0,1)
                    err = try
                        solve(A,b; max_iter=budget)
                        nothing
                    catch e
                        e
                    end
                    Test.@test err isa ErrorException
                    Test.@test err isa ErrorException && occursin("max_iter=$budget", sprint(showerror,err))
                end
                Test.@test vec(solve(A,b; max_iter=2)) ≈ [0,1.5]
                Test.@test solve(A,zeros(2); max_iter=0) == zeros(2,1)
                Test.@test solve(A,[-1.0,-1]; max_iter=0) == zeros(2,1)
                Test.@test vec(solve(A,b; rtol=0, max_iter=2)) ≈ [0,1.5]
                Test.@test_throws ArgumentError solve(A,b; max_iter=-1)
                # The typed validation helper rejects nonintegers by dispatch.
                Test.@test_throws MethodError solve(A,b; max_iter=1.5)
                # Bool is an Integer in Julia; true is a budget of one pass.
                Test.@test_throws ErrorException solve(A,b; max_iter=true)
                for badtol in (-1.0, Inf, NaN)
                    Test.@test_throws ArgumentError solve(A,b; tol=badtol)
                    Test.@test_throws ArgumentError solve(A,b; rtol=badtol)
                end
            end

            Test.@testset "different scales within a batch" begin
                A = Matrix(LA.SymTridiagonal(fill(0.75,7),fill(0.125,6)))
                truth = zeros(7,3)
                truth[5,1] = 1e8
                truth[2,2] = 1
                truth[6,3] = 1e-5
                B = A*truth
                X = solve(A,B; tol=0.0, max_iter=10)
                for j in axes(B,2)
                    Test.@test LA.norm(X[:,j]-truth[:,j],Inf) <= 1e-12*LA.norm(truth[:,j],Inf)
                    Test.@test X[:,j] ≈ vec(solve(A,B[:,j]; tol=0.0,max_iter=10))
                end
                if variant !== :comb
                    Test.@test NonNegLeastSquares.nonneg_lsq(A,B; alg=:pivot,variant,use_parallel=true,
                        tol=0.0,max_iter=10) ≈ X
                end
            end
        end
    end

    Test.@testset "Gram kernel and grouped warm start" begin
        A = [1.0 1; 0 1]
        b = [1.0,2]
        Test.@test_throws ErrorException NonNegLeastSquares.pivot_cache(A'*A,A'*b; max_iter=1)
        Test.@test NonNegLeastSquares.pivot_cache(A'*A,A'*b; max_iter=2) ≈ [0,1.5]
        P = reshape([false,true],2,1)
        Test.@test vec(NonNegLeastSquares.nonneg_lsq(A,b; alg=:pivot,variant=:comb,P! = P,max_iter=0)) ≈ [0,1.5]
    end

    Test.@testset "strict tolerances remain bounded" begin
        A = Matrix(LA.SymTridiagonal(fill(0.75,7),fill(0.125,6)))
        b = A[:,5]*1e8
        # Depending on BLAS roundoff, this either converges or exhausts the
        # budget. Do not require a particular platform to reproduce the cycle.
        for variant in (:none,:cache,:comb)
            try
                X = NonNegLeastSquares.nonneg_lsq(A,b; alg=:pivot,variant,tol=0.0,rtol=0,max_iter=3)
                Test.@test all(isfinite,X)
                Test.@test LA.norm(A*vec(X)-b) <= 1e-6
            catch err
                Test.@test err isa ErrorException
                Test.@test occursin("max_iter=3", sprint(showerror,err))
            end
        end
    end
end
