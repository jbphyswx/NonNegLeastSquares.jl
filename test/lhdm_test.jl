import NonNegLeastSquares.LHDM
using NonNegLeastSquares: DeviationMaximization, NNLSData, solve_nnls

@testset "Float32 and BigFloat" begin
    Random.seed!(5)
    for i in 1:100
        m = rand(1:10)
        n = rand(1:10)
        A = randn(m, n)
        b = randn(m)
        x1 = @inferred LHDM.lhdm(A, b)
        x2 = @inferred LHDM.lhdm(Float32.(A), Float32.(b))
        x3 = @inferred LHDM.lhdm(BigFloat.(A), BigFloat.(b))
        @test x1 ≈ x2
        @test x1 ≈ x3
    end
end

@testset "lhdm allocations" begin
    Random.seed!(101)
    for i in 1:50
        m = rand(20:100)
        n = rand(20:100)
        A = randn(m, n)
        b = randn(m)
        work = @inferred LHDM.LHDMWorkspace(A, b)
        @test @wrappedallocs(LHDM.lhdm!(work)) == 0
    end
end

@testset "lhdm workspace reuse" begin
    Random.seed!(200)
    m = 10
    n = 20
    work = @inferred LHDM.LHDMWorkspace(m, n)
    LHDM.lhdm!(work, randn(m, n), randn(m))
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        @test @wrappedallocs(LHDM.lhdm!(work, A, b)) == 0
        @test work.x ≈ LHDM.lhdm(A, b)
    end

    m = 20
    n = 10
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        LHDM.lhdm!(work, A, b)
        @test work.x ≈ LHDM.lhdm(A, b)
    end
end

@testset "non-Int Integer workspace" begin
    m = 10
    n = 20
    A = randn(m, n)
    b = randn(m)
    work = @inferred LHDM.LHDMWorkspace(A, b, Int32)
    # Compile
    LHDM.lhdm!(work)

    A = randn(m, n)
    b = randn(m)
    work = @inferred LHDM.LHDMWorkspace(A, b, Int32)
    @test @wrappedallocs(LHDM.lhdm!(work)) == 0
end

@testset "LHDM guaranteed positive exact solution and use_parallel" begin
    Random.seed!(101)
    m = 10
    n = 20
    for _ in 1:100
        A = randn(m, n)
        x = rand(n)
        b = A * x
        x_sparse = LHDM.lhdm(A, b)
        @test A * x_sparse ≈ b
    end
    k = 5
    for _ in 1:100
        A = randn(m, n)
        x = rand(n, k)
        b = A * x
        x_sparse = @inferred LHDM.lhdm(A, b, use_parallel=true)
        @test A * x_sparse ≈ b
    end
end

@testset "iteration limits" begin
    A = Matrix{Float64}(I,2,2)
    b = [2.0,1.0]
    options = (; kmax=1) # One variable enters per inner-loop iteration.

    @test LHDM.lhdm(A,b;max_iter=2,options...) ≈ b
    @test LHDM.lhdm(A,b,2;options...) ≈ b
    @test LHDM.lhdm(A,b;max_iter=Int32(2),options...) ≈ b
    for budget in (0,1)
        @test_throws ErrorException LHDM.lhdm(A,b;max_iter=budget,options...)
        @test_throws ErrorException LHDM.lhdm(A,b,budget;options...)
    end
    @test LHDM.lhdm(A,-b;max_iter=0,options...) == zeros(2)
    @test_throws ArgumentError LHDM.lhdm(A,b;max_iter=-1)
    @test_throws ArgumentError LHDM.lhdm(A,b,-1)

    # Workspace users retain the partial iterate and can inspect its status.
    work = LHDM.LHDMWorkspace(A,b)
    LHDM.lhdm!(work,1;options...)
    @test work.mode == 3
    @test work.x == [2,0]
    LHDM.lhdm!(work,A,b,2;options...)
    @test work.mode == 1
    @test work.x ≈ b
    @test_throws ArgumentError LHDM.lhdm!(work,-1)

    # Public entry points must forward the budget and surface exhaustion.
    solvers = ((A,B;kwargs...) -> nonneg_lsq(A,B;alg=:lhdm,kwargs...),
               (A,B;kwargs...) -> nonneg_lsq(A,B;alg=DeviationMaximization(),kwargs...),
               (A,B;kwargs...) -> solve_nnls(NNLSData(A,B),DeviationMaximization();kwargs...))
    for solve in solvers
        @test solve(A,b;max_iter=2,options...) ≈ reshape(b,2,1)
        @test_throws ErrorException solve(A,b;max_iter=1,options...)
        @test solve(A,-b;max_iter=0,options...) == zeros(2,1)
    end

    B = hcat(-b,zeros(2),b,2b)
    expected = hcat(zeros(2,2),b,2b)
    A0,B0 = copy(A),copy(B)
    for parallel in (false,true)
        @test LHDM.lhdm(A,B;max_iter=2,use_parallel=parallel,options...) ≈ expected
        @test LHDM.lhdm(A,B,2;use_parallel=parallel,options...) ≈ expected
        for solve in solvers
            @test solve(A,B;max_iter=2,use_parallel=parallel,options...) ≈ expected
            err = try
                solve(A,B;max_iter=1,use_parallel=parallel,options...)
                nothing
            catch e
                e
            end
            @test err isa (parallel && Threads.nthreads() > 1 ? TaskFailedException : ErrorException)
            @test occursin("max_iter=1",sprint(showerror,err))
        end
        @test A == A0 && B == B0

        # Validate before creating tasks, even when there are no RHS columns.
        for rhs in (B,zeros(2,0))
            @test_throws ArgumentError LHDM.lhdm(A,rhs;max_iter=-1,use_parallel=parallel)
            @test_throws ArgumentError LHDM.lhdm(A,rhs,-1;use_parallel=parallel)
        end
        @test size(LHDM.lhdm(A,zeros(2,0);max_iter=0,use_parallel=parallel)) == (2,0)
    end
end
