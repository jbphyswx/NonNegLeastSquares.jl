module NNLSTestUtils

using Test, LinearAlgebra

"""Check NNLS optimality against the original data, independently of the solver."""
function assert_kkt(A, b, result; rtol=1e-8, atol=1e-10)
    x = vec(result)
    @test length(x) == size(A,2)
    @test all(isfinite,x)
    residual = A*x-b
    gradient = A'*residual
    @test all(isfinite,residual)
    @test all(isfinite,gradient)
    xscale = max(norm(x,Inf),one(eltype(x)))
    gscale = opnorm(A,1)*(opnorm(A,Inf)*norm(x,Inf)+norm(b,Inf))
    xtol = atol+rtol*xscale
    gtol = atol+rtol*gscale
    @test minimum(x;init=zero(eltype(x))) >= -xtol
    @test minimum(gradient;init=zero(eltype(gradient))) >= -gtol
    @test maximum(abs,x.*gradient;init=zero(eltype(gradient))) <= xscale*gtol
    return nothing
end

"""Enumerate supports for small Float64 problems; return coefficients and objective."""
function exhaustive_nnls(A,b; tol=1e-10)
    n = size(A,2)
    n <= 10 || throw(ArgumentError("exhaustive reference is limited to 10 columns"))
    data, rhs = Matrix{Float64}(A), Vector{Float64}(b)
    best = zeros(n)
    bestobj = sum(abs2,rhs)
    for mask in 1:(2^n-1)
        indices = [j for j in 1:n if !iszero(mask & (1 << (j-1)))]
        M = data[:,indices]
        F = svd(M)
        cutoff = max(size(M)...)*eps(Float64)*maximum(F.S;init=0.0)
        z = F.U'*rhs
        for i in eachindex(F.S)
            z[i] = F.S[i] > cutoff ? z[i]/F.S[i] : 0.0
        end
        candidate = F.V*z
        minimum(candidate;init=0.0) >= -tol || continue
        candidate = max.(candidate,0.0)
        objective = sum(abs2,M*candidate-rhs)
        if objective < bestobj
            fill!(best,0.0)
            best[indices] = candidate
            bestobj = objective
        end
    end
    return best,bestobj
end

end
