"""
x = pivot(A, b; ...)

Solves non-negative least-squares problem by block principal pivoting method
(Algorithm 1) described in Kim & Park (2011).

Optional arguments:
    tol: absolute tolerance for nonnegativity constraints (default 1e-8)
    rtol: relative roundoff allowance (default 8*max(size(A)...,1)*eps(T))
    max_iter: maximum number of pivot passes; throws an error on exhaustion

The primal threshold is `tol + rtol*norm(x[P], Inf)`. The dual threshold is
`tol + rtol*opnorm(A,1)*(opnorm(A,Inf)*norm(x[P],Inf) + norm(b,Inf))`.
Set `rtol=0` for absolute-only tolerances. `max_iter=0` permits only an
already feasible initial solution. The relative allowance addresses roundoff,
not loss of accuracy from ill-conditioning.

References:
    J. Kim and H. Park, Fast nonnegative matrix factorization: An
    active-set-like method and comparisons, SIAM J. Sci. Comput., 33 (2011),
    pp. 3261–3281.
"""
function pivot(A,
               b::AbstractVector{T};
               tol::Float64=1e-8,
               rtol::Real=_pivot_default_rtol(A, b),
               max_iter=30*size(A,2)) where T

    _pivot_check_options(tol, rtol, max_iter)

    # dimensions, initialize solution
    q = size(A,2)

    x = zeros(T, q) # primal variables
    y = -A'*b    # dual variables

    # parameters for swapping
    α = 3
    β = q+1

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[~P]== 0, Y[~P] >= 0
    P = falses(q)
    inactive = trues(q)

    # Scale primal and dual tolerances separately; their units differ.
    aone = isempty(A) ? zero(tol) : opnorm(A, 1)
    ainf = isempty(A) ? zero(tol) : opnorm(A, Inf)
    binf = norm(b, Inf)
    (; tolx, toly) = _pivot_tolerances(x, P, aone, ainf, binf, tol, rtol)

    # identify indices of infeasible variables
    V = @. (P & (x < -tolx)) | (!P & (y < -toly))
    nV = sum(V)

    # while infeasible (number of infeasible variables > 0)
    iter = 0
    while nV > 0
        iter >= max_iter && error("pivot failed to converge within max_iter=$max_iter pivot passes")
        iter += 1

        if nV < β
            # infeasible variables decreased
            β = nV  # store number of infeasible variables
            α = 3   # reset α
        else
            # infeasible variables stayed the same or increased
            if α >= 1
                α = α-1 # tolerate increases for α cycles
            else
                # backup rule
                i = findlast(V)
                if i !== nothing
                    fill!(V, false)
                    V[i] = true
                else
                    error("V had no true values")
                end
            end
        end

        # update passive set
        #     P & ~V removes infeasible variables from P
        #     V & ~P  moves infeasible variables in ~P to P
        @. P = (P & !V) | (V & !P)
        @. inactive = !P

        # update primal/dual variables
        if !all(!, P)
            x[P] =  A[:,P] \ b
        end
        y[inactive] =  A[:,inactive]' * ((A[:,P]*x[P]) - b)

        # check infeasibility
        (; tolx, toly) = _pivot_tolerances(x, P, aone, ainf, binf, tol, rtol)
        @. V = (P & (x < -tolx)) | (!P & (y < -toly))
        nV = sum(V)
    end

    x[inactive] .= zero(eltype(x))
    return x
end

# Shared by the pivot variants. Use the arithmetic precision, not a fixed
# Float64 epsilon, and allow a dimension-dependent margin for roundoff.
function _pivot_default_rtol(A, B)
    T = float(promote_type(eltype(A), eltype(B)))
    return 8 * max(size(A)..., 1) * eps(real(T))
end

function _pivot_check_options(tol, rtol, max_iter::Integer)
    (isfinite(tol) && (tol >= 0)) || throw(ArgumentError("tol must be finite and nonnegative"))
    (isfinite(rtol) && (rtol >= 0)) || throw(ArgumentError("rtol must be finite and nonnegative"))
    (max_iter >= 0) || throw(ArgumentError("max_iter must be a nonnegative integer"))
    return nothing
end

function _pivot_tolerances(x, P, aone, ainf, binf, tol, rtol)
    iszero(rtol) && return (; tolx=tol, toly=tol)
    xscale = zero(real(eltype(x)))
    # Inactive entries can still hold values from an earlier passive solve.
    for i in eachindex(x, P)
        P[i] && (xscale = max(xscale, abs(x[i])))
    end
    tolx = tol + (rtol * xscale)
    toly = tol + (rtol * aone * (ainf * xscale + binf))
    isfinite(tolx) && isfinite(toly) || error("nonfinite pivot tolerance scale")
    return (; tolx, toly)
end


## if multiple right hand sides are provided, solve each problem separately.
function pivot(A,
               B::AbstractMatrix{T};
               use_parallel = true,
               kwargs...) where {T}

    n = size(A,2)
    k = size(B,2)

    # compute result for each column
    X = Array{T}(undef,n,k)
    if use_parallel && k > 1 && Threads.nthreads() > 1
        Threads.@threads for i = 1:k
            X[:,i] = pivot(A, view(B,:,i); kwargs...)
        end
    else
        for i = 1:k
            X[:,i] = pivot(A, view(B,:,i); kwargs...)
        end
    end

    return X
end
