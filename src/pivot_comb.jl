"""
    x = pivot_comb(A, b; ...)

Solves non-negative least-squares problem by block principal pivoting method
with combinatorial grouping of least-squares problems. Algorithm 2 described
in Kim & Park (2011).

Optional arguments:
* `tol`: tolerance for nonnegativity constraints; default: `1e-8`
* `rtol`: relative roundoff allowance; default: `8*max(size(A)...,1)*eps(T)`.
 Uses separate primal/dual thresholds as in `pivot_cache`, independently for
 each RHS column. Set `rtol=0` for absolute-only tolerances.
* `max_iter`: maximum number of batches of pivot swaps; default: `30*size(A,2)`.
 Throws an error on exhaustion. The initial warm-start solve is not a pivot
 pass, so an already feasible initial solution may return with `max_iter=0`.
* `P!`: initial guess of passive set; default: `falses(size(A,2), size(B,2))`
Alert: this optional argument, if provided, is mutated to become the final passive set.

References:
    J. Kim and H. Park, Fast nonnegative matrix factorization: An
    active-set-like method and comparisons, SIAM J. Sci. Comput., 33 (2011),
    pp. 3261–3281.
"""
function pivot_comb(
    A,
    B::AbstractMatrix{T};
    tol::Float64=1e-8,
    rtol::Real=_pivot_default_rtol(A, B),
    max_iter=30*size(A,2),
    P!::AbstractMatrix{Bool} = falses(size(A,2), size(B,2)),
) where {T}
    _pivot_check_options(tol, rtol, max_iter)

    # precompute constant portion of pseudoinverse
    AtA = A'*A
    AtB = A'*B

    # dimensions, initialize solution
    q,r = size(AtB)
    X = zeros(T, q,r) # primal variables
#   Y = -AtB       # dual variables

    # parameters for swapping
    α = ones(r)*3
    β = ones(r)*(q+1)

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[~P]== 0, Y[~P] >= 0
#   P = zeros(Bool,q,r) # initialized above

    # Update primal and dual variables
    cssls!(AtA, AtB, X, P!) # overwrite X[P]
    Y = AtA*X - AtB

    # identify infeasible columns of X
    infeasible_cols = Array{Bool}(undef,size(X,2))

    ginf = isempty(AtA) ? zero(tol) : opnorm(AtA, Inf)
    cinf = [norm(view(AtB,:,j), Inf) for j in 1:r]
    # Row matrices broadcast one threshold per RHS, not one for the whole batch.
    tolx = zeros(promote_type(T, typeof(tol), typeof(rtol)), 1, r)
    toly = similar(tolx)
    for j in 1:r
        tolx[j], toly[j] = _pivot_tolerances(view(X,:,j), view(P!,:,j),
            one(ginf), ginf, cinf[j], tol, rtol)
    end
    V = @. (P! & (X < -tolx)) | (!(P!) & (Y < -toly)) # infeasible variables
    any!(infeasible_cols, V') # collapse each column

    # while infeasible
    iter = 0
    while any(infeasible_cols)
        iter >= max_iter && error("pivot_comb failed to converge within max_iter=$max_iter pivot passes")
        iter += 1

        # check progress
        for j = 1:r
            nV = sum(V[:,j])

            # skip any column with no infeasible variables
            if nV == 0
                continue
            end

            if nV < β[j]
                # infeasible variables decreased for column j
                β[j] = nV  # store number of infeasible variables
                α[j] = 3   # reset α
            else
                # infeasible variables stayed the same or increased
                if α[j] >= 1
                    α[j] = α[j]-1 # tolerate increases for α cycles
                else
                    # backup rule
                    i = findlast(V[:,j])
                    V[:,j] = zeros(Bool,q)
                    V[i,j] = true
                end
            end
        end

        # Update passive set
        #     P & ~V removes infeasible variables from P
        #     V & ~P moves infeasible variables to the
        @. P! = (P! & !V) | (V & !(P!))

        # Update primal and dual variables
        cssls!(AtA, AtB, X, P!) # overwrite X[P]
        X[(!).(P!)] .= 0.0
        Y[:,infeasible_cols] = AtA*X[:,infeasible_cols] - AtB[:,infeasible_cols]
        Y[P!] .= 0.0

        # identify infeasible columns of X
        @inbounds for j in 1:r
            tolx[j], toly[j] = _pivot_tolerances(view(X,:,j), view(P!,:,j),
                one(ginf), ginf, cinf[j], tol, rtol)
        end
        @. V = (P! & (X < -tolx)) | (!(P!) & (Y < -toly)) # infeasible variables
        any!(infeasible_cols, V') # collapse each column
    end

    X[(!).(P!)] .= 0.0
    return X
end
