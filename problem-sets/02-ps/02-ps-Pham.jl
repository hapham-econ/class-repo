using Optim
using JuMP
using Distributions
using LinearAlgebra

### Problem 2
# 20000 x 20000 random number
x = rand(20000,20000)

# exponentiate by column
function exp_cols(x)
    
end

# exponentiate by row
function exp_rows(x)
    
end

# exponentiate full matrix
function exp_all(x)
    
end

exp_cols(x)
exp_rows(s)
exp_all(x)

@time exp_cols(x)
@time exp_rows(x)
@time exp_all(x)

### Problem 3
# solve Cournot by Newton
function cournot_newton(eta, c1, c2, initial_guess, tolerance)
    foc_1 = 
    foc_2 = 

    # loop to find q1 and q2 such that foc_1 and foc_2 are both 0
    
    return q1,q2
end

# solve suppose q1 and q2 are constrained, solve by PATH algorithm as in some electricity paper?


### Problem 4
function mle_newton(x, y, theta, tol)
    # define log likelihood

    # maximize to get central estimate

    # boostrap to get std error

end

### Problem 5
function utility_maximizer(kappa, eta, p1, p2, w, initial_guess, solver)

end

utility_maximizer()