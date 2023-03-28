using Optim
using Distributions
using LinearAlgebra
using ForwardDiff
using NLsolve
using Plots
using JuMP

### Problem 1
# 20000 x 20000 random number
x = rand(20000,20000)

# exponentiate by column
function exp_cols(x)
    col = [exp.(x[:, index]) for index in 1:size(x)[2]]
    result = hcat(col...)
    return result
end

# exponentiate by row
function exp_rows(x)
    row = [exp.(x[index, :]) for index in 1:size(x)[1]]
    result = hcat(row...)
    return result
end

# exponentiate full matrix
function exp_all(x)
    return exp.(x)
end

exp_cols(x)
exp_rows(x)
exp_all(x)
@time exp_cols(x)
# 13.842118 seconds (80.01 k allocations: 8.943 GiB, 63.19% gc time)
@time exp_rows(x)
# 63.305376 seconds (80.01 k allocations: 8.943 GiB, 7.88% gc time)
@time exp_all(x)
# 23.682283 seconds (2 allocations: 2.980 GiB, 41.02% gc time)

# strange, column is the fastest!

### Problem 2
# solve Cournot by Newton
function cournot_newton(eta, c1, c2, initial_guess, tolerance)
    foc_1(q1,q2) = (-1/eta) * (q1 + q2)^(-(1/eta) - 1) * q1 + (q1 + q2)^(-(1/eta)) - 2*c1*q1 
    foc_2(q1,q2) = (-1/eta) * (q1 + q2)^(-(1/eta) - 1) * q2 + (q1 + q2)^(-(1/eta)) - 2*c2*q2 
    foc(q1, q2) = [foc_1(q1, q2), foc_2(q1, q2)]

    # loop to find q1 and q2 such that foc_1 and foc_2 are both 0
    q = initial_guess
    q_new = 1000
    iter = 1
    error = 1e3
    while error > tolerance
        value = foc(q[1], q[2])
        jacobian = ForwardDiff.jacobian(q -> foc(q[1], q[2]), q)
        q_new = q - inv(jacobian) * value
        error = maximum(abs.(q_new .- q))
        q = q_new
        println("new guess is $q_new, error is $error")
        iter += 1
        if iter >= 300000
            println("too many iterations")
            break
        end

    end

    if iter >= 300000
        println("fail to reach fixed point")
    else
        q1 = q[1]
        q2 = q[2]
        println("Solved! q1 is $q1, q2 is $q2")
    end

    return q
end

# solve the model
eta = 1.6
c1 = 0.15
c2 = 0.2
res = cournot_newton(eta, c1, c2, [1.0, 1.0], 1e-3)

# plot the profit functions to make sure im not crazy
x = range(0.1, 5, length = 100)
f(x) = (x + res[2])^(-1/eta) * x - 0.15*(x^2)
f2(x) = (x + res[1])^(-1/eta) * x - 0.2*(x^2)
p1 = plot(x, f)
vline!(p1, [res[1]])
p2 = plot(x, f2)
vline!(p2, [res[2]])
plot(p1, p2, layout = (2, 1))

### Problem 3
function mle_center(X::Matrix, Y::Vector, guesses::Vector, tol::Float64)
    # define log likelihood
    residual(θ) = Y - X*θ

    # define a function to return the distribution with some standard deviation

    # maximize to get central estimate
    likelihood(θ) = -sum(log.(pdf.(Normal(), residual(θ))))
    result = optimize(θ -> likelihood(θ), guesses, f_tol=tol)

    return vec(result.minimizer)
end

function mle_std(X::Matrix, Y::Vector, guesses::Vector, tol::Float64)
    sample_std = vec(rand(1, size(guesses)[1]))
    for i in 1:100
        n_obs = Int(round(0.5*size(X)[1]))
        sample_rows = sample(axes(X, 1), n_obs)
        sample_x = X[sample_rows, :]
        sample_y = vec(Y[sample_rows, :])
        coef = mle_center(sample_x, sample_y, guesses, tol)
        sample_std = hcat(sample_std, coef)
    end

    sample_std = sample_std[:, 1:end .!= 1]
    result = [std(sample_std[index, :]) for index in 1:size(sample_std)[1]]
    return result
end

X = hcat(ones(100), randn(100, 1))
θ = [7.0, 8.0]
ϵ = randn(100)
Y = X*θ + ϵ

function mle_newton(X::Matrix, Y::Vector, guesses::Vector, tol::Float64)
    center =  mle_center(X::Matrix, Y::Vector, guesses::Vector, tol::Float64)
    boost_std = mle_std(X::Matrix, Y::Vector, guesses::Vector, tol::Float64)
    result = center, boost_std
    println("Estimated θ is $center and boostrapped standard error is $boost_std")
    return result
end

mle_newton(X, Y, [10.0, 10.0], 1e-5)

### Problem 5
function utility_maximizer(kappa, eta, p1, p2, w, initial_guess)
    f(c1) = (kappa*exp.(c1[1])^(1-1/eta) + (1-kappa)*((w-p1*exp.(c1[1]))/p2)^(1- 1/eta))^(eta/(eta - 1)) 
    result = optimize(c1 -> -f(c1), initial_guess)
    c1 = exp.(result.minimizer[1])
    c2 = (w-p1*c1)/p2
    println("Optimal consumption is c1=$c1, c2=$c2")
    print(result)
    return c1,c2
end

kappa = 0.5
eta = 2
p1 = 10
p2 = 1
w = 110
utility_maximizer(kappa, eta, p1, p2, w, [-2.0])

# alternatively use JuMP, much cleaner...
using Ipopt
model = Model(Ipopt.Optimizer)
@variable(model, c1)
@variable(model, c2)
@NLobjective(model, Max, (kappa*c1^(1-1/eta) + (1-kappa)*c2^(1- 1/eta))^(eta/(eta - 1)))
@constraint(model, constraint1, p1*c1 + p2*c2 <= 110)
@constraint(model, c1 >= 0)
@constraint(model, c2 >= 0)
optimize!(model)
value(c1)
value(c2)