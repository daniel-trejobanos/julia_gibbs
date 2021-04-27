using Distributions



function InvScChisq(rng, dof, scale)
    df*scale/rand(rng, Chisq(dof))
end

function exp_normalize_trick(pi, f, phi)
    log_prob = pi + f + phi
    log_prob .-= maximum(log_prob)
    exp.(log_prob)/sum(exp.(log_prob))
end


struct  spike_slab
    v0E::Real
    s02E::Real
    v0G::Real
    s02G::Real
    v0c::Real
    s02c::Real
    M::Int
    N::Int
end

struct mcmc_conf
    iter::Int
    thin::Int
    burnin::Int
end

struct linear_model
    y::RealVector
    X::Matrix
end

mutable struct linear_model_stats
    means::RealVector
    sds::RealVector
end


_exp(x::AbstractVecOrMat) = exp.(x .- maximum(x))
_exp(x::AbstractVecOrMat, theta::AbstractFloat) = exp.((x .- maximum(x) )*theta)
_sftmax(e::AbstractVecOrMat, d::Integer) = (e ./ sum(e,dims=d))

function softmax(X::AbstractVecOrMat{T}, dim::Integer)::AbstractVecOrMat where T <: AbstractFloat
    _sftmax(_exp(x), dim)
end

function softmax(X::AbstractVecOrMat{T}, dim::Integer, theta::Float64 )::AbstractVecOrMat where T <: AbstractFloat
    _sftmax(_exp(x, theta),dim)
end

function get_log_lik(log_pi, num, denom, sigmaG, sigmaE, components, Nm1)
     log_pi -  + 0.5 * (num^2)./(sigmaE*denom)
end

jsumnorm(A) = sum( abs2, A )

function jsimdnorm(A)
	∑= zero(eltype(A))
	@simd for i ∈ A
		∑ += i * i
	end
	∑
end

function javxnorm(A)
	∑= zero(eltype(A))
	@avx for i ∈ eachindex(A)
		∑ += A[i] * A[i]
	end
	∑
end

function bayesR!(rng, parameters::spike_slab, linear_obs::linear_model, epsilon, beta, sigmaE, sigmaG, pi)

    epsilon = zeros(float64, parameters.N)
    mu = rand(Normal(),1)
    epsilon.-= mu
    mixture = matrix()
    Nm1 = N - 1.0
    z = rand(Normal(), parameters.M)
    markerI = Vector(1:parameters.M)
    Random.shuffle!(rng, markerI)
    log_pi = log.(pi)
    prior_term = vcat( [0.0],0.5 * log.(sigmaG * components * Nm1 / sigmaE + 1))
    sE_sG = sigmaE / sigmaG
    for i = 1:M
        beta_old = beta[markerI]
        num = sum(x[markerI] .* (eps + beta_old*x[markerI]))
        denom = Nm1 +  sE_sG / components
        lik_term = vcat( [0.0], 0.5 * (num^2)./(sigmaE*denom))
        log_lik = log_pi - prior_term  + lik_term
        beta_sample = Categorical(softmax(log_lik))
        beta_nzero = num / denom + sqrt(sigmaE / denom )*z[markerI]
        beta_value = vcat([0.0], beta_nzero)
        beta[markerI] = beta_value[beta_sample]
        mixture[beta_sample] += 1
        epsilon .+= (beta_old - beta_new ) * x[markerI]
    end
    sigmaG = InvScChisq(parameters.v0G + m0,
                        (jsumnorm(beta) * m0 + parameters.v0G * parameters.s02G) /
                        (parameters.v0G + m0))
    pi = Dirichlet(1 + mixture)
    sigmaE = InvScChisq(parameters.v0E + parameters.N,
                        (jsumnorm(epsilon) + parametrs.v0E * parameters.s02E)/
                        (parameters.v0E + parameters.N))
end


rng = MersenneTwister(1234)
parameters = spike_slab(0.001, 0.001, 0.001, 0.001, 0.001, 0.001, ,)
mcmc = mcmc_conf(1000, 100,1)

function Gibbs(rng, conf, linear_model, model_parameters)
    epsilon = copy(linear_model.y)
    beta = zeros(model_parameters.M)
    sigmaE = rand(rng)
    sigmaG = 1 - sigmaE
    pi = 1
    samples = zeros(2, conf.iter / conf.thin)
    for i = 1:conf.iter
        bayesR!(rng, model_parameters, linear_model, epsilon, beta, sigmaE, sigmaG, pi)
        samples[i,1] = sigmaE
        samples[i,2] = sigmaG
    end
end
