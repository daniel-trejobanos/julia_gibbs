using Distributions
using Random
using StatsBase
using Infiltrator
using BenchmarkTools
using Plots
using Logging
using DataFrames
using Arrow
using Impute

debuglogger = ConsoleLogger(stderr, Logging.Info)
global_logger(debuglogger)

function InvScChisq(rng, dof, scale)
    dof*scale/rand(rng, Chisq(dof))
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
	mixture::Vector
end

struct mcmc_conf
    iter::Int
    thin::Int
    burnin::Int
end

struct linear_model
    y::Vector
    X::Matrix
end

mutable struct linear_model_stats
    means::Vector
    sds::Vector
end


_exp(x::AbstractVecOrMat) = exp.(x .- maximum(x))
_exp(x::AbstractVecOrMat, theta::AbstractFloat) = exp.((x .- maximum(x) )*theta)
_sftmax(e::AbstractVecOrMat, d::Integer) = (e ./ sum(e,dims=d))

function softmax(X::AbstractVecOrMat{T}, dim::Integer)::AbstractVecOrMat where T <: AbstractFloat
    _sftmax(_exp(X), dim)
end

function softmax(X::AbstractVecOrMat{T}, dim::Integer, theta::Float64 )::AbstractVecOrMat where T <: AbstractFloat
    _sftmax(_exp(X, theta),dim)
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


function bayesR!(rng, parameters::spike_slab, X, epsilon, beta, sigmaE, sigmaG, pi_vec, markerI)

    components = parameters.mixture
	@debug "components: $components"
    Nm1 = N - 1.0
	@debug "N: $N"
    z = rand(rng,Normal(), size(X)[2])
    log_pi = log.(pi_vec)

	@debug "log_pi $log_pi"
    prior_term = 0.5 * log.(sigmaG[] * components * Nm1 / sigmaE[] .+ 1.0)
    sE_sG = sigmaE[] / sigmaG[]
	m0 = 0
	mixture = zeros(length(log_pi))
    for i = 1:parameters.M
		marker = markerI[i]
        beta_old = beta[marker]

        num = sum(X[:,marker] .* (epsilon + beta_old*X[:,marker]))
        denom = Nm1 .+  sE_sG ./ components
        lik_term =  0.5 * (num^2)./(sigmaE[]*denom)

        log_lik = log_pi + vcat([0.0], - prior_term  + lik_term )
        beta_sample = rand(rng,Categorical(softmax(log_lik,1)))
        beta_nzero = num ./ denom + sqrt.(sigmaE[]./ denom ).*z[marker]

        beta_new = if (beta_sample==1) 0.0 else beta_nzero[beta_sample-1] end
		mixture[beta_sample]+=1
		if(beta_sample !=1)
			m0+=1
		end
        epsilon .+= (beta_old - beta_new ) * X[:,marker]
		beta[marker] = beta_new
    end
	norm = jsumnorm(beta)
	@debug "m0: $m0"
	@debug "norm: $norm"
    sigmaG[] = InvScChisq(rng, parameters.v0G + m0,
                        (norm * m0 + parameters.v0G * parameters.s02G) /
                        (parameters.v0G + m0))
	@debug "mixture $mixture"
    pi_vec .= rand(rng,Dirichlet(1 .+ mixture))
	@debug "new pi: $pi_vec"
    sigmaE[] = InvScChisq(rng,parameters.v0E + parameters.N,
                        (jsumnorm(epsilon) + parameters.v0E * parameters.s02E)/
                        (parameters.v0E + parameters.N))
	@debug "new sigmaE: $(sigmaE[])"
	@debug "new sigmaG: $(sigmaG[])"
	mixture[1]
end


rng = MersenneTwister(1234)
M_0 = 100
N = 10000 # number of individuals
M = 200 # total number of SNP

G = rand(rng, Binomial(2, 0.2), N,M)
beta = zeros(M)
beta[1:M_0] = rand(rng, Normal(0,sqrt(0.6/M_0)), M_0)

dt = fit(ZScoreTransform, convert( Matrix{Float64},G), dims=2)
X = StatsBase.transform(dt,convert( Matrix{Float64},G))

g = X *  beta # same as scale(G) %*% beta
e = rand(rng, Normal(0, sqrt(0.4) ), N)
y = g + e

parameters = spike_slab(0.001, 0.001, 0.001, 0.001, 0.001, 0.001,M ,N, [0.001,0.1,1])
mcmc = mcmc_conf(10000, 100,1)
lin = linear_model(y,X)

function Gibbs(rng, conf, linear_model, model_parameters)
    epsilon = deepcopy(linear_model.y)
    beta = (sqrt(0.5)/model_parameters.M) * ones(model_parameters.M)
    sigmaE = Ref{Float64}(0.5)
    sigmaG = Ref{Float64}(0.5)
	K = size(model_parameters.mixture,1)
    pi_vec = rand(rng,Dirichlet(ones(K + 1)))
    samples = zeros(3, convert(Int64,ceil(conf.iter / conf.thin)))
	samples[1,1]=sigmaE[]
	samples[2,1]=sigmaG[]
	samples[3,1] = 0
	markerI = Vector(1:parameters.M)

    for i = 2:conf.iter
		Random.shuffle!(rng, markerI)
		@debug "pi: $pi_vec"
		@debug "sigmaE: $(sigmaE[])"
		@debug "sigmaG: $(sigmaG[])"
		@debug "VE: $(sigmaG[]/(sigmaG[]+sigmaE[]))"
        samples[3,i] = bayesR!(rng, model_parameters, linear_model.X, epsilon, beta, sigmaE, sigmaG, pi_vec, markerI)
        samples[1,i] = sigmaE[]
        samples[2,i] = sigmaG[]
    end
	samples
end

this_conf = mcmc_conf(1000,1,1)

struct matrix_linear_model
    Y::Matrix
    X::Matrix
	p::Int
end

this_conf = mcmc_conf(100,1,1)

lin = matrix_linear_model(repeat(y,1,3),X,3)
function Gibbs_columns(rng, conf, matrix_linear_model, model_parameters)
    epsilon = zeros(model_parameters.N)
    beta = (sqrt(0.5)/model_parameters.M) * ones(model_parameters.M)
    sigmaE = Ref{Float64}(0.5)
    sigmaG = Ref{Float64}(0.5)
	K = size(model_parameters.mixture,1)
    pi_vec = rand(rng,Dirichlet(ones(K + 1)))
    samples = zeros(3, convert(Int64,ceil(conf.iter / conf.thin)))
	samples[1,1]=sigmaE[]
	samples[2,1]=sigmaG[]
	samples[3,1] = 0
	markerI = Vector(1:model_parameters.M)
	m0 = 0
	pip = zeros(model_parameters.M)
    for i = 2:conf.iter
		for j = 1:1
			epsilon = matrix_linear_model.Y[:,j] + epsilon
			current_shuffle = shuffle(rng,markerI[1:end .!= j])

			# Random.shuffle!(rng, markerI)
			@debug "pi: $pi_vec"
			@debug "sigmaE: $(sigmaE[])"
			@debug "sigmaG: $(sigmaG[])"
			@debug "VE: $(sigmaG[]/(sigmaG[]+sigmaE[]))"
			parameters_iter = spike_slab(0.001, 0.001, 458, 1.0/M, 0.001, 0.001,M-1 ,N, [1])
	        m0=bayesR!(rng, parameters_iter, matrix_linear_model.X, epsilon, beta, sigmaE, sigmaG, pi_vec, current_shuffle)

			epsilon= epsilon - matrix_linear_model.Y[:,j]

		end
		Random.shuffle!(rng, markerI)
		@debug "pi: $pi_vec"
		@debug "sigmaE: $(sigmaE[])"
		@debug "sigmaG: $(sigmaG[])"
		@debug "VE: $(sigmaG[]/(sigmaG[]+sigmaE[]))"
        samples[3,i] = m0
        samples[1,i] = sigmaE[]
        samples[2,i] = sigmaG[]
		pip = pip + (beta .> 0)
    end
	samples, (pip / conf.iter), beta
end


table = Arrow.Table("/Users/Daniel/git/aurora/data/X_bamf_simulation.feather")
table=DataFrame(table)
table=table[:,2:end]
krakow_1d = Matrix(table)

imputed = Impute.impute(krakow_1d, Impute.Substitute(; statistic=mean); dims=:cols)
dt = fit(ZScoreTransform, identity.(imputed), dims=1)
dt = StatsBase.transform(dt, identity.(imputed))
M = size(dt)[2]
N =size(dt)[1]
lin = matrix_linear_model(dt,dt,91)
this_conf = mcmc_conf(5000,1,1)
parameters = spike_slab(0.001, 0.001, 458, 1.0/M, 0.001, 0.001,M ,N, [1])
result= Gibbs_columns(rng, this_conf, lin, parameters)
samples = result[1]
samples_1 = result[2]
plot(1:this_conf.iter, samples[2,:])
bar(result[2])
plot(1:this_conf.iter, samples[3,:])

names_1 = names(DataFrame(table))[findall(result[2] .> 0.5)]

result= Gibbs_columns(rng, this_conf, lin, parameters)
samples = result[1]
samples_2 = result[2]
plot(1:this_conf.iter, samples[2,:])
bar(result[2])
plot(1:this_conf.iter, samples[3,:])
names(DataFrame(table))[findall(result[2] .> 0.5)]
names_2=names(DataFrame(table))[findall(result[2] .> 0.5)]

result= Gibbs_columns(rng, this_conf, lin, parameters)
samples = result[1]
samples_3 = result[2]
plot(1:this_conf.iter, samples[2,:])
bar(result[2])
plot(1:this_conf.iter, samples[3,:])
names(DataFrame(table))[findall(result[2] .> 0.5)]
names_3=names(DataFrame(table))[findall(result[2] .> 0.5)]

result= Gibbs_columns(rng, this_conf, lin, parameters)
samples = result[1]
samples_4 = result[2]
plot(1:this_conf.iter, samples[2,:])
bar(result[2])
plot(1:this_conf.iter, samples[3,:])
names(DataFrame(table))[findall(result[2] .> 0.5)]
names_4=names(DataFrame(table))[findall(result[2] .> 0.5)]


result= Gibbs_columns(rng, this_conf, lin, parameters)
samples = result[1]
samples_5 = result[2]
plot(1:this_conf.iter, samples[2,:])
bar(result[2])
plot(1:this_conf.iter, samples[3,:])
names(DataFrame(table))[findall(result[2] .> 0.5)]
names_5=names(DataFrame(table))[findall(result[2] .> 0.5)]
#println(intersect(names_1, names_2, names_3, names_4, names_5))

samples_t = this_conf.iter * hcat(samples_1, samples_2, samples_3, samples_4, samples_5)
samples_prob_t = sum(samples_t,dims=2) / (5*this_conf.iter)
names_t = names(DataFrame(table))[findall(samples_prob_t .> 0.5)]
