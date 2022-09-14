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
using Statistics

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


struct matrix_linear_model
    Y::Matrix
    X::Matrix
	p::Int
end


function Gibbs_columns(rng, conf, matrix_linear_model, model_parameters)
    epsilon = zeros(model_parameters.N)
    beta = (sqrt(0.5)/model_parameters.M) * ones(model_parameters.M)
    sigmaE = Ref{Float64}(0.5)
    sigmaG = Ref{Float64}(0.5)
	K = size(model_parameters.mixture,1)
    pi_vec = rand(rng,Dirichlet(ones(K + 1)))
    total_samples = conf.iter - conf.burnin
    samples = zeros(3, convert(Int64,ceil(total_samples / conf.thin)))
	samples[1,1]=sigmaE[]
	samples[2,1]=sigmaG[]
	samples[3,1] = 0
	markerI = Vector(1:model_parameters.M)
	m0 = 0
	pip = zeros(model_parameters.M)
    beta_samples = zeros(model_parameters.M, total_samples)
    k = 0
    for i = 2: conf.iter
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
        if (i > conf.burnin)
            k += 1
            samples[3,k] = m0
            samples[1,k] = sigmaE[]
            samples[2,k] = sigmaG[]
            beta_samples[:,k] = beta
            pip = pip + (beta .> 0)
        end
    end
	samples, (pip / k), beta_samples
end

input_file = "/Users/Daniel/git/aurora/data/X_bamf_simulation.feather"
output_path = "/Users/Daniel/git/aurora/data"



@info "Starting Gibbs"
@info "Reading data from: $input_file"
@info "output_path: $output_path"
table = Arrow.Table(input_file)

@info "table loaded"
table=DataFrame(table)
table=table[:,2:end]

matrix_data = Matrix(table)

imputed = Impute.impute(matrix_data, Impute.Substitute(; statistic=mean); dims=:cols)
dt = fit(ZScoreTransform, identity.(imputed), dims=1)
dt = StatsBase.transform(dt, identity.(imputed))
@info "Calculating the correlation matrix"
correlation_matrix = cor(dt)
mean_cor = mean(correlation_matrix, dims=2)

M = size(dt)[2]
@info "M: $M"
N =size(dt)[1]
@info "N: $N"
lin = matrix_linear_model(dt,dt,M)
@info "linear model created"
iters=5000
@info "iters: $iters"
burnin=1000
@info "burnin: $burnin"
thin=1
@info "thin: $thin"
this_conf = mcmc_conf(iters,thin,burnin)
@info "conf created"
parameters = spike_slab(0.001, 0.001, N, 1.0/M, 0.001, 0.001,M ,N, [1])
@info "parameters created"
seed = 439813
@info "seed: $seed"
rng = MersenneTwister(seed)
@info "rng created"
@info "Starting Gibbs chain 1"
@time samples_1, pip_1, beta_1 = Gibbs_columns(rng, this_conf, lin, parameters)

@info "Starting Gibbs chain 2"
@time samples_2, pip_2, beta_2 = Gibbs_columns(rng, this_conf, lin, parameters)

@info "Starting Gibbs chain 3"
@time samples_3, pip_3, beta_3 = Gibbs_columns(rng, this_conf, lin, parameters)

@info "Starting Gibbs chain 4"
@time samples_4, pip_4, beta_4 = Gibbs_columns(rng, this_conf, lin, parameters)

@info "Starting Gibbs chain 5"
@time samples_5, pip_5, beta_5 = Gibbs_columns(rng, this_conf, lin, parameters)

total_samples = this_conf.iter - this_conf.burnin

samples_t = (total_samples /this_conf.thin )* hcat(pip_1, pip_2, pip_3, pip_4, pip_5)
samples_prob_t = sum(samples_t,dims=2) / (5*total_samples)
names_t = names(DataFrame(table))
samples_prob_df = DataFrame(amu= names_t,  probability=samples_prob_t[:,1], mean_cor=mean_cor[:,1])


output_prob = output_path * "/" * "samples_prob_df.feather"
@info "Writing total probabilities to: $output_prob"
Arrow.write(output_prob, samples_prob_df)

pip_t = [pip_1, pip_2, pip_3, pip_4, pip_5]
samples_t = [samples_1, samples_2, samples_3, samples_4, samples_5]
beta_t = [beta_1, beta_2, beta_3, beta_4, beta_5]
for i=1:5
    local output_samples = output_path * "/" * "samples_$(i).feather"
    @info "Writing data to: $output_samples"
    local samples_df = DataFrame(samples_t[i]', :auto)
    rename!(samples_df, Symbol.(["sigmaE", "sigmaG", "m0"]))
    Arrow.write(output_samples, samples_df)

    local output_pip = output_path * "/" * "pip_$(i).feather"
    @info "Writing data to: $output_pip"
    local pip_df = DataFrame(pip_t[i]', :auto)
    rename!(pip_df, names_t)
    Arrow.write(output_pip, pip_df)

    local output_beta = output_path * "/" * "beta_$(i).feather"
    @info "Writing data to: $output_beta"
    local beta_df = DataFrame(beta_t[i]', :auto)
    rename!(beta_df, names_t)
    Arrow.write(output_beta, beta_df)
end

@info "calculating pareto front"
df = DataFrame(amu=names_t, x=samples_prob_t[:,1], y= 1 .- mean_cor[:,1])
sort!(df,[:x, :y] ,rev=true);

pareto = df[1:1, :];

foreach(row -> row.y > pareto.y[end] && push!(pareto, row), eachrow(df));

@info "writing pareto front"
output_pareto = output_path * "/" * "pareto.feather"
@info "Writing data to: $output_pareto"
Arrow.write(output_pareto, pareto)
