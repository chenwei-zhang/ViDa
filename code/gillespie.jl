using JLD, Printf, Statistics, Plots, Distributions, HDF5
import Base.@kwdef

const States{S<:Integer} = Vector{S}
const Time = Float64
const StateProbs = Vector{Float64}
const RateMatrix = Matrix{Float64}
const ExitRates = Vector{Float64}
const TransitionProbs = Matrix{Float64}
struct AbsorbingStates{S<:Integer}
    ss::States{S}
end
struct StoppingLength
    l::Int
end
struct StoppingTime
    t::Time
end

const StoppingPredicate{S<:Integer} =
    Union{AbsorbingStates{S},StoppingLength,StoppingTime}
@kwdef struct StoppingCondition
    pred::StoppingPredicate
end
@kwdef mutable struct SimTrajectory{S<:Integer}
    times::Vector{Time}
    states::States{S}
    total_time::Time
end


# Define Exit rate matrix
function generate_Ematrix(K::RateMatrix)
    """
    Returns the diagonal exit rate matrix, given the rate matrix K
    """
    vec(sum(K, dims=2))
end


 # Define P matrix
 function generate_Pmatrix(K::RateMatrix)
    n = size(K)[1]

    E = generate_Ematrix(K)
    P = K ./ E

    # Confirm that transition probabilities sum up to 1, starting from each state
    @assert isapprox(sum(P,dims=2), ones(n,1))

    return P, E
end


# For vector p giving discrete probabilities, generates a random sample
function sampleDiscrete(pt)
	findfirst(cumsum(pt[:]) .> rand())  #TODO: using library (Distributions)
end


# Check stopping condition
function check_stopping(sim::SimTrajectory{S}, pred::AbsorbingStates{S}) :: Bool where {S<:Integer}
    length(intersect(sim.states[end], pred.ss)) > 0
end
function check_stopping(sim::SimTrajectory{S}, pred::StoppingLength) :: Bool where {S<:Integer}
    length(sim.states) >= pred.l
end
function check_stopping(sim::SimTrajectory{S}, pred::StoppingTime) :: Bool where {S<:Integer}
    sim.total_time >= pred.t
end


function gillespie(Si::S, K::RateMatrix, stop::StoppingCondition, N::Int) where {S<:Integer}
    P, E = generate_Pmatrix(K)
    gillespie(Si, P, E, stop, N)
end

function gillespie(Si::S, P::TransitionProbs, E::ExitRates, stop::StoppingCondition) where {S<:Integer}
    p1 = zeros(length(E))
    p1[Si] = 1
    gillespie(p1, P, E, stop)
end

function gillespie(Si::S, P::TransitionProbs, E::ExitRates, stop::StoppingCondition, N::Int) where {S<:Integer}
    S_samples = Vector{S}[]
    t_samples = Time[]
    t_per = []

    for i in 1:N
        simGillespie = gillespie(Si,P,E,stop)
        push!(S_samples,simGillespie.states)
        push!(t_per,simGillespie.times)
        push!(t_samples,simGillespie.total_time)
    end

    return S_samples, t_per, t_samples
end

function gillespie(p1::StateProbs, P::TransitionProbs, E::ExitRates, stop::StoppingCondition) where {S<:Integer}
    sim = SimTrajectory(times=[0.0], states=[sampleDiscrete(p1)], total_time=0.0)
    while ! check_stopping(sim, stop.pred)
        gillespie(sim, P, E)
    end
    return sim
end

function gillespie(sim::SimTrajectory{S}, P::TransitionProbs, E::ExitRates) where {S<:Integer}
    s = sampleDiscrete(P[sim.states[end],:])
    lambda = E[s] # get lambda of current state for exponential distribution
    dt = rand(Exponential(1/lambda))

    append!(sim.states, s)
    append!(sim.times, dt)
    sim.total_time += dt
end
