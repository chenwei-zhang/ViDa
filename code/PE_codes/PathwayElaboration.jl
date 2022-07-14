using JLD, Printf, Statistics, Plots, Distributions, HDF5, LinearAlgebra

include("gillespie.jl")

function distance(K::RateMatrix, Sf::S) where {S<:Integer}
    "computes distance from Sf to all other states, assuming
    that the graph implied by K is fully connected and undirected"

    n = size(K)[1]
    dist = zeros(S, n)
    seen = zeros(Bool, n)
    seen[Sf] = 1

    queue = [(Sf,0)]
    while queue != []
        s,d = popfirst!(queue)
        dist[s] = d
        for i=1:n
            if K[s,i] > 0 && seen[i] == 0
                push!(queue, (i,d+1))
                seen[i] = 1
            end
        end
    end

    # indicator matrix
    idx = zeros(S, n, n)
    for i=1:n
        idx[i,findall(dist.==dist[i]-1)].=1
    end

    return dist, idx

end

# Define bias probability matrix
function generate_Pbias(K::RateMatrix, P::TransitionProbs, Sf::S, beta::Float64) where {S<:Integer}

    n = size(K)[1]
    Phat = zeros(n,n)

    _, idx = distance(K,Sf)
    Kbias = K .* idx
    for i in 1:n
        if i != Sf
            Phat[i,:] = Kbias[i,:]/sum(Kbias[i,:])
        else
            Phat[Sf,Sf] = 1
        end
    end

    Pbias = beta*P + (1-beta)*Phat
    @assert isapprox(sum(Pbias, dims=2), ones(n,1))

    return Pbias
end

function constructPathway(
    Si::S,Sf::S, K::RateMatrix, P::TransitionProbs, E::ExitRates,
    N::Int, beta::Float64) where {S<:Integer}

    Pbias = generate_Pbias(K,P,Sf,beta)

    trajectories, _ = gillespie(Si, Pbias, E, StoppingCondition(pred=AbsorbingStates([Sf])), N)

    return vcat(trajectories...)
end

function elaborateState(
    s::S,P::RateMatrix, E::ExitRates,
    N_ssa::Int,kappa::Float64) where {S<:Integer}

    trajectories, _ = gillespie(s, P, E, StoppingCondition(pred=StoppingTime(kappa)), N_ssa)

    return vcat(trajectories...)
end

# Probability matrix not defined yet
function pathwayElaboration(
    Si::S,Sf::S,K::RateMatrix,
    N_bias::Int,beta::Float64,N_ssa::Int,kappa::Float64) where {S<:Integer}

    P, E = generate_Pmatrix(K)
    pathwayElaboration(Si,Sf,K,P,E,N_bias,beta,N_ssa,kappa)
end

# Define Pathway Elaboration Algorithm
function pathwayElaboration(
    Si::S,Sf::S,K::RateMatrix,P::TransitionProbs, E::ExitRates,
    N_bias::Int,beta::Float64,N_ssa::Int,kappa::Float64) where {S<:Integer}

    Shat = S[]

    trajs = constructPathway(Si,Sf,K,P,E,N_bias,beta)
    for traj in trajs
        for s in traj
            Sprime = elaborateState(s,P,E,N_ssa,kappa)
            append!(Shat,Sprime)
        end
    end
    Shat = unique(Shat) ##TODO: check if correct?
    Si_hat = findall(Shat .== Si)[1]
    Sf_hat = findall(Shat .== Sf)[1]

    # Generate Khat rate matrix from Shat
    n = size(Shat)[1]
    Khat = zeros(n,n)
    for i=1:n
        for j=1:n
            Khat[i,j] = K[Shat[i],Shat[j]]
        end
    end

    return Shat,Khat,Si_hat,Sf_hat
end

function exact_fptd(t::Time,Si::S,K::RateMatrix) where {S<:Integer}
    E = generate_Ematrix(K)
    Q = K - diagm(E)
    exp(t*Q)[Si,:]
end

function exact_fptd(t::Time,Si::S,K::RateMatrix,E::ExitRates) where {S<:Integer}
    Q = K - diagm(E)
    exp(t*Q)[Si,:]
end

function exact_mfpt(Sf::S,K::RateMatrix) where {S<:Integer}
    E = generate_Ematrix(K)
    exact_mfpt(Sf,K,E)
end

function exact_mfpt(Sf::S,K::RateMatrix,E::ExitRates) where {S<:Integer}
    n = size(K)[1]

    Q = K - diagm(E)
    Q_tilde = Q[1:end .!= Sf, 1:end .!= Sf]

    t = - Q_tilde \ ones(n-1)
    t = vcat(t[1:Sf-1], [0], t[Sf:end])
end

function deltaPruning(Si::S, Sf::S,K::RateMatrix,delta::Float64) where {S<:Integer}

    Sdp = [Sf]
    S_full = collect(1:size(K)[1])

    mfpt = exact_mfpt(Sf,K)
    tau_i = mfpt[Si]

    for s in setdiff(S_full,Si,Sf)
        if mfpt[s] < tau_i*delta
            append!(Sdp,s)
        end
    end

    Sdelta1 = setdiff(S_full,Sdp) # Set Sdelta without Sd
    n = size(Sdelta1)[1]

    Sd = n+1  # Create Sd as the last state with index n+1

    Kdelta = zeros(n+1,n+1)
    for i=1:n
        Kdelta[i,1:n] = K[Sdelta1[i],Sdelta1]
        Kdelta[i,n+1] = sum(K[Sdelta1[i],Sdp])
    end

    return Sdp, Kdelta, Sd
end