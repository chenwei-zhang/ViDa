using HDF5, Printf, NNlib, Statistics, Plots, JLD, LinearAlgebra, Distributions

pwd()
cd("/Users/chenwei/Desktop/Github/RPE/code")
# run this file from /code

include("gillespie.jl")
include("PathwayElaboration.jl")
include("occupancyPlot.jl")

dataname = "synthetic50"

## Original data
data = load("data/original/$dataname.h5")
sep = hcat(split.(setdiff(keys(data),["states"]), "/")...)
# get dataset name
data_keys = Set(String.(sep[1, :]))
# get Si, Sf, energies, K
data_values = Set(String.(sep[2, :]))
states = data["states"]
states[45:60,:]

## Result CTMC
data_hat = load("data/results/$dataname.h5")
sep_hat = hcat(split.(setdiff(keys(data_hat)), "/")...)
# get dataset name (each dataset has different configuration, i.e. diff Si and Sf)
data_keys_hat = Set(String.(sep_hat[1, :]))
# get Khat, Shat, Si_hat, est_mfpt, obs_mfpt, Sf_hat
data_values_hat = Set(String.(sep_hat[2, :]))


## Test 
Si = (data["side_basin2/Si"])
Si = Int(data["side_basin2/Si"][10])
Sf = data["side_basin2/Sf"]
Sf = Int(data["side_basin2/Sf"][10])
K = data["side_basin2/K"]
energy = data["side_basin2/energies"]

Si_hat = data_hat["side_basin2_10/Si_hat"]
Sf_hat = data_hat["side_basin2_10/Sf_hat"]
Shat = data_hat["side_basin2_10/Shat"]
Khat = data_hat["side_basin2_10/Khat"]
obs_mfpt = data_hat["side_basin2_10/obs_mfpt"]
est_mfpt = data_hat["side_basin2_10/est_mfpt"]

### gillespie samples on full matrix
N = 1 # number of samples
traj, t_per, t_total = gillespie(Si_hat, Khat, StoppingCondition(pred=AbsorbingStates([Sf_hat])), N)
traj[1]
t_per[1]

maximum(t_per[1])
argmax(t_per[1])
length(t_per[1])
traj[1][argmax(t_per[1])]

t_per[1][argmax(t_per[1])]
t_per[1][Sf_hat]

scatter(traj[1],t_per[1])

d = size(states)[2]
l = length(Shat)
states_hat = zeros(l,d)
for i=1:l
    states_hat[i,:] = states[Shat[i],:]
end

X=[]; Y=[]
for trj in traj[1]
    x, y = states_hat[trj,:]
    append!(X,x)
    append!(Y,y)
end
X
Y

plot(X,Y)

plot(X,Y,t_per)
scatter(X,Y,t_per)



# State plot with General scatter plot function
function statePlot(SAMPLES::Vector,Si::S,Sf::S,K::RateMatrix,states::Matrix,logscale::String) where {S<:Integer}
    # Calculate state frequency
    n = size(K)[1]
    state_freq = zeros(n)
    visited = vcat(SAMPLES...)
    for i=1:n
        state_freq[i] = sum(visited.==i)
    end
    state_freq = state_freq./sum(state_freq) # normalization

    X=[];Y=[];Z=[]  # Get position and corresponding frequency of each state
    for s in visited
        x, y = states[s,:]
        z = state_freq[s]
        append!(X,x);append!(Y,y);append!(Z,z)
    end

    if logscale == "ON"
        Z = log.(Z)
        colorbar_title="\nLog probability of state"
    elseif logscale == "OFF"
        colorbar_title="\nProbability of state"
    end

    # Scatter plot
    plot()
    scatter!(X,Y,m=cgrad(rev=true),zcolor=Z,markersize=5,legend=false,
        colorbar=true,colorbar_title=colorbar_title,xlabel="X",ylabel="Y",grid=false)
    xlims!(-0.4,sqrt(n)+1.4); ylims!(-0.4,sqrt(n)+1.4)
    scatter!((states[Si,:]),(states[Sf,:]),markersize=0,
    series_annotations = [("Si",:center,:green,8),("Sf",:center,:red,8)], right_margin = 4Plots.mm, left_margin = 4Plots.mm)
end


