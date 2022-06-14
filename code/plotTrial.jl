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

function getCoord(Shat,states,samples,Si_hat,Sf_hat)
    d = size(states)[2]
    l = length(Shat)    
    states_hat = zeros(l,d)
    for i=1:l
        states_hat[i,:] = states[Shat[i],:]
    end

    X=[];Y=[]
    for Trj in samples
        xx=[];yy=[]
        for trj in Trj
            x, y = states_hat[trj,:]
            append!(xx,x)
            append!(yy,y)
        end
        push!(X,xx)
        push!(Y,yy)
    end
    
    x_Si,y_Si=states_hat[Si_hat,:]
    x_Sf,y_Sf=states_hat[Sf_hat,:]

    return X,Y,x_Si,y_Si,x_Sf,y_Sf
end

samples, t_per, t_total = gillespie(Si_hat, Khat, StoppingCondition(pred=AbsorbingStates([Sf_hat])), N);
plot(X,Y)
plotlyjs()

plot()
plot(X,Y,t_per)
(x_Si,y_Si)=states_hat[Si_hat,:]
(x_Sf,y_Sf)=states_hat[Sf_hat,:]

plot!([x_Si],[y_Si], [0],seriestype = :scatter, markersize=5, color=cgrad(:reds)[.25],)
        #series_annotations = [("I",:left,20)])
plot!([x_Sf],[y_Sf], [0], seriestype = :scatter, markersize=5, color=cgrad(:greens)[.25],)
        #series_annotations = [("F",:bottom,20)], right_margin = 4Plots.mm, left_margin = 4Plots.mm)


plot([x_Si],[y_Si], [0],seriestype = :scatter, markersize=12, color=cgrad(:reds)[.25],
series_annotations = [("I",:bottom,20)])


scatter(traj[1],t_per[1])
scatter(X,Y,t_per)


plot()
@gif for i in 1:length(X[1])
    plot!([x_Si],[y_Si], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
        series_annotations = [("I",:center,8)])
    plot!([x_Sf],[y_Sf], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
        series_annotations = [("F",:center,8)], right_margin = 4Plots.mm, left_margin = 4Plots.mm)
    xlims!(-.9,50.1); ylims!(-.9,50.1);



    if i < length(X[1])
        if t_per[1][i] > 5e-7
            plot!(X[1][i:i+1],Y[1][i:i+1],color=:red,linewidth=5,legend=false)
        else
            plot!(X[1][i:i+1],Y[1][i:i+1],color=:grey,legend=false)
        end
    end
end


plot()
@gif for i in 1:length(X[1])
    plot!([x_Si],[y_Si], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
        series_annotations = [("I",:center,8)])
    plot!([x_Sf],[y_Sf], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
        series_annotations = [("F",:center,8)], right_margin = 4Plots.mm, left_margin = 4Plots.mm)
    xlims!(-.9,50.1); ylims!(-.9,50.1);
    if i < length(X[1])
        if t_per[1][i] > 5e-7
            scatter!(X[1][i:i+1],Y[1][i:i+1],t_per[1][i:i+1],color=cgrad(:cool)[0.1],markersize=8,legend=false)
        else
            scatter!(X[1][i:i+1],Y[1][i:i+1],t_per[1][i:i+1],color=cgrad(:Blues)[0.1],legend=false)
        end    
    end
end