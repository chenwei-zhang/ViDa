using JLD, Printf, Statistics, Plots, Distributions, HDF5, LinearAlgebra
import Plots.pdf

include("gillespie.jl")
include("PathwayElaboration.jl")


## State density scatter plot function
# State plot with Gillespie Sf stop condition
function statePlot(Si::S,Sf::S,K::RateMatrix,states::Matrix,N::Int;logscale::String) where {S<:Integer}
    SAMPLES, _ = gillespie(Si,K,StoppingCondition(pred=AbsorbingStates([Sf])),N)
    statePlot(SAMPLES,Si,Sf,K,states,logscale)

    title!("Estimated State Occupancy by Gillespie with Final State Stop Condition after $(N) Samples")
    # savefig("Project/code/plots/state_freq_Sf_log$(logscale).pdf")
    savefig("plots/state_freq_Sf_log$(logscale).pdf")
end

# State plot with Gillespie Timelimit stop condition
function statePlot(Si::S,Sf::S,K::RateMatrix,states::Matrix,N::Int,timeLimit::Float64;logscale::String) where {S<:Integer}
    SAMPLES, _ = gillespie(Si,K,StoppingCondition(pred=StoppingTime(timeLimit)),N)
    statePlot(SAMPLES,Si,Sf,K,states,logscale)

    title!("Estimated State Occupancy by Gillespie within $(timeLimit) Seconds after $(N) Samples")
    # savefig("Project/code/plots/state_freq_timeLimit=$(timeLimit)_log$(logscale).pdf")
    savefig("plots/state_freq_timeLimit=$(timeLimit)_log$(logscale).pdf")
end

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

# ## Exact state occupancy plot
# # Plot exact state occupancy plot of full CTMC
# function exact_statePlot(t::Float64,Si::S,Sf::S,K::RateMatrix,states::Matrix;logscale::String) where {S<:Integer}
#     dim = sqrt(size(states)[1])
#     exact_statePlot(t,Si,Sf,K,states,logscale,dim)

#     title!("Exact State Occupancy of full CTMC with t=$(round(t)) Seconds",titlefontsize=9)
#     # savefig("Project/code/plots/Exact_occupancy_FULL_log$(logscale).pdf")
#     savefig("plots/Exact_occupancy_FULL_log$(logscale).pdf")
# end

## Exact state occupancy plot
# Plot exact state occupancy plot of full CTMC
function exact_statePlot(t::Float64,Si::S,Sf::S,K::RateMatrix,states::Matrix;logscale::String,name::String,path::String) where {S<:Integer}
    dim = sqrt(size(states)[1])
    exact_statePlot(t,Si,Sf,K,states,logscale,dim)

    title!(name)
    savefig(path)
end

## Exact state occupancy plot
# Plot exact state occupancy plot of full CTMC
# function exact_statePlot(t::Float64,Si::S,Sf::S,K::RateMatrix,states::Matrix,SAMPLES;logscale::String) where {S<:Integer}
#     dim = dim = sqrt(size(states)[1])
#     exact_statePlot(t,Si,Sf,K,states,logscale,dim,SAMPLES)

#     title!("Exact State Occupancy of full CTMC with t=$(round(t)) Seconds",titlefontsize=9)
#     # savefig("Project/code/plots/Exact_occupancy_FULL_log$(logscale).pdf")
#     savefig("plots/Exact_occupancy_FULL_log$(logscale).pdf")
# end

function exact_statePlot(t::Float64,Si::S,Sf::S,K::RateMatrix,states::Matrix,SAMPLES;logscale::String,name::String,path::String) where {S<:Integer}
    dim = dim = sqrt(size(states)[1])
    exact_statePlot(t,Si,Sf,K,states,logscale,dim,SAMPLES)

    title!(name)
    savefig(path)
end

# # Plot exact state occupancy plot of PE truncated CTMC
# function exact_statePlot(t::Float64,Si_hat::S,Sf_hat::S,Shat::Vector,Khat::RateMatrix,states::Matrix;logscale::String) where {S<:Integer}
#     # Generate states (coordinates)
#     d = size(states)[2]
#     l = length(Shat)
#     states_hat = zeros(l,d)
#     for i=1:l
#         states_hat[i,:] = states[Shat[i],:]
#     end

#     dim = sqrt(size(states)[1])

#     exact_statePlot(t,Si_hat,Sf_hat,Khat,states_hat,logscale,dim)

#     title!("Exact State Occupancy of PE truncated CTMC with t=$(round(t)) Seconds",titlefontsize=9)
#     # savefig("Project/code/plots/Exact_occupancy_PE_log$(logscale).pdf")
#     savefig("plots/Exact_occupancy_PE_log$(logscale).pdf")
# end

# Plot exact state occupancy plot of PE truncated CTMC
function exact_statePlot(t::Float64,Si_hat::S,Sf_hat::S,Shat::Vector,Khat::RateMatrix,states::Matrix;logscale::String,name::String,path::String) where {S<:Integer}
    # Generate states (coordinates)
    d = size(states)[2]
    l = length(Shat)
    states_hat = zeros(l,d)
    for i=1:l
        states_hat[i,:] = states[Shat[i],:]
    end

    dim = sqrt(size(states)[1])

    exact_statePlot(t,Si_hat,Sf_hat,Khat,states_hat,logscale,dim)

    title!(name)
    savefig(path)
end

# # Plot exact state occupancy plot of PE truncated CTMC
# function exact_statePlot(t::Float64,Si_hat::S,Sf_hat::S,Shat::Vector,Khat::RateMatrix,states::Matrix,SAMPLES;logscale::String) where {S<:Integer}
#     # Generate states (coordinates)
#     d = size(states)[2]
#     l = length(Shat)
#     states_hat = zeros(l,d)
#     for i=1:l
#         states_hat[i,:] = states[Shat[i],:]
#     end

#     dim = sqrt(size(states)[1])

#     exact_statePlot(t,Si_hat,Sf_hat,Khat,states_hat,logscale,dim,SAMPLES)

#     title!("Exact State Occupancy of PE truncated CTMC with t=$(round(t)) Seconds",titlefontsize=9)
#     # savefig("Project/code/plots/Exact_occupancy_PE_log$(logscale).pdf")
#     savefig("plots/Exact_occupancy_PE_log$(logscale).pdf")
# end

# Plot exact state occupancy plot of PE truncated CTMC
function exact_statePlot(t::Float64,Si_hat::S,Sf_hat::S,Shat::Vector,Khat::RateMatrix,states::Matrix,SAMPLES;logscale::String,name::String,path::String) where {S<:Integer}
    # Generate states (coordinates)
    d = size(states)[2]
    l = length(Shat)
    states_hat = zeros(l,d)
    for i=1:l
        states_hat[i,:] = states[Shat[i],:]
    end

    dim = sqrt(size(states)[1])

    exact_statePlot(t,Si_hat,Sf_hat,Khat,states_hat,logscale,dim,SAMPLES)

    title!(name)
    savefig(path)
end

# General exact_statePlot function with overlayed trajectories
function exact_statePlot(t::Float64,Si::S,Sf::S,K::RateMatrix,states::Matrix,logscale::String,dim,SAMPLES) where {S<:Integer}
    E = generate_Ematrix(K)
    p_t = exact_fptd(t,Si,K,E)

    if logscale == "ON"
        p_t = log.(p_t)
        colorbar_title="\nLog probability of state"
    elseif logscale == "OFF"
        colorbar_title="\nProbability of state"
    end

    X=[]; Y=[]
    n = size(K)[1]
    for i=1:n
        x, y = states[i,:]
        append!(X,x);append!(Y,y)
    end

    scatter(X,Y,m=cgrad(rev=true),zcolor=p_t,markersize=4,alpha=.5,markerstrokewidth=0,legend=false,colorbar=true,
    colorbar_title=colorbar_title,xlabel="X",ylabel="Y",grid=false)

    for traj in SAMPLES
        visited = vcat(traj...)
        x = []
        y = []
        for i in visited
            x1, y1 = states[i,:]
            append!(x,x1)
            append!(y,y1)
        end

        plot!(x,y,legend=false,c=cgrad()[0.01])
    end

    xlims!(-.9,dim); ylims!(-.9,dim)
    (x_Si,y_Si)=states[Si,:]; (x_Sf,y_Sf)=states[Sf,:]
    plot!([x_Si],[y_Si], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
    series_annotations = [("I",:center,8),("F",:center,8)])
    plot!([x_Sf],[y_Sf], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
    series_annotations = [("F",:center,8),("F",:center,8)], right_margin = 4Plots.mm, left_margin = 4Plots.mm)

end


# General exact_statePlot function
function exact_statePlot(t::Float64,Si::S,Sf::S,K::RateMatrix,states::Matrix,logscale::String,dim) where {S<:Integer}
    E = generate_Ematrix(K)
    p_t = exact_fptd(t,Si,K,E)

    if logscale == "ON"
        p_t = log.(p_t)
        colorbar_title="\nLog probability of state"
    elseif logscale == "OFF"
        colorbar_title="\nProbability of state"
    end

    X=[]; Y=[]
    n = size(K)[1]
    for i=1:n
        x, y = states[i,:]
        append!(X,x);append!(Y,y)
    end

    scatter(X,Y,m=cgrad(rev=true),zcolor=p_t,markersize=4,markerstrokewidth=0,legend=false,colorbar=true,
    colorbar_title=colorbar_title,xlabel="X",ylabel="Y",grid=false)

    xlims!(-.9,dim); ylims!(-.9,dim)
    (x_Si,y_Si)=states[Si,:]; (x_Sf,y_Sf)=states[Sf,:]
    plot!([x_Si],[y_Si], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
    series_annotations = [("I",:center,8),("F",:center,8)])
    plot!([x_Sf],[y_Sf], seriestype = :scatter, markersize=12, color=cgrad(:greens)[.25],
    series_annotations = [("F",:center,8),("F",:center,8)], right_margin = 4Plots.mm, left_margin = 4Plots.mm)

end


## Transition density plot function
# Transation plot with Gillespie Sf stop condition
function transationPlot(Si::S,Sf::S,K::RateMatrix,states::Matrix,N::Int;logscale::String) where {S<:Integer}
    SAMPLES, _ = gillespie(Si,K,StoppingCondition(pred=AbsorbingStates([Int(Sf)])),N)
    transationPlot(SAMPLES,Si,Sf,K,states,logscale)

    title!("Estimated Transition Occupancy by Gillespie with Final State Stop Condition after $(N) Samples")
    # savefig("Project/code/plots/tran_freq_Sf_log$(logscale).pdf")
    savefig("plots/tran_freq_Sf_log$(logscale).pdf")
end

# Transation plot with Gillespie Timelimit stop condition
function transationPlot(Si::S,Sf::S,K::RateMatrix,states::Matrix,N::Int,timeLimit::Float64;logscale::String) where {S<:Integer}
    SAMPLES, _ = gillespie(Si,K,StoppingCondition(pred=StoppingTime(timeLimit)),N)
    transationPlot(SAMPLES,Si,Sf,K,states,logscale)

    title!("Estimated Transition Occupancy by Gillespie within $(timeLimit) Seconds after $(N) Samples")
    # savefig("Project/code/plots/tran_freq_timeLimit=$(timeLimit)_log$(logscale).pdf")
    savefig("plots/tran_freq_timeLimit=$(timeLimit)_log$(logscale).pdf")
end

# Transation plot with general plot function
function transationPlot(SAMPLES::Vector,Si::S,Sf::S,K::RateMatrix,states::Matrix,logscale::String) where {S<:Integer}
    # Calculate transition frequency
    n = size(K)[1]
    tran_freq = zeros(Int,n,n)
    for traj in SAMPLES
        for i=1:(length(traj)-1)
            tran_freq[traj[i],traj[i+1]] += 1
        end
    end
    tran_freq_nrom = tran_freq./sum(tran_freq)

    # Transition plot
    plot()
    for i in 1:n
        for j in 1:n
            count = tran_freq_nrom[i,j]+tran_freq_nrom[j,i]
            if count > 0
                x1, y1 = states[i,:]
                x2, y2 = states[j,:]

                if logscale == "ON"
                    count = log(count)
                    colorbar_title="\nLog probability of transitions occured"
                elseif logscale == "OFF"
                    colorbar_title="\nProbability of transitions occured"
                end

                plot!([x1,x2],[y1,y2],line_z=count,legend=false,c=cgrad(rev=true),
                colorbar_title=colorbar_title,colorbar=true,xlabel="X",ylabel="Y",
                grid=false)
            end
        end
    end
    # xlims!(-0.4,sqrt(n)+1.4); ylims!(-0.4,sqrt(n)+1.4)
    (x_Si,y_Si)=states[Si,:]; (x_Sf,y_Sf)=states[Sf,:]
    scatter!([(x_Si,y_Si),(x_Sf,y_Sf)],markercolor=[:green,:red],
    series_annotations = [("Si",:top,:green,9),("Sf",:bottom,:red,9)], right_margin = 4Plots.mm, left_margin = 4Plots.mm)
end


