using HDF5, Printf, NNlib, Statistics, Plots, JLD, LinearAlgebra, Distributions

# run this file from /code

include("gillespie.jl")
include("PathwayElaboration.jl")
include("occupancyPlot.jl")

# datanames = ["synthetic20", "synthetic_rev20"]
# data = load("Project/code/data/synthetic20.h5")
# data_hat = load("Project/code/results/synthetic20.h5")
# states = data["states"]
# sep = hcat(split.(setdiff(keys(data),["states"]), "/")...)
# data_keys = Set(String.(sep[1, :]))

datanames = ["synthetic20", "synthetic_rev20", "synthetic50", "synthetic_rev50","synthetic100", "synthetic_rev100"]

pwd()

S_HAT = []; MFPT_obs = []; MFPT_est=[]; AE = []; With3=[]; NOTWITH3=[];UNDER=[];YESWITH3=[]
for dataname in datanames
    data = load("data/$dataname.h5")
    data_hat = load("results/$dataname.h5")
    states = data["states"]

    sep = hcat(split.(setdiff(keys(data),["states"]), "/")...)
    data_keys = Set(String.(sep[1, :]))

    shat = []; mfpt_obs = []; mfpt_est=[]; s = []; ae = []; with3=0; notwith3=0; under=0
    for id in data_keys
        nconfig = length(data["$(id)/Si"])
        for config in 1:nconfig
            # Load truncated CTMC data
            id_hat = id*"_"*string(config)
            Shat = data_hat["$(id_hat)/Shat"]

            # Get t from results
            t_obs = data_hat["$(id_hat)/obs_mfpt"]
            t_est = data_hat["$(id_hat)/est_mfpt"]

            t_logdiff = abs(log10(t_obs)-log10(t_est))

            append!(ae,t_logdiff)
            append!(shat,length(Shat))
            append!(mfpt_obs,t_obs)
            append!(mfpt_est,t_est)

            if t_logdiff <= log10(3)
                with3+=1
            end 

            if t_est < t_obs
                under+=1
            end 
            
            if t_logdiff >= log10(3) &&  t_est < t_obs
                notwith3+=1
            end 
        end
    end
    push!(With3,with3)
    push!(NOTWITH3,notwith3)
    push!(UNDER,under)
    push!(AE,ae)
    push!(S_HAT,shat)
    push!(MFPT_obs,mfpt_obs)
    push!(MFPT_est,mfpt_est)

end

S_HAT
MFPT_obs
MFPT_est
AE
With3
@show(NOTWITH3)
UNDER
# YESWITH3


#
Strunc = mean(collect(Iterators.flatten(S_HAT[5:6]))./10000)
#
s_T_est = collect(Iterators.flatten(MFPT_est[5:6]))
s_T_obs = collect(Iterators.flatten(MFPT_obs[5:6]))
#
mae = mean(collect(Iterators.flatten(AE[3:4])))
maximum(collect(Iterators.flatten(AE[5:6])))
#
sum(s_T_est .< s_T_obs) / length(s_T_est)
#
W3 = sum(With3[5:6])/length(s_T_est)
#
NW3 =  sum(NOTWITH3[5:6])/length(s_T_est)

cd("/Users/chenwei/Desktop/Github/CPSC540_Assignments/Project/code")

# DNA dataset calculation
dataname = "dna23"
data = load("data/$dataname.h5")
data_hat = load("results/$dataname.h5")
sep = hcat(split.(setdiff(keys(data),["states"]), "/")...)
data_keys = Set(String.(sep[1, :]))


Si = data["Zhang4/Si"]
Sf = data["Zhang4/Sf"]
state = data["Zhang4/states"]
K = data["Zhang4/K"]
conc = data["Zhang4/conc"]
energy = data["Zhang4/energies"]


Si_hat = data_hat["Zhang4/Si_hat"]
Sf_hat = data_hat["Zhang4/Sf_hat"]
state_hat = data_hat["Zhang4/Shat"]
K_hat = data_hat["Zhang4/Khat"]
t_obs = data_hat["Zhang4/obs_mfpt"]


Si_4 = data["MorrisonTrue6/Si"]
Sf_4 = data["MorrisonTrue6/Sf"]
state_4 = data["MorrisonTrue6/states"]
state_4hat = data_hat["MorrisonTrue6/Shat"]


# Collect truncated dataset
SHAT = []; S=[]; MFPT_obs = []; MFPT_est=[]; AE = []; with3=0; notwith3=0
for id in data_keys
    # Load full and PE CTMC data
    states = data["$(id)/states"]
    Shat = data_hat["$(id)/Shat"]

    # Get t from results
    t_obs = data_hat["$(id)/obs_mfpt"]
    t_est = data_hat["$(id)/est_mfpt"]

    t_logdiff = abs(log10(t_obs)-log10(t_est))

    if t_logdiff <= log10(3)
        with3+=1
    end 

    if t_logdiff >= log10(3) &&  t_est < t_obs
        notwith3+=1
    end 

    append!(SHAT,length(Shat)); append!(MFPT_obs,t_obs); append!(MFPT_est,t_est);
    append!(S,length(states));append!(AE,t_logdiff)
end

S
SHAT
MFPT_obs
MFPT_est
AE
with3
notwith3
#
with3/length(SHAT)
notwith3/length(SHAT)
#
mean(AE)
maximum(AE)
#
sum(MFPT_est .< MFPT_obs) / length(MFPT_obs)
#
mean(SHAT ./ S)



######## PLOT ########

dataname = "synthetic50"

data = load("data/$dataname.h5")
data_hat = load("results/$dataname.h5")
states = data["states"]

id = "mountain_detour"
config = 1

Si = data["$(id)/Si"][config]
Sf = data["$(id)/Sf"][config]
K = data["$(id)/K"]

t_obs = data_hat[id*"_$config/obs_mfpt"]
t_estimate = data_hat[id*"_$config/est_mfpt"]

###
dataname = "dna23"
data_hat = load("results/$dataname.h5")
id = "MorrisonTrue6"
t_obs = data_hat["$(id)/obs_mfpt"]
t_estimate = data_hat["$(id)/est_mfpt"]



# Get t from results
t = data_hat[id*"_$config/obs_mfpt"]
t = 1   

P, E = generate_Pmatrix(K)
LOG="ON"
i = 1

path = "Project/code/syn50_plots/"

for beta in [0.2,0.4,0.6,0.8,1]

    Pbias = generate_Pbias(K,P,Sf,beta)
    figname = "biased_t_beta"*string(beta)*".png"

    println("----------- START $(i) -----------")

    ### 6. biased time stop samples (time at est-mfpt)
    SAMPLES, _ = gillespie(Si,Pbias,E,StoppingCondition(pred=StoppingTime(t)),5)
    println("----------- DONE Gillespie, $(i) -----------")
    exact_statePlot(t,Si,Sf,K,states, SAMPLES,logscale=LOG,
                    name="5 biased path samples, time limit at estimated MFPT",
                    path=path*figname)
    println("----------- DONE starePlot, $(i) -----------")
    println("∇")
    i+=1
end




