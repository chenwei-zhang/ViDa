using NPZ, Printf, NNlib, Statistics, Plots, JLD2, LinearAlgebra, Distributions

# batch save plots
function statePlot(fname;seq,plot_type,vis_method)
    # load data
    data_h5 =load("code/data/helix_assoc/$(fname).h5")

    # get coords from different vis methods and get each plot type data 
    all_coords, X, Y, x, y, trj_id, bg, color, linecolor, colorbar, colorbar_title = 
        statePlot(data_h5,vis_method,plot_type)

    # plot trajectory overlay background landscape
    for i=1:size(trj_id)[1]
        titlename = "$(vis_method): trajectory on $(plot_type) map of $(seq) in text file $(i-1) "
        filename = "$(seq)_text$(i-1)_$(vis_method)_$(plot_type)"

        # plot background landscape
        plot()
        statePlot(X,Y,x,y,bg,colorbar_title,color,colorbar)
        title!(titlename,titlefontsize=12)

        # plot trajectory
        statePlot(trj_id,i,all_coords,linecolor)
        
        # save plots
        path = "/Users/chenwei/Desktop/Github/RPE/plot/helix_assoc_$(seq)/$(plot_type)/$(vis_method)"
        mkpath(path)
        savefig("$(path)/$(filename)")
    end
end


# get coords from different vis methods and get each plot type data 
function statePlot(data_h5,vis_method,plot_type)
    # vis methods
    if vis_method == "PCA"
        all_coords = data_h5["pca_all_coords"]
        uniq_coords = data_h5["pca_coords"]
        # Z = uniq_coords[3,:]; z = all_coords[3,:]
    elseif vis_method == "PHATE"
        all_coords = data_h5["phate_all_coords"]
        uniq_coords = data_h5["phate_coords"]
    
    elseif vis_method == "UMAP"
        all_coords = data_h5["umap_all_coord_2d"]
        uniq_coords = data_h5["umap_coord_2d"]
    
    elseif vis_method == "tSNE"
        all_coords = data_h5["tsne_all_coord_2d"]
        uniq_coords = data_h5["tsne_coord_2d"]
    end

    # assign data to variables
    X = uniq_coords[1,:]; Y = uniq_coords[2,:]
    x = all_coords[1,:]; y = all_coords[2,:]
    trj_id = get_per_trj(data_h5)

    # plot types
    if plot_type == "energy"
        bg = data_h5["SIMS_G_uniq"]
        color = cgrad(:plasma)
        linecolor = palette(:grays)[1]
        colorbar = true
        colorbar_title = "\n Free energy"
    elseif plot_type == "pair"
        bg = data_h5["SIMS_pair_uniq"]
        color = cgrad(:grays)
        linecolor = palette(:default)[1]
        colorbar = false
        colorbar_title = false
    elseif plot_type == "occp"
        bg = log.(data_h5["occ_density_S"])
        color = cgrad(:grays,rev=true)
        linecolor = palette(:default)[1]
        colorbar = true
        colorbar_title="\n Log occupancy density of state"
    end

    return all_coords, X, Y, x, y, trj_id, bg, color, linecolor, colorbar, colorbar_title
end


# plot background: energy landscape, occ, pair maps
function statePlot(X,Y,x,y,bg,colorbar_title,color,colorbar=true)
    
    scatter!(X,Y,zcolor=bg,m=color,markerstrokewidth=0,markersize=4,colorbar=colorbar,
        colorbar_title=colorbar_title,legend=false,right_margin=5Plots.mm)
    
    plot!([x[1]],[y[1]], seriestype = :scatter, markersize=8, color=cgrad(:greens)[.25],
            series_annotations = [("I",:center,10)])
    plot!([x[end]],[y[end]], seriestype = :scatter, markersize=8, color=cgrad(:greens)[.25],
            series_annotations = [("F",:center,10)])
    plot!(xlabel="X", ylabel="Y")
    xlims!(minimum(X)*1.05,maximum(X)*1.05) 
    ylims!(minimum(Y)*1.05,maximum(Y)*1.05)
end


# plot trajectory
function statePlot(trj_id,i,all_coords,linecolor)
    if i == 1
        XX = all_coords[1,1:trj_id[i]]
        YY = all_coords[2,1:trj_id[i]]
    else
        XX = all_coords[1,trj_id[i-1]+1:trj_id[i]]
        YY = all_coords[2,trj_id[i-1]+1:trj_id[i]]
    end

    plot!(XX,YY,color=linecolor,linewidth=0.6,alpha=0.95)
end


# get each individual trajectory
function get_per_trj(data_h5,file_num=100)
    A = data_h5["pca_all_coords"]
    a = data_h5["pca_all_coords"][:,end]

    trj_id = Int[]; count=0
    for i in 1:size(A)[2]
        if A[:,i] == a
            count = count + 1
            append!(trj_id,i)
        end
    end

    @assert file_num == count
    return trj_id
end


# # parameters
SEQ = ["PT0", "PT3", "PT4", "PT3_hairpin", "PT4_hairpin"]
PLOT_TYPE = ["energy", "pair", "occp"]
VIS_METHOD = ["PCA", "PHATE", "UMAP", "tSNE"]

# # batch save plots for each vis method and plot type of each reaction
for seq in SEQ
    for plot_type in PLOT_TYPE
        for vis_method in VIS_METHOD
            # plot
            fname = "helix_assoc_$(seq)_multrj_100epoch_jl"
            statePlot(fname;seq=seq,plot_type=plot_type,vis_method=vis_method)
        end
    end
end


