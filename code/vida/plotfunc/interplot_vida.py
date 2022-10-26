import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

"""
Make ViDa plots
"""

###############################################################################
# get the ith single trajectory
###############################################################################

def plot_trj(trj_id,dfall,i,vis,dim):
        TRJ_ID = trj_id+1
        
        if i == 0:
                s = 0
                s_prime = TRJ_ID[i]
        elif i == len(trj_id):
                s = TRJ_ID[i-1]
                s_prime = len(dfall)
        else:
                s = TRJ_ID[i-1]
                s_prime = TRJ_ID[i]
        
        # get energy, pair, DP, HT, TotalT for each trajectory
        subdf = pd.DataFrame(data={
            "Energy": dfall["Energy"][s:s_prime],
            "Pair": dfall["Pair"][s:s_prime],
            "DP": dfall["DP"][s:s_prime],
            "HT": dfall["HT"][s:s_prime],
            "TotalT": dfall["TotalT"][s:s_prime],
            "correct_interpair": dfall["correct_interpair"][s:s_prime],
            "intrapair_top": dfall["intrapair_top"][s:s_prime],
            "intrapair_bot": dfall["intrapair_bot"][s:s_prime],
            "all_interpair": dfall["all_interpair"][s:s_prime],
            }
            )
        
        # get step numbers for the trajectory less than 2000 steps
        if len(subdf["DP"]) < 2000:
                Step = []
                for i in range(len(subdf["DP"])):
                        step=[]
                        for j in range(len(subdf["DP"])):
                                if subdf["DP"].iloc[i] == subdf["DP"].iloc[j]:
                                        step.append(j+1)
                        Step.append(step)
                subdf["Step"] = np.array(Step,dtype=object)
        else:
                subdf["Step"] = None
        
        # get X,Y,Z (if applicable) coordinates for the trajectory              
        if dim=="2D":
                subdf["sub X"] = dfall["{} 1".format(vis)][s:s_prime]
                subdf["sub Y"] = dfall["{} 2".format(vis)][s:s_prime]

                Zi=None
                Zf=None
                
        elif dim=="3D":
                subdf["sub X"] = dfall["{} X".format(vis)][s:s_prime]
                subdf["sub Y"] = dfall["{} Y".format(vis)][s:s_prime]
                subdf["sub Z"] = dfall["{} Z".format(vis)][s:s_prime]
                 
                Zi=subdf["sub Z"].iloc[0]
                Zf=subdf["sub Z"].iloc[-1]
                 
        # get initial and final coordinates: list structure
        Xi=subdf["sub X"].iloc[0]; Xf=subdf["sub X"].iloc[-1]
        Yi=subdf["sub Y"].iloc[0]; Yf=subdf["sub Y"].iloc[-1]
        
        return subdf,(Xi,Xf),(Yi,Yf),(Zi,Zf)


###############################################################################
# get more hover over information
###############################################################################

def hover_addon(SIMS_adj_uniq,SIMS_dict_uniq):
    correct_interpair_uniq = []
    intrapair_top_uniq = []
    intrapair_bot_uniq = []
    all_interpair_uniq = []

    for s_adj in SIMS_adj_uniq:
        count = 0
        for i in range(25):
            if s_adj[i,49-i] == 1.0:
                count += 1
        correct_interpair_uniq.append(count)
        
    for i in range(len(SIMS_dict_uniq)):
        ss = SIMS_dict_uniq[i][0].split("+")
        intrapair_top_uniq.append(ss[0].count(")")) # number of intra base pairs for both strands
        intrapair_bot_uniq.append(ss[1].count("(")) # number of intra base pairs for both strands
        all_interpair_uniq.append(ss[0].count("(") - ss[0].count(")")) # number of inter base pairs (correct+incorrect)
        
    return  np.array(correct_interpair_uniq), np.array(intrapair_top_uniq), np.array(intrapair_bot_uniq), np.array(all_interpair_uniq)


  
###############################################################################
# plot 2D energy landscape
###############################################################################

def interactive_plotly_2D(SEQ,n_trace,df,dfall,trj_id,vis):
    fig = go.Figure()
    
    # plot energy landscape background
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)], 
            mode='markers',
            marker=dict(
                sizemode='area',
                size=df["HT"],
                sizeref=5e-11,
                color=df["Energy"],
                colorscale="Plasma",
                showscale=True,
                # colorbar_x=-0.2,
                colorbar=dict(
                    title="Free energy (kcal/mol)",  
                    x=-0.2,
                    titleside="top",  
                    len=1.065,
                    y=0.5,
                ),
                line=dict(width=0.2
                ),
            ),
            customdata = np.stack((df['Pair'],
                                   np.log(df["Occp"]),
                                   df["correct_interpair"],
                                   df["intrapair_top"],
                                   df["intrapair_bot"],
                                   df["all_interpair"],
                                   ),axis=-1),
            text=df['DP'],
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{marker.color:.3f} kcal/mol<br><br>"+
                
                "Intra-strand base pairs (left strand):  %{customdata[3]}<br>"+
                "Intra-strand base pairs (right strand):  %{customdata[4]}<br>"+
                "Correctly paired inter-strand base pairs: %{customdata[2]}<br>"+
                "Total inter-strand base pairs: %{customdata[5]}<br>"+
                "Two strands binding: (0->No; 1->Yes):  %{customdata[0]}<br><br>"+

                "Average holding time:  %{marker.size:.5g} s<br>"+
                "Occupancy density (logscale):  %{customdata[1]:.3f}<br>",
    
            name="background",
            # showlegend=False,
        )
    )

    # layout trajectory on top of energy landscape
    for i in range(n_trace):
        subdf = plot_trj(trj_id,dfall,i,vis,dim="2D")[0]
        fig.add_trace(
            go.Scattergl(
                x=subdf["sub X"], 
                y=subdf["sub Y"],
                mode='lines+markers',
                line=dict(
                    # color='rgb({}, {}, {})'.format((i/100*255),(i/100*255),(i/100*255)),
                    color="black",
                    width=2,
                ),
                marker=dict(
                    sizemode='area',
                    size=subdf["HT"],
                    # sizeref=8e-11,
                    sizeref=5e-11,
                    color=subdf["Energy"],
                    colorscale="Plasma",
                    # showscale=False,
                    colorbar=dict(
                        x=-0.2,
                        tickvals=[],
                        y=0.5,
                        len=1,
                        ),
                ),
                
                text=subdf["Step"],
                customdata = np.stack((subdf['Pair'],
                                       subdf["TotalT"],
                                       subdf["DP"],
                                       subdf["correct_interpair"],
                                       subdf["intrapair_top"],
                                       subdf["intrapair_bot"],
                                       subdf["all_interpair"],
                                       ),axis=-1),
                hovertemplate=
                    "Step:  <b>%{text}</b><br><br>"+
                    "DP notation: <br> <b>%{customdata[2]}</b><br>" +
                    "X: %{x}   " + "   Y: %{y} <br>"+
                    "Energy:  %{marker.color:.3f} kcal/mol<br><br>"+
                    
                    "Intra-strand base pairs (left strand):  %{customdata[4]}<br>"+
                    "Intra-strand base pairs (right strand):  %{customdata[5]}<br>"+
                    "Correctly paired inter-strand base pairs: %{customdata[3]}<br>"+
                    "Total inter-strand base pairs: %{customdata[6]}<br>"+
                    "Two strands binding: (0->No; 1->Yes):  %{customdata[0]}<br><br>"+

                    "Holding time for last step:  %{marker.size:.5g} s<br>"+
                    "Total time until current state:  %{customdata[1]:.5e} s<br>",
                visible='legendonly'
                        )
        )

    # label initial and final states
    fig.add_trace(
        go.Scattergl(
            x=plot_trj(trj_id,dfall,0,vis,dim="2D")[1],
            y=plot_trj(trj_id,dfall,0,vis,dim="2D")[2],
            mode='markers+text',
            marker_color="lime", 
            marker_size=15,
            text=["I", "F"],
            textposition="middle center",
            textfont=dict(
            family="sans serif",
            size=16,
            color="black"
        ),
            hoverinfo='skip',
            showlegend=False,
                        )
    )
    
    fig.update_xaxes(
        range=[min(df["{} 1".format(vis)])*1.1,max(df["{} 1".format(vis)])*1.1]
    )
    
    fig.update_yaxes(
        range=[min(df["{} 2".format(vis)])*1.1,max(df["{} 2".format(vis)])*1.1]
    )

    fig.update_layout(
        # autosize=True,
        # width=700,
        # height=700,
        # margin=dict(
        #     l=50,
        #     r=50,
        #     b=100,
        #     t=100,
        #     pad=4
        # ),
        title="{}: {} Vis".format(SEQ,vis),
        xaxis=dict(
                title="{} 1".format(vis),
            ),
        yaxis=dict(
                title="{} 2".format(vis),
            ),
        legend=dict(
            title="Single Trajectory",
            title_font=dict(size=10),
            font=dict(
                # family="Courier",
                size=10,
                color="black"
        )
        )
    )
    
    return fig



###############################################################################
# plot 3D energy landscape
###############################################################################
def interactive_plotly_3D(SEQ,df,dfall,trj_id,vis):
    fig = go.Figure()
    
    # plot energy landscape background
    fig.add_trace(go.Scatter3d(
            x=df["{} X".format(vis)], 
            y=df["{} Y".format(vis)], 
            z=df["{} Z".format(vis)],
            mode='markers',
            marker=dict(
                sizemode='area',
                size=2,
                sizeref=1e-11,
                color=df["Energy"],
                colorscale="Plasma",
                showscale=True,
                colorbar_x=-0.2,
            ),
            customdata = np.stack((df['Pair'],np.log(df["Occp"]),df["HT"]),axis=-1),
            text=df['DP'],
            # hovertext=df['DP'],
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br><br>" +
                "X: %{x}   " + "   Y: %{y}   " + "   Z: %{z} <br>"+
                "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                "Average Holding Time:  %{customdata[2]:.5g} s<br>"+
                "Occupancy Density (logscale):  %{customdata[1]:.3f}<br>"+
                "Pair (0/1->unpaired/paired,resp):  %{customdata[0]}",
            name="background",
            # showlegend=False,
        )
    )

    # layout trajectory on top of energy landscape
    for i in range(100):
        subdf = plot_trj(trj_id,dfall,i,vis,dim="3D")[0]
        fig.add_trace(
            go.Scatter3d(
                x=subdf["sub X"], 
                y=subdf["sub Y"],
                z=subdf["sub Z"],
                mode='lines+markers',
                line=dict(
                    color="black",
                    width=1+i/100,
                ),
                marker=dict(
                    sizemode='area',
                    size=subdf["HT"],
                    sizeref=8e-11,
                    color=subdf["Energy"],
                    colorscale="Plasma",
                    # showscale=False,
                    colorbar_orientation='h',

                ),
                
                text=subdf["Step"],
                customdata = np.stack((subdf['Pair'],subdf["TotalT"],subdf["DP"]),axis=-1),
                hovertemplate=
                    "Step:  <b>%{text}</b><br><br>"+
                    "X: %{x}   " + "   Y: %{y}   "+ "   Z: %{z} <br>"+
                    "DP: %{customdata[2]}<br>" +
                    "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                    "Average Holding Time:  %{marker.size:.5g} s<br>"+
                    "Total Time:  %{customdata[1]:.5e} s <br>" +
                    "Pair (0/1->unpaired/paired,resp):  %{customdata[0]} ",
                visible='legendonly'
                        )
        )

    # label initial and final states
    fig.add_trace(
        go.Scatter3d(
            x=plot_trj(trj_id,dfall,0,vis,dim="3D")[1],
            y=plot_trj(trj_id,dfall,0,vis,dim="3D")[2],
            z=plot_trj(trj_id,dfall,0,vis,dim="3D")[3],
            mode='markers+text',
            marker_color="lime", 
            marker_size=12,
            text=["I", "F"],
            textposition="middle center",
            textfont=dict(
            family="sans serif",
            size=16,
            color="black"
        ),
            showlegend=False,
            hovertext=["Initial State", "Final State"],
            customdata=np.stack(([int(subdf['Pair'].iloc[0]),int(subdf['Pair'].iloc[-1])], 
                                 [subdf["TotalT"].iloc[0],subdf["TotalT"].iloc[-1]],
                                 [subdf["DP"].iloc[0],subdf["DP"].iloc[-1]], 
                                 [subdf["HT"].iloc[0],subdf["HT"].iloc[-1]],
                                 [subdf["Energy"].iloc[0],subdf["Energy"].iloc[-1]],),axis=-1),
            
            hovertemplate=
                "<b>%{hovertext}</b><br>"+
                "X: %{x}   "+"   Y: %{y}   "+"   Z: %{z} <br>"+
                "DP: %{customdata[2]}<br>" +
                "Energy:  %{customdata[4]:.3f} kcal/mol<br>"+
                "Average Holding Time:  %{customdata[3]:.5g} s<br>"+
                "Total Time:  %{customdata[1]:.5e} s <br>" +
                "Pair (0/1->unpaired/paired,resp):  %{customdata[0]} ",   
            name="I/F States",           
                        ),
    )
    
    fig.update_layout(
        # autosize=True,
        # width=700,
        # height=700,
        # margin=dict(
        #     l=50,
        #     r=50,
        #     b=100,
        #     t=100,
        #     pad=4
        # ),
        title="{}: {} Vis".format(SEQ,vis),
        scene = dict(
            xaxis_title="{} X".format(vis),
            yaxis_title="{} Y".format(vis),
            zaxis_title="{} Z".format(vis),
        ),
       legend=dict(
            title="Single Trajectory",
            title_font=dict(size=10),
            font=dict(
                # family="Courier",
                size=10,
                color="black"
        )
        )
    )
    
    return fig



###############################################################################
# separate paired and unpaired data 2D plots
###############################################################################

def interactive_plotly_2D_parivsunpair(SEQ,n_trace,df0,df1,dfall,trj_id,vis):
    fig = go.Figure()
    
    # plot energy landscape background
    fig.add_trace(go.Scattergl(
            x=df0["{} 1".format(vis)], 
            y=df0["{} 2".format(vis)], 
            mode='markers',
            marker=dict(
                symbol='x',
                # size=10,
                sizemode='area',
                size=df0["HT"],
                sizeref=1e-11,
                color=df0["Energy"],
                # colorscale="Plasma",
                colorscale="OrRd",
                showscale=True,
                colorbar=dict(
                    title="Free energy <br> (kcal/mol)",  
                    x=1.15,
                    titleside="top",  
                    # len=1.065,
                    # y=0.5,
                ),
            ),
            customdata = np.stack((df0['Pair'],
                                   np.log(df0["Occp"]),
                                   df0["HT"],
                                   df0["correct_interpair"],
                                   df0["intrapair_top"],
                                   df0["intrapair_bot"],
                                   df0["all_interpair"],
                                   ),axis=-1),
            text=df0['DP'],
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{marker.color:.3f} kcal/mol<br><br>"+
                
                "Intra-strand base pairs (left strand): %{customdata[4]} <br>"+
                "Intra-strand base pairs (right strand): %{customdata[5]} <br>"+
                "Correctly paired inter-strand base pairs: %{customdata[3]} <br>"+
                "Total inter-strand base pairs: %{customdata[6]} <br>"+
                "Two strands binding: (0->No; 1->Yes): %{customdata[0]} <br><br>"+
                
                "Average holding time:  %{customdata[2]:.5g} s<br>"+
                "Occupancy density (logscale):  %{customdata[1]:.3f}<br>",
            
            name="Unpaired",
            # showlegend=False,
        )
    )

    fig.add_trace(go.Scattergl(
            x=df1["{} 1".format(vis)], 
            y=df1["{} 2".format(vis)], 
            mode='markers',
            marker=dict(
                sizemode='area',
                # size=4,
                size=df1["HT"],
                sizeref=5e-11,
                color=df1["Energy"],
                # colorscale="Plasma",
                colorscale="Greens",
                opacity=0.5,
                showscale=True,
                colorbar=dict(
                    title="Free energy <br> (kcal/mol)",  
                    x=-0.2,
                    titleside="top",  
                    # len=1.065,
                    # y=0.5,
                ),                                
            ),
            customdata = np.stack((df1['Pair'],
                                   np.log(df1["Occp"]),
                                   df1["HT"],
                                   df1["correct_interpair"],
                                   df1["intrapair_top"],
                                   df1["intrapair_bot"],
                                   df1["all_interpair"],
                                   ),axis=-1),
            text=df1['DP'],
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{marker.color:.3f} kcal/mol<br><br>"+
                
                "Intra-strand base pairs (left strand): %{customdata[4]} <br>"+
                "Intra-strand base pairs (right strand): %{customdata[5]} <br>"+
                "Correctly paired inter-strand base pairs: %{customdata[3]} <br>"+
                "Total inter-strand base pairs: %{customdata[6]} <br>"+
                "Two strands binding: (0->No; 1->Yes): %{customdata[0]} <br><br>"+
                
                "Average holding time:  %{customdata[2]:.5g} s<br>"+
                "Occupancy density (logscale):  %{customdata[1]:.3f}<br>",

            name="Paired",
            # showlegend=False,
        )
    )

    # layout trajectory on top of energy landscape
    for i in range(n_trace):
        subdf = plot_trj(trj_id,dfall,i,vis,dim="2D")[0]
        fig.add_trace(
            go.Scattergl(
                x=subdf["sub X"], 
                y=subdf["sub Y"],
                mode='lines+markers',
                line=dict(
                    # color='rgb({}, {}, {})'.format((i/100*255),(i/100*255),(i/100*255)),
                    color="black",
                    width=2,
                ),
                marker=dict(
                    sizemode='area',
                    size=subdf["HT"],
                    sizeref=8e-11,
                    color=subdf["Energy"],
                    colorscale="plasma",
                    # showscale=False,
                    colorbar=dict(
                        orientation='h',
                        tickvals=[],
                        # showticklabels=False,
                        ),
                    # colorbar_orientation='h',
                ),
                
                text=subdf["Step"],
                customdata = np.stack((subdf['Pair'],
                                       subdf["TotalT"],
                                       subdf["DP"],
                                       subdf["correct_interpair"],
                                       subdf["intrapair_top"],
                                       subdf["intrapair_bot"],
                                       subdf["all_interpair"],
                                       ),axis=-1),
                hovertemplate=
                    "Step:  <b>%{text}</b><br><br>"+
                    "DP notation: <br> <b>%{customdata[2]}</b><br>" +
                    "X: %{x}   " + "   Y: %{y} <br>"+
                    "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                    
                    "Intra-strand base pairs (left strand): %{customdata[4]} <br>"+
                    "Intra-strand base pairs (right strand): %{customdata[5]} <br>"+
                    "Correctly paired inter-strand base pairs: %{customdata[3]} <br>"+
                    "Total inter-strand base pairs: %{customdata[6]} <br>"+
                    "Two strands binding: (0->No; 1->Yes): %{customdata[0]} <br><br>"+
                    
                    "Holding time for last step:  %{marker.size:.5g} s<br>"+
                    "Total time until current state:  %{customdata[1]:.5e} s<br>",
                    
                visible='legendonly',
                name=f"trace {i+1}",
                        ),
        )

    # label initial and final states
    fig.add_trace(
        go.Scattergl(
            x=plot_trj(trj_id,dfall,0,vis,dim="2D")[1],
            y=plot_trj(trj_id,dfall,0,vis,dim="2D")[2],
            mode='markers+text',
            marker_color="lime", 
            marker_size=15,
            text=["I", "F"],
            textposition="middle center",
            textfont=dict(
            family="sans serif",
            size=16,
            color="black"
        ),
            hoverinfo='skip',
            showlegend=False,
                        )
    )
    
    fig.update_xaxes(
        range=[min(dfall["{} 1".format(vis)])*1.1,max(dfall["{} 1".format(vis)])*1.1]
    )
    
    fig.update_yaxes(
        range=[min(dfall["{} 2".format(vis)])*1.1,max(dfall["{} 2".format(vis)])*1.1]
    )

    fig.update_layout(
        title="{}: {} Vis".format(SEQ,vis),
        xaxis=dict(
                title="{} 1".format(vis),
            ),
        yaxis=dict(
                title="{} 2".format(vis),
            ),
        legend=dict(
            title="Single Trajectory",
            title_font=dict(size=10),
            font=dict(
                # family="Courier",
                size=10,
                color="black"
        )
        ),
        # hovermode="x",
    )
    
    return fig



###############################################################################
# get states density heatmap
###############################################################################

def interactive_density_heatmap(SEQ,df,dfall,trj_id,vis,state_type):
    if state_type == "unique":
        # get unique states' density heatmap
        fig = px.density_heatmap(
                df, x=f"{vis} 1", y=f"{vis} 2",
                text_auto=True, 
                title=f"{SEQ} by {vis}: Structure Density Heatmap - No Repetition",
                )
    elif state_type == "all":
        # get all states' density heatmap
        fig = px.density_heatmap(
            dfall, x=f"{vis} 1", y=f"{vis} 2",
            histfunc = "count",
            text_auto=True,
            title=f"{SEQ} by {vis}: Structure Density Heatmap ",
            )
    
    # label initial and final states
    fig.add_trace(
        go.Scattergl(
            x=plot_trj(trj_id,dfall,0,vis,dim="2D")[1],
            y=plot_trj(trj_id,dfall,0,vis,dim="2D")[2],
            mode='markers+text',
            marker_color="lime", 
            marker_size=20,
            text=["I", "F"],
            textposition="middle center",
            textfont=dict(
            family="sans serif",
            size=16,
            color="black"
        ),
            hoverinfo='skip',
            showlegend=False,
                        )
    )
    
    return fig



###################### OLD CODE ################################
# old hard-code interactive plot function
###################### OLD CODE ################################

def interactive_plot_old(data_npz,SEQ,vis):
    if vis == "PCA":
        coords = "pca_coords"
        all_coords = "pca_all_coords"
    elif vis == "PHATE":
        coords = "phate_coords"
        all_coords = "phate_all_coords"
    elif vis == "UMAP":
        coords = "umap_coord_2d"
        all_coords = "umap_all_coord_2d"
    elif vis == "tSNE":
        coords = "tsne_coord_2d"
        all_coords = "tsne_all_coord_2d"

    # get coordinates of the data points
    X = data_npz[coords][:,0]
    Y = data_npz[coords][:,1]
    # get coordinates of Si and Sf
    X_i = data_npz[all_coords][0][0]; Y_i = data_npz[all_coords][0][1]
    X_f = data_npz[all_coords][-1][0]; Y_f = data_npz[all_coords][-1][1]
    # get hover text
    dp_annot = data_npz["SIMS_dict_uniq"][:,0]
    energy_annot = data_npz["SIMS_G_uniq"]

    # figure setup
    fig,ax = plt.subplots(figsize=(10,6))
    fig.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.0,
        right=0.75,
        hspace=0.2,
        wspace=0.2
        )

    # scatter plot
    sc = plt.scatter(X, Y,
            c=data_npz["SIMS_G_uniq"],
            cmap='plasma',
            s=12
            )
    
    cbar = plt.colorbar(sc,location ='left') # show colorbar
    cbar.set_label("Free energy",fontsize=15)
    plt.title("Energy landscape of strands {} via {}".format(SEQ,vis),fontsize=18)
    plt.xlabel("X")
    plt.ylabel("Y")

    # annotations
    annotations_IF=["I","F"]
    x = [X_i,X_f]
    y = [Y_i,Y_f]
    plt.scatter(x,y,s=60, c="lime", alpha=1)
    for i, label in enumerate(annotations_IF):
        plt.annotate(label, (x[i],y[i]),fontsize=12,c="black",fontweight="bold",
                     horizontalalignment='center',verticalalignment='center')

    # mouse over
    annot = ax.annotate("", 
                        xy=(0,0), 
                        xytext=(10,10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"),
                        fontsize=8,
                        )
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "POS: (x={0:.3f}, y={1:.3f}) \nEnergy: {2:.3f} kcal/mol \nDP: ".format(
                    pos[0], 
                    pos[1],
                    energy_annot[ind["ind"]][0]
                    ) + r"$\bf{}$".format(
                    dp_annot[ind["ind"]][0],
                    )
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.7)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()