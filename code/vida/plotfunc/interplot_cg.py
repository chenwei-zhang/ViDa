import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px

"""
Make coarse-grained plots
"""

###############################################################################
# get coarse-grainning landscapes and trajectory plots
###############################################################################

def plot_trj_cg(i, trj_id,correct_interpair,not_infinalstructure):
        TRJ_ID_cg = trj_id+1
        
        if i == 0:
                s = 0
                s_prime = TRJ_ID_cg[i]
        elif i == len(trj_id):
                s = TRJ_ID_cg[i-1]
                s_prime = len(dfall)
        else:
                s = TRJ_ID_cg[i-1]
                s_prime = TRJ_ID_cg[i]
                
        X = correct_interpair[s:s_prime]
        Y = not_infinalstructure[s:s_prime]
        
        return X,Y
    
def grid_energy(correct_interpair, not_infinalstructure, SIMS_G):
    XY = np.stack((correct_interpair,not_infinalstructure),axis=1)
    grid_G = np.zeros((26,26))
    
    for i in range(26):
        for j in range(26):
            xy = XY==[i,j]
            idx=np.argwhere(xy[:,0] & xy[:,1])
            ij_g = SIMS_G[idx].mean()
            if np.isnan(ij_g):
                grid_G[i,j] = 0
            else:
                grid_G[i,j] = ij_g
                
    return grid_G

def interactive_cgplot(SEQ,n_trace,grid_G,trj_id,correct_interpair,not_infinalstructure):
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x = np.arange(26),
        y = np.arange(26),
        z=grid_G.T,
        
        showscale=True,
        colorbar=dict(
            title="Free energy (kcal/mol)",
            titleside="right",
            x=-0.2,
        ),
        # colorscale = 'Viridis',
        colorscale = 'plasma',
        hovertemplate=
                "# bp in final:  %{x}<br>" + 
                "# bp NOT in final:  %{y}<br>" + 
                "Average energy:  %{z:.5f} kcal/mol<br>",
        name="background",
        showlegend=True,
                )
    )

    for i in range(n_trace):
        X,Y = plot_trj_cg(i, trj_id,correct_interpair,not_infinalstructure)
        fig.add_trace(go.Scatter(
            x=X,
            y=Y,
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=True,
            hoverinfo='all',
            visible='legendonly'
        )
                      )
        
        
    # label initial and final states
    fig.add_trace(
        go.Scattergl(
            x=[correct_interpair[0], correct_interpair[-1]],
            y=[not_infinalstructure[0], not_infinalstructure[-1]],
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
        range=[-1,26]
    )
    
    fig.update_yaxes(
        range=[-1,26]
    )

    fig.update_layout(
            title=f"{SEQ}: Coarsed-grained plot",
            xaxis=dict(
                    title="Number of base pairs in the final structure",
                ),
            yaxis=dict(
                    title="Number of base pairs  NOT in the final structure",
                ),
            autosize=True,
            legend=dict(
                title="Single Trajectory",
                title_font=dict(size=10),
                font=dict(
                    size=10,
                    color="black"
            )
            )
        )
    
    return fig

        