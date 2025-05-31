import numpy as np
import pandas as pd
import plotly.graph_objects as go



def map_id_to_shortname(id):
    if id == [(0, 1, 2)]:
        return 'Incumbent+Substrate+Invader'
    elif id == [(0, 1), (2,)]:
        return 'Incumbent+Substrate  Invader'
    elif id == [(0,), (1, 2)]:
        return 'Incumbent  Substrate+Invader'
    elif id == [(0, 2, 1)]:
        return 'Incumbent+Invader+Substrate'
    elif id == [(0, 2), (1,)]:
        return 'Incumbent+Invader  Substrate'
    elif id == [(0,), (2, 1)]:
        return 'Incumbent  Invader+Substrate'
    else:
        raise ValueError("Invalid reaction ordering")
    


def sort_machinek(plt_args):
    # Load the data
    trj_id, dp_og, trans_time, hold_time, energy, cum_time, freq, \
        pca_coords, phate_coords, order_ids, \
        dp_og_uniq, hold_time_uniq, energy_uniq, cum_time_uniq, freq_uniq, \
        pca_coords_uniq, phate_coords_uniq, id_uniq_name, \
        = plt_args
        
    # List of arrays to split
    arrays_to_split = [dp_og, trans_time, energy, pca_coords, phate_coords, order_ids]
    # Get each trajectory using a single loop
    subtrj_id = (trj_id+1)[:-1]
    sub_arrays = [np.split(arr, subtrj_id) for arr in arrays_to_split]
    # Process each trajectory to get unique states
    processed_arrays = [[] for _ in range(len(arrays_to_split))]
    
    for i in range(len(sub_arrays[0])):  # For each trajectory
        dp_og_i = sub_arrays[0][i]
        
        # Get unique states and their indices
        _, idx = np.unique(dp_og_i, axis=0, return_index=True)
        
        # Apply uniqueness to relevant arrays
        processed_arrays[0].append(dp_og_i[idx])           # dp_og (unique)
        processed_arrays[1].append(sub_arrays[1][i])       # trans_time (all)
        processed_arrays[2].append(sub_arrays[2][i][idx])  # energy (unique)
        processed_arrays[3].append(sub_arrays[3][i][idx])  # pca_coords (unique)
        processed_arrays[4].append(sub_arrays[4][i][idx])  # phate_coords (unique)
        processed_arrays[5].append(sub_arrays[5][i][idx])  # order_ids (unique)
    
    # Sort trajectories by reaction time (last element of trans_time)
    sorted_indices = np.argsort([arr[-1] for arr in processed_arrays[1]])[::-1]
    # Apply sorting to all arrays
    sorted_arrays = [np.array(arr, dtype=object)[sorted_indices] for arr in processed_arrays]
    # Unpack the sorted arrays into separate variables
    sorted_sub_dp_og, sorted_sub_trans_time, sorted_sub_energy, sorted_sub_pca_coords, sorted_sub_phate_coords, sorted_sub_order_ids = sorted_arrays
    
    # make dataframe for plotting   
    df = pd.DataFrame(data={
                "Energy": energy_uniq, "DP": dp_og_uniq, "HT": hold_time_uniq,
                "CumT": cum_time_uniq, "Freq": freq_uniq,
                "PCA 1": pca_coords_uniq[:,0], "PCA 2": pca_coords_uniq[:,1],
                "PHATE 1": phate_coords_uniq[:,0], "PHATE 2": phate_coords_uniq[:,1],
                "ID_Name": id_uniq_name,
                }
                )
    dfall = pd.DataFrame(data={
            "Energy": sorted_sub_energy, "DP": sorted_sub_dp_og, "TransT": sorted_sub_trans_time, 
            "PCA": sorted_sub_pca_coords, "PHATE": sorted_sub_phate_coords,
            "IDX": sorted_indices, "OrderID": sorted_sub_order_ids,
            }
            )
    return df, dfall




###############################################################################
# plot 2D landscape (sorted)
###############################################################################

def plot_machineck(df,dfall,vis):
    fig = go.Figure()
    
    # plot energy landscape background
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)], 
            mode='markers',
            marker=dict(
                sizemode='diameter',
                # size=df["HT"],
                # sizeref=1e-8,
                size=5,
                color=df["Energy"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(
                    title="Free energy (kcal/mol)",  
                    x=-0.2,
                    titleside="top",  
                    len=1.065,
                    y=0.5,
                ),
                line=dict(width=0),
            ),
            text=df['DP'],
            customdata=np.stack((
                    df['HT'],
                    df['ID_Name'],
                    ),axis=-1),
            hovertemplate=
                "%{customdata[1]}<br>" +
                "<b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                "Expected holding time:  %{customdata[0]:.3e} s<br>",
            name="Energy landscape",
            # visible='legendonly',
        )
    )

    # plot cumulative time landscape background
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)],
            mode='markers',
            marker=dict(
                sizemode='diameter',
                size=df["CumT"],
                sizeref=2e-4,  # PRF: 5e-3,
                color=df["Energy"], 
                colorscale="Plasma",
                showscale=False,
                line=dict(width=0),
            ),
            text=df['DP'],
            customdata=np.stack((
                df["Energy"],
                df["Freq"],
                df["ID_Name"],
                ),axis=-1),
            hovertemplate=
                "%{customdata[2]}<br>" +
                "<b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[0]:.3f} kcal/mol<br>"+
                "Cumulative time:  %{marker.size:.3e} s<br>"+
                "Frequency:  %{customdata[1]:d} <br>",
            name="Cumu_time landscape",
            visible='legendonly',
        )
    )
    
    # plot frequency landscape background
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)],
            mode='markers',
            marker=dict(
                sizemode='diameter',
                size=df["Freq"],
                sizeref=20000, # PRF: 25000,  mm14: 6500, 
                color=df["Energy"],
                colorscale="Plasma",
                showscale=False,
                line=dict(width=0),
            ),
            text=df['DP'],
            customdata=np.stack((
                df["Energy"],
                df["CumT"],
                df["ID_Name"],
                ),axis=-1),
            hovertemplate=
                "%{customdata[2]}<br>" +
                "<b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[0]:.3f} kcal/mol<br>"+
                "Cumulative time:  %{customdata[1]:.3e} s<br>"+
                "Frequency:  %{marker.size:d} <br>",
            name="Frequency landscape",
            visible='legendonly',
        )
    )
    
    # # layout trajectory on top of energy landscape
    # for i in range(0, len(dfall)):
    #     fig.add_trace(
    #         go.Scattergl(
    #             x=dfall[f"{vis}"][i][:,0],
    #             y=dfall[f"{vis}"][i][:,1],
    #             mode='lines+markers',
    #             line=dict(
    #                 color='rgba(0,0,0,0.6)',
    #                 width=1,
    #             ),
    #             marker=dict(
    #                 sizemode='diameter',
    #                 size=4.5,
    #                 color=dfall["Energy"][i],
    #                 colorscale="Plasma",
    #                 # color=[color_mapping[type_val] for type_val in dfall["ShortName"][i]],
    #                 # colorbar=dict(
    #                 #     x=-0.2,
    #                 #     y=0.5,
    #                 #     tickvals=[],
    #                 #     len=1,
    #                 # ),
    #             ),
    #             customdata=np.stack((
    #                 dfall['DP'][i],
    #                 dfall['Energy'][i],
    #             ),axis=-1),
    #             hovertemplate=
    #                 "<b>%{customdata[0]}<br>" +
    #                 "X: %{x}   " + "   Y: %{y} <br>"+
    #                 "Energy:  %{customdata[1]:.3f} kcal/mol<br>",
    #             visible='legendonly',
    #             name = "Trace {}".format(dfall["IDX"][i]),
    #         )
    #     )
    
    # Record the first successful trajectory's index
    for i in range(0, len(dfall)):
        if dfall["OrderID"][i][-1] == [(0,), (1, 2)] or dfall["OrderID"][i][-1] == [(0,),(2, 1)]:
            succ_idx = i
            break
        else:
            succ_idx = 0 # no successful traj, just take the first traj
    
    # label initial  # and final states
    fig.add_trace(
        go.Scattergl(
            x=[dfall[f"{vis}"][succ_idx][0,0],dfall[f"{vis}"][succ_idx][-1,0]],
            y=[dfall[f"{vis}"][succ_idx][0,1],dfall[f"{vis}"][succ_idx][-1,1]],
            # x=[dfall[f"{vis}"][0][0,0]],
            # y=[dfall[f"{vis}"][0][0,1]],
            mode='markers+text',
            marker_color="lime", 
            marker_size=20,
            text=["I", "F"],
            # text=["I"],
            textposition="middle center",
            textfont=dict(
            family="sans serif",
            size=15,
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
        title="ViDa-{} Vis".format(vis),
        xaxis=dict(
                title="{} 1".format(vis),
            ),
        yaxis=dict(
                title="{} 2".format(vis),
            ),
        legend=dict(
            title_font=dict(size=10),
            font=dict(
                size=10,
                color="black"
                )
            )
    )
    
    return fig
  


def plot_machineck_png(df, dfall, vis, output_dir):
    import os
    from plotly.io import write_image
    import plotly.io as pio
    
    pio.renderers.default = 'png'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the base figure with energy landscape background
    for i in range(0, len(dfall)):
        print(f"Plotting trajectory {dfall['IDX'][i]}")
        
        # Create a new figure for each trajectory
        fig = go.Figure()
        
        # Add energy landscape background
        fig.add_trace(go.Scattergl(
                x=df["{} 1".format(vis)], 
                y=df["{} 2".format(vis)], 
                mode='markers',
                marker=dict(
                    sizemode='diameter',
                    size=4.5,
                    color='rgba(211,211,211,0.8)',
                    line=dict(width=0),
                ),
                text=df['DP'],
                customdata=np.stack((
                        df['HT'],
                        ),axis=-1),
                hovertemplate=
                    "<b>%{text}</b><br>" +
                    "X: %{x}   " + "   Y: %{y} <br>"+
                    "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                    "Expected holding time:  %{customdata[0]:.3e} s</b><br>",
                name="Full Energy landscape",
            )
        )
        
        # Add just the line trace for trajectories
        fig.add_trace(
            go.Scattergl(
                x=dfall[f"{vis}"][i][:,0],
                y=dfall[f"{vis}"][i][:,1],
                mode='lines',      # Only lines, no markers
                line=dict(
                    color='rgba(0,0,0,0.6)',
                    width=1,
                ),
                hoverinfo='skip',  # Skip hover on lines for speed
                name = "Trace {}".format(dfall["IDX"][i]),
            )
        )
        
        # For markers, find unique positions to reduce redundancy
        points = dfall[f"{vis}"][i]
        energies = dfall["Energy"][i]
        dp_values = dfall["DP"][i]
        
        point_dtype = [('x', float), ('y', float)]
        unique_points_structured = np.array([(p[0], p[1]) for p in points], dtype=point_dtype)
        unique_indices = np.unique(unique_points_structured, return_index=True)[1]
        
        unique_indices = np.sort(unique_indices)
        unique_points = points[unique_indices]
        unique_energies = energies[unique_indices] if len(energies) == len(points) else energies
        unique_dp = dp_values[unique_indices] if len(dp_values) == len(points) else dp_values
        print(f"Reduced marker points from {len(points)} to {len(unique_points)}")
        
        # Add markers for unique states only
        fig.add_trace(
            go.Scattergl(
                x=unique_points[:,0],
                y=unique_points[:,1],
                mode='markers',
                marker=dict(
                    sizemode='diameter',
                    size=4.5,
                    color=unique_energies,
                    colorscale="Plasma",
                ),
                customdata=np.stack((
                    unique_dp,
                    unique_energies,
                ),axis=-1),
                hovertemplate=
                    "<b>%{customdata[0]}<br>" +
                    "X: %{x}   " + "   Y: %{y} <br>"+
                    "Energy:  %{customdata[1]:.3f} kcal/mol<br>",
                name = "States in Trace {}".format(dfall["IDX"][i]),
            )
        )
        
        # Add initial and final state markers
        fig.add_trace(
            go.Scattergl(
                x=[dfall[f"{vis}"][i][0,0],dfall[f"{vis}"][i][-1,0]],
                y=[dfall[f"{vis}"][i][0,1],dfall[f"{vis}"][i][-1,1]],
                mode='markers+text',
                marker_color="lime", 
                marker_size=20,
                text=["I", "F"],
                textposition="middle center",
                textfont=dict(
                family="sans serif",
                size=15,
                color="black"
                ),
                hoverinfo='skip',
                showlegend=False,
            )
        )

        # Set axis ranges
        fig.update_xaxes(
            range=[min(df["{} 1".format(vis)])*1.1,max(df["{} 1".format(vis)])*1.1]
        )
        fig.update_yaxes(
            range=[min(df["{} 2".format(vis)])*1.1,max(df["{} 2".format(vis)])*1.1]
        )
        
        # Add annotations (optional)
        if dfall["OrderID"][i][-1] == [(0,), (1, 2)] or dfall["OrderID"][i][-1] == [(0,),(2, 1)]:
            text_ammo = "SUCCESS"
        else:
            text_ammo = "FAILURE"
                
        # Update layout
        fig.update_layout(
            title=f"Trajectory {dfall['IDX'][i]} - ViDa-{vis} Vis | " + 
                    f"{text_ammo} | " +
                    f"Total elementary steps: {len(dfall[f'{vis}'][i])} | " +
                    f"Total reaction time: {dfall['TransT'][i][-1]:.3e} s",
            xaxis=dict(
                    title="{} 1".format(vis),
                ),
            yaxis=dict(
                    title="{} 2".format(vis),
                ),
            legend=dict(
                title_font=dict(size=10),
                font=dict(
                    size=10,
                    color="black"
                    )
                ),
        )
       
        # Save the figure as a PNG file
        output_file = os.path.join(output_dir, f"{vis}_{i}_trajectory-{dfall['IDX'][i]}-{text_ammo}.png")
        write_image(fig, output_file, width=1200, height=800, engine="kaleido")                
        
        print(f"Saved trajectory {dfall['IDX'][i]} to {output_file}")
        
        if i == 5:
            break