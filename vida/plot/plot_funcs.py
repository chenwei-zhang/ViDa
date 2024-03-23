import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Function to replace multiple long strings with their short notations
def replace_strings(s):
    
    # For Machinek-PRF
    invader = "GGTGAGTTTGAGGTTGAATGTGGA"
    incumbent = "TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA"
    substrate = "CCCTCCACATTCAACCTCAAACTCACC"
    
    s = s.replace(invader, "Invader")
    s = s.replace(incumbent, "Incumbent")
    s = s.replace(substrate, "Substrate")
    
    return s



def sort_gao(plt_args):
    # Load the data
    trj_id, dp_og, trans_time, hold_time, energy, pair, cum_time, freq, \
        pca_coords, phate_coords, \
        dp_og_uniq, hold_time_uniq, energy_uniq, pair_uniq, cum_time_uniq, freq_uniq, \
        pca_coords_uniq, phate_coords_uniq = plt_args
        
    # List of arrays to split
    arrays_to_split = [dp_og, trans_time, hold_time, energy, pair, cum_time, freq, pca_coords, phate_coords]
    # Get each trajectory using a single loop
    subtrj_id = (trj_id+1)[:-1]
    sub_arrays = [np.split(arr, subtrj_id) for arr in arrays_to_split]
    # Use zip to unpack the sub-arrays into separate variables if needed
    sub_dp_og, sub_trans_time, sub_hold_time, sub_energy, sub_pair, sub_cum_time, sub_freq, \
        sub_pca_coords, sub_phate_coords = sub_arrays
    # Sort the trajectories by reaction time
    sorted_indices = np.argsort([sub_array[-1] for sub_array in sub_trans_time])[::-1]
    # Use list comprehension and zip to sort all arrays simultaneously
    sorted_arrays = [np.array(arr,dtype=object)[sorted_indices] for arr in sub_arrays] 
    # Unpack the sorted arrays into separate variables
    sorted_sub_dp_og, sorted_sub_trans_time, sorted_sub_hold_time, sorted_sub_energy, sorted_sub_pair, \
        sorted_sub_cum_time, sorted_sub_freq, sorted_sub_pca_coords, sorted_sub_phate_coords = sorted_arrays
        
    # make dataframe for plotting   
    df = pd.DataFrame(data={
                "Energy": energy_uniq, "Pair": pair_uniq, "DP": dp_og_uniq, "HT": hold_time_uniq,
                "CumT": cum_time_uniq, "Freq": freq_uniq,
                "PCA 1": pca_coords_uniq[:,0], "PCA 2": pca_coords_uniq[:,1],
                "PHATE 1": phate_coords_uniq[:,0], "PHATE 2": phate_coords_uniq[:,1]
                }
                )

    dfall = pd.DataFrame(data={
            "Energy": sorted_sub_energy, "Pair": sorted_sub_pair, "DP": sorted_sub_dp_og, "HT": sorted_sub_hold_time, 
            "TransT": sorted_sub_trans_time, "CumT": sorted_sub_cum_time, "Freq": sorted_sub_freq,
            "PCA": sorted_sub_pca_coords, "PHATE": sorted_sub_phate_coords,
            "IDX": sorted_indices
            }
            )
    
    return df, dfall
    
#TODO
def sort_hata(plt_args):
    # Load the data
    trj_id, dp_og, trans_time, hold_time, energy, pair, pca_coords, phate_coords, energy_uniq, pair_uniq, dp_og_uniq, hold_time_uniq, pca_coords_uniq, phate_coords_uniq = plt_args
    # get each trajectory for Gao reaction
    # List of arrays to split
    arrays_to_split = [dp_og, trans_time, hold_time, energy, pair, pca_coords, phate_coords]
    # Get each trajectory using a single loop
    subtrj_id = (trj_id+1)[:-1]
    sub_arrays = [np.split(arr, subtrj_id) for arr in arrays_to_split]
    # Use zip to unpack the sub-arrays into separate variables if needed
    sub_dp_og, sub_trans_time, sub_hold_time, sub_energy, sub_pair, sub_pca_coords, sub_phate_coords = sub_arrays
    


def sort_machinek(plt_args):
    # Load the data
    trj_id, dp_og, trans_time, hold_time, energy, pair, cum_time, freq, \
        pca_coords, phate_coords, \
        dp_og_uniq, hold_time_uniq, energy_uniq, pair_uniq, cum_time_uniq, freq_uniq, \
        pca_coords_uniq, phate_coords_uniq, \
        seq_uniq, shortname_uniq, incbinvpair_uniq, \
        seq, shortname, incbinvpair \
        = plt_args
    
    vectorized_replace = np.vectorize(replace_strings)
    seqlabel_uniq = vectorized_replace(seq_uniq)
    seqlabel = vectorized_replace(seq)

    # List of arrays to split
    arrays_to_split = [dp_og, trans_time, hold_time, energy, pair, cum_time, freq, pca_coords, phate_coords, seqlabel, shortname, incbinvpair]
    # Get each trajectory using a single loop
    subtrj_id = (trj_id+1)[:-1]
    sub_arrays = [np.split(arr, subtrj_id) for arr in arrays_to_split]
    # Use zip to unpack the sub-arrays into separate variables if needed
    sub_dp_og, sub_trans_time, sub_hold_time, sub_energy, sub_pair, sub_cum_time, sub_freq, \
        sub_pca_coords, sub_phate_coords, sub_seqlabel, sub_shortname, sub_incbinvpair = sub_arrays
    # Sort the trajectories by reaction time
    sorted_indices = np.argsort([sub_array[-1] for sub_array in sub_trans_time])[::-1]
    # Use list comprehension and zip to sort all arrays simultaneously
    sorted_arrays = [np.array(arr,dtype=object)[sorted_indices] for arr in sub_arrays] 
    # Unpack the sorted arrays into separate variables
    sorted_sub_dp_og, sorted_sub_trans_time, sorted_sub_hold_time, sorted_sub_energy, sorted_sub_pair, \
        sorted_sub_cum_time, sorted_sub_freq, sorted_sub_pca_coords, sorted_sub_phate_coords, sorted_sub_seqlabel, \
            sorted_sub_shortname, sorted_sub_incbinvpair = sorted_arrays
        
        
    # make dataframe for plotting   
    df = pd.DataFrame(data={
                "Energy": energy_uniq, "Pair": pair_uniq, "DP": dp_og_uniq, "HT": hold_time_uniq,
                "CumT": cum_time_uniq, "Freq": freq_uniq, "SeqLabel": seqlabel_uniq,
                "PCA 1": pca_coords_uniq[:,0], "PCA 2": pca_coords_uniq[:,1],
                "PHATE 1": phate_coords_uniq[:,0], "PHATE 2": phate_coords_uniq[:,1],
                "IncbInvPair": incbinvpair_uniq, "ShortName": shortname_uniq,
                }
                )

    dfall = pd.DataFrame(data={
            "Energy": sorted_sub_energy, "Pair": sorted_sub_pair, "DP": sorted_sub_dp_og, "HT": sorted_sub_hold_time, 
            "TransT": sorted_sub_trans_time, "CumT": sorted_sub_cum_time, "Freq": sorted_sub_freq,  "SeqLabel": sorted_sub_seqlabel,
            "PCA": sorted_sub_pca_coords, "PHATE": sorted_sub_phate_coords,
            "IncbInvPair": sorted_sub_incbinvpair, "ShortName": sorted_sub_shortname,
            "IDX": sorted_indices,
            }
            )
    
    return df, dfall






###############################################################################
# plot 2D landscape (sorted)
###############################################################################

def plot_gao(df,dfall,vis):
    fig = go.Figure()
    
    # plot energy landscape background
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)], 
            mode='markers',
            marker=dict(
                sizemode='diameter',
                size=df["HT"],
                sizeref=1e-9,
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
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{marker.color:.3f} kcal/mol<br><br>"+
                "Expected holding time:  %{marker.size:.3e} s<br>",
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
                sizeref=5e-8,
                color=df["Energy"],
                colorscale="Plasma",
                showscale=False,
                line=dict(width=0),
            ),
            text=df['DP'],
            customdata=np.stack((
                df["Energy"],
                df["Freq"],
                ),axis=-1),
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[0]:.3f} kcal/mol<br><br>"+
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
                sizeref=50,
                color=df["Energy"],
                colorscale="Plasma",
                showscale=False,
                line=dict(width=0),
            ),
            text=df['DP'],
            customdata=np.stack((
                df["Energy"],
                df["CumT"],
                ),axis=-1),
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[0]:.3f} kcal/mol<br><br>"+
                "Cumulative time:  %{customdata[1]:.3e} s<br>"+
                "Frequency:  %{marker.size:d} <br>",
            name="Frequency landscape",
            visible='legendonly',
        )
    )
    
    # plot bounded/unbounded background
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)],
            mode='markers',
            marker=dict(
                sizemode='diameter',
                size=4,
                color=df["Pair"],
                showscale=False,
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
            customdata=np.stack((df["CumT"],
                                 df["Energy"],
                                 df["Freq"],
                                     ),axis=-1),
            hovertemplate=
                "DP notation: <br> <b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[1]:.3f} kcal/mol<br>"+
                "Cumulative time:  %{customdata[0]:.3e} s<br>"+
                "Frequency:  %{customdata[2]:d} <br>",
            name="Bound/Unbound",
            visible='legendonly',
        )
    )

    # layout trajectory on top of energy landscape
    for i in range(len(dfall)):
        Step = []
        if len(dfall["DP"][i]) < 2000:
            Step = np.arange(len(dfall["DP"][i]))
        else:
            Step = np.full(len(dfall["DP"][i]), None, dtype=object)
        
        fig.add_trace(
            go.Scattergl(
                x=dfall[f"{vis}"][i][:,0],
                y=dfall[f"{vis}"][i][:,1],
                mode='lines+markers',
                line=dict(
                    color="black",
                    width=1,
                ),
                marker=dict(
                    sizemode='diameter',
                    size=2,
                    color=dfall["Energy"][i],
                    colorscale="Plasma",
                    colorbar=dict(
                        x=-0.2,
                        y=0.5,
                        tickvals=[],
                        len=1,
                    ),
                ),
                text=Step,
                customdata=np.stack((dfall['Pair'][i],
                                     dfall['TransT'][i],
                                     dfall['DP'][i],
                                     dfall['HT'][i],
                                     ),axis=-1),
                hovertemplate=
                    "Step:  <b>%{text}</b><br><br>"+
                    "DP notation: <br> <b>%{customdata[2]}</b><br>" +
                    "X: %{x}   " + "   Y: %{y} <br>"+
                    "Energy:  %{marker.color:.3f} kcal/mol<br><br>"+
                    "Expected holding time:  %{customdata[3]:.3e} s<br>"+
                    "Total transition time until current state:  %{customdata[1]:.3e} s<br>",
                visible='legendonly',
                name = "Trace {}".format(dfall["IDX"][i]),
            )
        )

    # label initial and final states
    fig.add_trace(
        go.Scattergl(
            x=[dfall[f"{vis}"][0][0,0],dfall[f"{vis}"][0][-1,0]],
            y=[dfall[f"{vis}"][0][0,1],dfall[f"{vis}"][0][-1,1]],
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


    
# TODO  ## see jupyter notebook for details ##
def plot_hata():
    pass



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
                    df['SeqLabel'],
                    df['HT'],
                    df['ShortName'],
                    df['IncbInvPair'],
                    ),axis=-1),
            hovertemplate=
                # "<b>%{customdata[0]}</b><br>" +
                "<b>%{customdata[2]}</b><br>" +
                "<b>%{text}</b><br><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                "Expected holding time:  %{customdata[1]:.3e} s<br>"+
                "IncbInvPair: <b>%{customdata[3]}</b><br>",
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
                sizeref=5e-3,
                color=df["Energy"],
                colorscale="Plasma",
                showscale=False,
                line=dict(width=0),
            ),
            text=df['DP'],
            customdata=np.stack((
                df["Energy"],
                df["Freq"],
                df['ShortName'],
                ),axis=-1),
            hovertemplate=
                "<b>%{customdata[2]}</b><br>" +
                "<b>%{text}</b><br><br>" +
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
                sizeref=2500,
                color=df["Energy"],
                colorscale="Plasma",
                showscale=False,
                line=dict(width=0),
            ),
            text=df['DP'],
            customdata=np.stack((
                df["Energy"],
                df["CumT"],
                df['ShortName'],
                ),axis=-1),
            hovertemplate=
                "<b>%{customdata[2]}</b><br>" +
                "<b>%{text}</b><br><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[0]:.3f} kcal/mol<br>"+
                "Cumulative time:  %{customdata[1]:.3e} s<br>"+
                "Frequency:  %{marker.size:d} <br>",
            name="Frequency landscape",
            visible='legendonly',
        )
    )
    
    
    # plot strand-level background
    # color_mapping = {
    #     "Incumbent Invader+Substrate": "red",
    #     "Incumbent Substrate+Invader": "red",
        
    #     "Incumbent+Substrate Invader": "blue",
    #     "Invader Incumbent+Substrate": "blue",
    #     "Invader Substrate+Incumbent": "blue",
    #     "Substrate+Incumbent Invader": "blue",
            
    #     "Incumbent+Substrate+Invader": "grey",
    #     "Invader+Substrate+Incumbent": "grey",
    #     "Invader+Incumbent+Substrate": "grey",
    #     "Substrate+Incumbent+Invader": "grey",
    # }
    
    color_mapping = {
        "inv+sub+incb": "grey",
        "incb+sub+inv": "red",
        "incb+sub inv": "blue",    
        "incb sub+inv": "orange",
    }
    
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)],
            mode='markers',
            marker=dict(
                sizemode='diameter',
                size=5,
                color=[color_mapping[type_val] for type_val in df["ShortName"]],
                showscale=False,
            ),
            text=df['DP'],
            customdata=np.stack((df["CumT"],
                                 df["Energy"],
                                 df["Freq"],
                                 df['ShortName'],
                                     ),axis=-1),
            hovertemplate=
                "<b>%{customdata[3]}</b><br>" +
                "<b>%{text}</b><br><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[1]:.3f} kcal/mol<br>"+
                "Cumulative time:  %{customdata[0]:.3e} s<br>"+
                "Frequency:  %{customdata[2]:d} <br>",
            name="Strand-level",
            visible='legendonly',
        )
    )

    
    # plot bounded/unbounded background
    fig.add_trace(go.Scattergl(
            x=df["{} 1".format(vis)], 
            y=df["{} 2".format(vis)],
            mode='markers',
            marker=dict(
                sizemode='diameter',
                size=5,
                color=df["IncbInvPair"],
                showscale=False,
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
            customdata=np.stack((df["CumT"],
                                 df["Energy"],
                                 df["Freq"],
                                 df['ShortName'],
                                     ),axis=-1),
            hovertemplate=
                "<b>%{customdata[3]}</b><br>" +
                "<b>%{text}</b><br>" +
                "X: %{x}   " + "   Y: %{y} <br>"+
                "Energy:  %{customdata[1]:.3f} kcal/mol<br>"+
                "Cumulative time:  %{customdata[0]:.3e} s<br>"+
                "Frequency:  %{customdata[2]:d} <br>",
            name="IncbInvPair",
            visible='legendonly',
        )
    )

    # layout trajectory on top of energy landscape
    for i in range(0, len(dfall), 10):
    # for i in [0,2,4,221,222,223]:
    
        Step = []
        if len(dfall["DP"][i]) < 2000:
            Step = np.arange(len(dfall["DP"][i]))
        else:
            Step = np.full(len(dfall["DP"][i]), None, dtype=object)
        
        fig.add_trace(
            go.Scattergl(
                x=dfall[f"{vis}"][i][:,0],
                y=dfall[f"{vis}"][i][:,1],
                mode='lines+markers',
                line=dict(
                    color='rgba(0,0,0,0.6)',
                    width=1,
                ),
                marker=dict(
                    sizemode='diameter',
                    size=4.5,
                    # color=dfall["Energy"][i],
                    # colorscale="Plasma",
                    color=[color_mapping[type_val] for type_val in dfall["ShortName"][i]],
                    # colorbar=dict(
                    #     x=-0.2,
                    #     y=0.5,
                    #     tickvals=[],
                    #     len=1,
                    # ),
                ),
                text=Step,
                customdata=np.stack((dfall['Pair'][i],
                                     dfall['TransT'][i],
                                     dfall['DP'][i],
                                     dfall['HT'][i],
                                     dfall['ShortName'][i],
                                     ),axis=-1),
                hovertemplate=
                    # "Step:  <b>%{text}</b><br><br>"+
                    "%{customdata[4]}<br>" +
                    "<b>%{customdata[2]}</b><br>" +
                    "X: %{x}   " + "   Y: %{y} <br>"+
                    "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                    "Expected holding time:  %{customdata[3]:.3e} s<br>",
                    # "Total transition time until current state:  %{customdata[1]:.3e} s<br>",
                visible='legendonly',
                name = "Trace {}".format(dfall["IDX"][i]),
            )
        )
        
    # plot the last 5 traces
    for i in range(len(dfall)-5, len(dfall)):
        fig.add_trace(
            go.Scattergl(
                x=dfall[f"{vis}"][i][:,0],
                y=dfall[f"{vis}"][i][:,1],
                mode='lines+markers',
                line=dict(
                    color='rgba(0,0,0,0.6)',
                    width=1,
                ),
                marker=dict(
                    sizemode='diameter',
                    size=4.5,
                    # color=dfall["Energy"][i],
                    # colorscale="Plasma",
                    color=[color_mapping[type_val] for type_val in dfall["ShortName"][i]],                    
                ),
                text=Step,
                customdata=np.stack((dfall['Pair'][i],
                                     dfall['TransT'][i],
                                     dfall['DP'][i],
                                     dfall['HT'][i],
                                     dfall['ShortName'][i],
                                     ),axis=-1),
                hovertemplate=
                    "%{customdata[4]}<br>" +
                    "<b>%{customdata[2]}</b><br>" +
                    "X: %{x}   " + "   Y: %{y} <br>"+
                    "Energy:  %{marker.color:.3f} kcal/mol<br>"+
                    "Expected holding time:  %{customdata[3]:.3e} s<br>",
                visible='legendonly',
                name = "Trace {}".format(dfall["IDX"][i]),
            )
        )
    
    

    # label initial and final states
    fig.add_trace(
        go.Scattergl(
            x=[dfall[f"{vis}"][0][0,0],dfall[f"{vis}"][0][-1,0]],
            y=[dfall[f"{vis}"][0][0,1],dfall[f"{vis}"][0][-1,1]],
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