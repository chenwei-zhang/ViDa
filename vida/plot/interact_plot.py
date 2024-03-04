import numpy as np
import time
import argparse
from plot_funcs import sort_gao, sort_hata, sort_machinek, plot_gao, plot_hata, plot_machineck

if __name__ == '__main__': 
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--predata', required=True, help='preprocessed data file')
    parser.add_argument('--timedata', required=True, help='time data file')
    parser.add_argument('--seqdata', required=True, help='sequence data file')
    parser.add_argument('--embeddata', required=True, help='embedded data file')
    parser.add_argument('--outpath', required=True, help='output plot in html format')
    
    args = parser.parse_args()
    
    predata = args.predata
    timedata = args.timedata
    embeddata = args.embeddata
    seqdata = args.seqdata
    outpath = args.outpath
    
    
    # Load the data
    print(f"[Plot] Loading preprocessed data from {predata}")
    
    loaded_data = np.load(predata)
    
    energy_uniq = loaded_data["energy_uniq"]
    pair_uniq = loaded_data["pair_uniq"]
    dp_og_uniq = loaded_data["dp_og_uniq"]
    trans_time = loaded_data["trans_time"]
    indices_uniq = loaded_data["indices_uniq"]
    indices_all = loaded_data["indices_all"]
    
    energy  = energy_uniq[indices_all]
    dp_og = dp_og_uniq[indices_all]
    pair = pair_uniq[indices_all]
    
    if "Hata" in predata:
        type_uniq = loaded_data["type_uniq"]
        type = type_uniq[indices_all]
        
    
    print(f"[Plot] Loading time data from {timedata}")
    
    loaded_data = np.load(timedata)
    
    hold_time_uniq = loaded_data["hold_time_uniq"]
    cum_time_uniq = loaded_data["cum_time_uniq"]
    freq_uniq = loaded_data["freq_uniq"]
    hold_time = hold_time_uniq[indices_all]
    trj_id = loaded_data["trj_id"]
    
    cum_time = cum_time_uniq[indices_all]
    freq = freq_uniq[indices_all]
    
    
    print(f"[Plot] Loading sequence data from {seqdata}")
    loaded_data = np.load(seqdata, allow_pickle=True)
    seqlabel_uniq = loaded_data["seqlabel_uniq"]
    
    
    print(f"[Plot] Loading embedded data from {embeddata}")
    
    loaded_data = np.load(embeddata)
    
    pca_coords_uniq = loaded_data["pca_coords_uniq"]
    phate_coords_uniq = loaded_data["phate_coords_uniq"]
    
    pca_coords = pca_coords_uniq[indices_all]
    phate_coords = phate_coords_uniq[indices_all]
    seqlabel = seqlabel_uniq[indices_all]
    
    plt_args = (trj_id, dp_og, trans_time, hold_time, energy, pair, cum_time, freq, 
                pca_coords, phate_coords, 
                dp_og_uniq, hold_time_uniq, energy_uniq, pair_uniq, cum_time_uniq, freq_uniq,
                pca_coords_uniq, phate_coords_uniq)
    
    if "Hata" in predata:
        plt_args = plt_args + (type_uniq, type)
        
    if "Machinek" in predata:
        # plt_args = plt_args + (seqlabel_uniq)
        plt_args = (*plt_args, seqlabel_uniq, seqlabel)
    
    
    # Sort trajectories by their hold time
    print(f"[Plot] Sorting trajectories by their hold time")
    
    if "Hata" in predata:
        df, dfsucc, dffail = sort_hata(plt_args)
        
    elif "Gao" in predata:
        df, dfall = sort_gao(plt_args)
    
    # TODO
    elif "Machinek" in predata:
        df, dfall = sort_machinek(plt_args)
    
    
    # Make the plot
    print(f"[Plot] Making plot")    
        
    if "Hata" in predata:
        fig = plot_hata(df,dfsucc,dffail,vis='PHATE')
        savename = outpath
        fig.write_html(savename)
        
    elif "Gao" in predata:
        fig = plot_gao(df,dfall,vis='PHATE')
        savename = outpath
        fig.write_html(savename)
    
    # TODO
    elif "Machinek" in predata:
        for vis in ["PCA","PHATE"]:
            fig = plot_machineck(df,dfall,vis=vis)
            savename = outpath+"_"+vis+".html"
            fig.write_html(savename)
        
    print(f"[Plot] Done!")
    
    # Record the end time
    end_time = time.time()
    print(f"[Plot] Elapsed Time: {(end_time - start_time):.3f} seconds")
    
    
    
    
    
    
    
    
