import numpy as np
import time
import argparse
from plot_funcs import sort_gao, sort_hata, sort_machinek, plot_gao, plot_hata, plot_machineck, plot_machineck_png, map_id_to_shortname

if __name__ == '__main__': 
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--predata', required=True, help='preprocessed data file')
    parser.add_argument('--timedata', required=True, help='time data file')
    parser.add_argument('--embeddata', required=True, help='embedded data file')
    parser.add_argument('--outpath', required=True, help='output plot in html format')
    
    args = parser.parse_args()
    
    predata = args.predata
    timedata = args.timedata
    embeddata = args.embeddata
    outpath = args.outpath
    
    
    # Load the data
    print(f"[Plot] Loading preprocessed data from {predata}")
    
    loaded_data = np.load(predata, allow_pickle=True)
    
    energy_uniq = loaded_data["energy_uniq"]
    dp_og_uniq = loaded_data["dp_og_uniq"]
    trans_time = loaded_data["trans_time"]
    indices_uniq = loaded_data["indices_uniq"]
    id_uniq = loaded_data["id_uniq"]
    indices_all = loaded_data["indices_all"]
    energy  = energy_uniq[indices_all]
    dp_og = dp_og_uniq[indices_all]
    
    if "Gao" in predata:
        pair_uniq = loaded_data["pair_uniq"]
        pair = pair_uniq[indices_all]
        
    if "Hata" in predata:
        pair_uniq = loaded_data["pair_uniq"]
        pair = pair_uniq[indices_all]
        type_uniq = loaded_data["type_uniq"]
        type = type_uniq[indices_all]
        
    
    # if "Machinek" in predata:
        # seq_uniq = loaded_data["seq_uniq"]
        # shortname_uniq = loaded_data["shortname_uniq"]
        # incbinvpair_uniq = loaded_data["incbinvpair_uniq"]
        # seq = seq_uniq[indices_all]
        # shortname = shortname_uniq[indices_all]
        # incbinvpair = incbinvpair_uniq[indices_all]
   
    
    print(f"[Plot] Loading time data from {timedata}")
    
    loaded_data = np.load(timedata)
    
    hold_time_uniq = loaded_data["hold_time_uniq"]
    cum_time_uniq = loaded_data["cum_time_uniq"]
    freq_uniq = loaded_data["freq_uniq"]
    trj_id = loaded_data["trj_id"]
    
    hold_time = hold_time_uniq[indices_all]
    cum_time = cum_time_uniq[indices_all]
    freq = freq_uniq[indices_all]
    order_ids = id_uniq[indices_all]
    
    print(f"[Plot] Loading embedded data from {embeddata}")
    
    loaded_data = np.load(embeddata)
    
    pca_coords_uniq = loaded_data["pca_coords_uniq"]
    phate_coords_uniq = loaded_data["phate_coords_uniq"]
    
    pca_coords = pca_coords_uniq[indices_all]
    phate_coords = phate_coords_uniq[indices_all]
    
    print(f"[Plot] Converting order ids to names")
    id_uniq_name = np.array([map_id_to_shortname(i) for i in id_uniq])
    
    plt_args = (trj_id, dp_og, trans_time, hold_time, energy, cum_time, freq, 
                pca_coords, phate_coords, order_ids,
                dp_og_uniq, hold_time_uniq, energy_uniq, cum_time_uniq, freq_uniq,
                pca_coords_uniq, phate_coords_uniq, id_uniq_name,
                )
    
    if "Gao" in predata:
        plt_args = (*plt_args, 
                    pair_uniq, pair)
        
    if "Hata" in predata:
        plt_args = (*plt_args, 
                    pair_uniq, pair,
                    type_uniq, type)

    # Sort trajectories by their hold time
    print(f"[Plot] Sorting trajectories by their reaction time")
    
    if "Hata" in predata:
        df, dfsucc, dffail = sort_hata(plt_args)
        
    elif "Gao" in predata:
        df, dfall = sort_gao(plt_args)
    
    # TODO
    else:
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
    else:
        # for vis in ["PCA","PHATE"]:
        for vis in ["PHATE"]:
            plot_machineck_png(df,dfall,vis=vis, output_dir=outpath)
            
            fig = plot_machineck(df,dfall,vis=vis)
            savename = outpath+"_"+vis+".html"
            fig.write_html(savename)
            print(f"[Plot] Plot saved to {savename}")
        
    print(f"[Plot] Done!")
    
    # Record the end time
    end_time = time.time()
    print(f"[Plot] Elapsed Time: {(end_time - start_time):.3f} seconds")
    
    
    
    
    
    
    
    
