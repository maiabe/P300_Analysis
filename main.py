from src.plot_data import plot_data
from src.save_original_raw import save_original_raw
from src.preprocessing import filter_data
from src.preprocessing import rereference
from src.preprocessing import remove_artifact
from src.get_erp import create_epochs
from src.get_erp import average_epochs
from src.get_erp import create_grand_average
from src.stats_test import compute_cluster_permutation

from src.preprocessing import interpolate_bads

###########################################################################################
# Note: Please make sure you have the original csv data under data/allData 
#       Once you make sure you have the original data, please run each step sequentially.
###########################################################################################
def main():
    ##### -------------- Load Data --------------
    #-- Save original csv data as raw files
    # save_original_raw()
    
    
    # for i in range(1, 26+1):   # Num Subjects
    #     for j in range(1, 5+1):    # Num Sessions
    #         plot_data('original', f"{i:02}", f"{j:02}")

    # plot_data('original', "16", "02")

    #-- Interpolate bad channel attempt not sure if it is working
    # interpolate_bads()  // not in use

    ##### -------------- Data Preprocessing --------------
    
    #-- 1. rereferencing --- no rereference electrode
    # rereference("data/originalRaw", "raw")
    # plot_data("rereferenced", "16", "02")
    
    #-- 2. Filter Data by bandpass filters and save data as filtered raw files
    # filter_data(low_freq=1, high_freq=40)

    #-- Plot Original vs Filtered Data
    subject = "16"
    session = "02"
    # plot_data("original_vs_filtered", subject, session)

    #-- 3. Remove Arti1facts using ICA and save data as cleaned raw files
    # remove_artifact(n_components=20, random_state=97, max_iter=800)

    #-- Plot Filtered vs Cleaned Data
    subject = "16"
    session = "02"
    # plot_data("filtered_vs_cleaned", subject, session)
    

    ##### -------------- Epoch Cleaned Data --------------
    # preprocessed_dir = 'data/preprocessed/interpolated'
    # # identifier_fname = 'interpolated_raw'
    
    preprocessed_dir = 'data/preprocessed/artifactRemoved'
    identifier_fname = 'cleaned_raw'
    
    # preprocessed_dir = 'data/preprocessed/rereferenced'
    # rr_identifier_fname = 'referenced_raw'
    
    # preprocessed_dir = 'data/preprocessed/bandpassFiltered'
    # identifier_fname = 'filtered_raw'
    
    # create_epochs(preprocessed_dir, identifier_fname, sfreq=200)

    #-- Plot Epochs
    subject = "16"
    session = "02"
    # plot_data("epochs", subject, session)

    
    ##### -------------- Compute Grand Average --------------
    #-- Averages the epochs in each session by event types, then averages all evoked objects to get the grand average
    # create_grand_average()  #-- Also saves the evoked objects for each subject in raw file 
    
    #-- Plot Grand Average as ERP waveform
    # plot_data("grand_average")

    ##### -------------- Compute Stats --------------

    #-- Get Peaks and plot topomaps of amplitude at peaks
    # plot_data("peaks")

    #-- Conduct Permutation Cluster Test and Plot Cluster Map and Create Topomap movie of significant timepoints
    # plot_data("cluster_permutation")

    #-- Create Topomap movie at statistically important times


    print("main function called")

 





if __name__ == "__main__":
    main()

