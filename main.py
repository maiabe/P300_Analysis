from src.plot_data import plot_data
from src.save_original_raw import save_original_raw
from src.preprocessing import filter_data
from src.preprocessing import remove_artifact
from src.get_erp import create_epochs
from src.get_erp import average_epochs
from src.get_erp import create_grand_average
from src.stats_test import compute_cluster_permutation

###########################################################################################
# Note: Please make sure you have the original csv's under data/allData 
#       Once you make sure you have the original data, please run each step sequentially.
###########################################################################################
def main():
    ##### -------------- Load Data --------------

    #-- Save original csv data as raw files
    # save_original_raw()
    
    ##### -------------- Data Preprocessing --------------
    #-- Filter Data by bandpass filters and save data as filtered raw files
    # filter_data()

    #-- Plot Original vs Filtered Data
    # subject = "01"
    # session = "03"
    # plot_data("original_vs_filtered", subject, session)
    
    #-- Remove Artifacts using ICA and save data as cleaned raw files
    # remove_artifact()
    
    #-- Plot Filtered vs Cleaned Data
    # subject = "01"
    # session = "03"
    # plot_data("filtered_vs_cleaned", subject, session)
    

    ##### -------------- Epoch Cleaned Data --------------
    sfreq=200
    # create_epochs(sfreq=sfreq)

    #-- Plot Epochs
    # subject = "01"
    # session = "03"
    # plot_data("epochs", subject, session)


    ##### -------------- Average Epochs --------------
    # average_epochs()

    #-- Plot Averaged Epochs
    # subject = "01"
    # session = "03"
    # plot_data("evoked", subject, session)
    
    
    ##### -------------- Compute Grand Average --------------
    # create_grand_average()  #-- Also saves the evoked objects for each subject in raw file 
    
    #-- Plot Grand Average as ERP waveform
    # plot_data("grand_average")

    ##### -------------- Compute Stats --------------

    #-- Get Peaks and plot topomaps of amplitude at peaks
    # plot_data("peaks")

    #-- Conduct Permutation Cluster Test and Plot Cluster Map and Create Topomap movie of significant timepoints
    plot_data("cluster_permutation")

    #-- Create Topomap movie at statistically important times


    print("main function called")

 





if __name__ == "__main__":
    main()

