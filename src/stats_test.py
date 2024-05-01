import os
import numpy as np
import mne
from mne.stats import permutation_cluster_test


def compute_cluster_permutation():
    #-- Get data for each condition over each epoch (each observation)
    epochs_dir = "data/epochs"
    
    #-- Prep data for correct and incorrect responses based on epochs for all sessions
    data_correct = []
    data_incorrect = []
    for subject_id in range(1, 27):    # Loop through subjects
        for session_id in range(1, 6):    # Loop through sessions
            # Define path and read epochs data for this session
            file_path = f'{epochs_dir}/Data_S{subject_id:02d}_Sess{session_id:02d}_epochs_epo.fif'
            epochs = mne.read_epochs(file_path, preload=True)

            # Append data for each event type to the aggregate list
            data_correct.append(epochs['correct'].get_data(copy=True))
            data_incorrect.append(epochs['incorrect'].get_data(copy=True))

    # Convert lists of arrays into a single array for each condition
    data_correct = np.concatenate(data_correct, axis=0)  # Shape (n_correct_epochs, n_channels, n_timepoints)
    data_incorrect = np.concatenate(data_incorrect, axis=0)  # Shape (n_incorrect_epochs, n_channels, n_timepoints)

    print("-------------------- Data Correct --------------------")
    print(data_correct)
    print(data_correct.shape)
    print("-------------------- Data Incorrect --------------------")
    print(data_incorrect)
    print(data_incorrect.shape)

    # Conduct the permutation cluster test
    X = [data_correct, data_incorrect]
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(X, threshold=0.05, n_permutations=1000, tail=1)
    # Observed Statistic
    print("------------------ T_obs --------------------")
    print(T_obs)
    # Cluster Information
    print("------------------ Clusters --------------------")
    print(clusters)
    # p-values for each cluster
    print("------------------ cluster_p_values --------------------")
    print(cluster_p_values)
    # Permutation distribution of the max statistic
    print("------------------ H0 --------------------")
    print(H0)

    # Get info and start time from any existing Evoked object
    sample_epoch_path = f"{epochs_dir}/Data_S01_Sess01_epochs_epo.fif"
    sample_epoch = mne.read_epochs(sample_epoch_path, preload=True)
    info = sample_epoch[0].info
    tmin = sample_epoch[0].times[0]

    print("-------------------- Info --------------------")
    print(info)

    # Create an Evoked object containing the T-values from the permutation test 
    t_evoked = mne.EvokedArray(T_obs, info, tmin=tmin)

    # Save the evoked object based on T-values obtained from cluster permutation test
    perm_test_dir = "data/stats"

    # Ensure permTest directory exists
    if not os.path.exists(perm_test_dir):
        os.makedirs(perm_test_dir)

    # Save Evoked object containing the T-values form the permutation test
    perm_test_path = f"{perm_test_dir}/perm_test_evoked-ave.fif"
    t_evoked.save(perm_test_path, overwrite=True)

