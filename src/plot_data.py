import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.stats import permutation_cluster_test

channels_to_plot = ['Fp1', 'F1', 'Fz', 'FC1', 'FCz', 'C1', 'Cz', 'CP1', 'CPz', 'P3', 'P1', 'Pz', 'EOG']
preprocess_scalings = { 'eeg': 200e-6 }
epochs_scalings = { 'eeg': 50e-6 }


def plot_data(type, subject=None, session=None):

    ########## Plot Original vs Filtered Data
    if type == "original_vs_filtered":
        original_sample = f"data/originalRaw/Data_S{subject}_Sess{session}_raw.fif"
        filtered_sample = f"data/preprocessed/bandpassFiltered/Data_S{subject}_Sess{session}_raw.fif"

        if os.path.exists(original_sample):
            # Plot original Data
            print("----------------- Plot Original Data -----------------")
            original_raw = mne.io.read_raw_fif(original_sample, preload=True)
            original_fig = original_raw.plot(n_channels=13, scalings=preprocess_scalings, show=False, picks=channels_to_plot)
            original_fig.set_size_inches(15, 5)

        else:
            print(f"Missing original file for Subject {subject} Session {session}")

        if os.path.exists(filtered_sample):
            # Plot filtered Data
            print("----------------- Plot Filtered Data -----------------")
            filtered_raw = mne.io.read_raw_fif(filtered_sample, preload=True)
            filtered_fig = filtered_raw.plot(n_channels=13, scalings=preprocess_scalings, show=False, picks=channels_to_plot)
            filtered_fig.set_size_inches(15, 5)
        else:
            print(f"Missing filtered file for Subject {subject} Session {session}")

    ########## Plot Filtered vs Cleaned Data
    if type == "filtered_vs_cleaned":
        filtered_sample = f"data/preprocessed/bandpassFiltered/Data_S{subject}_Sess{session}_raw.fif"
        cleaned_sample = f"data/preprocessed/artifactRemoved/Data_S{subject}_Sess{session}_cleaned_raw.fif"

        if os.path.exists(filtered_sample):
            # Plot filtered Data
            print("----------------- Plot Filtered Data -----------------")
            filtered_raw = mne.io.read_raw_fif(filtered_sample, preload=True)
            filtered_fig = filtered_raw.plot(n_channels=13, scalings=preprocess_scalings, show=False, picks=channels_to_plot)
            filtered_fig.set_size_inches(15, 5)
        else:
            print(f"Missing filtered file for Subject {subject} Session {session}")

        if os.path.exists(cleaned_sample):
            # Plot cleaned Data
            print("----------------- Plot Cleaned Data -----------------")
            cleaned_raw = mne.io.read_raw_fif(cleaned_sample, preload=True)
            cleaned_fig = cleaned_raw.plot(n_channels=13, scalings=preprocess_scalings, show=False, picks=channels_to_plot)
            cleaned_fig.set_size_inches(15, 5)
        else:
            print(f"Missing filtered file for Subject {subject} Session {session}")


    if type == "epochs":
        epochs_sample = f"data/epochs/Data_S{subject}_Sess{session}_epochs_epo.fif"

        # Define event_id dictionary
        event_id = {'incorrect': 1, 'correct': 2}

        # Define colors for correct and incorrect feedback types
        event_colors = { 'incorrect': 'red', 'correct': 'blue' }

        if os.path.exists(epochs_sample):
            # Plot Epochs
            print("----------------- Plot Epoched Data -----------------")
            epochs = mne.read_epochs(epochs_sample, preload=True)
            epochs_fig = epochs.plot(picks=channels_to_plot, scalings=epochs_scalings, n_epochs=5, n_channels=1,  title=f'S{subject} Sess{session} Epochs', events=True, event_color=event_colors, show=False, event_id=event_id)
            epochs_fig.set_size_inches(15, 5)

        else:
            print(f"Missing epoched file for Subject {subject} Session {session}")

    if type == "evoked":
        evoked_sample = f"data/evoked/Data_S{subject}_Sess{session}_evoked-ave.fif"

        if os.path.exists(evoked_sample):

            # Plot Correct Evoked Data
            print("----------------- Plot Correct Evoked Data -----------------")
            evokeds = mne.read_evokeds(evoked_sample)
            print(evokeds[0].info)
            evokeds[0].plot_joint(title="Correct Evoked", show=False)
            
            print("----------------- Plot Incorrect Evoked Data -----------------")
            print(evokeds[1].info)
            evokeds[1].plot_joint(title="Incorrect Evoked", show=False)
            
        else:
            print(f"Missing correct evoked file for Subject {subject} Session {session}")

    if type == "grand_average":
        grand_average_path = "data/grandAverage/grand_averages-ave.fif"

        # Load all evoked objects
        # all_evoked = mne.read_evokeds(grand_average_path)

        # Load specific evoked object by comment
        correct_evoked = mne.read_evokeds(grand_average_path, condition='correct')
        incorrect_evoked = mne.read_evokeds(grand_average_path, condition='incorrect')

        # Compute differences of correct vs incorrect
        difference_evoked = mne.combine_evoked([correct_evoked, incorrect_evoked], [1, -1])


        # Plot grand averages
        correct_evoked.plot(show=False, titles="Correct Response ERP")
        incorrect_evoked.plot(show=False, titles="Incorrect Response ERP")
        difference_evoked.plot(show=False, titles="Difference Waves")

    if type == "peaks":
        
        # Find Peaks
        # time_window = (0.25, 0.5)  # 250 to 500 milliseconds
        # ch_name = 'Pz'  # Commonly used channel for P300
        # correct_peak_time, correct_peak_amplitude = correct_evoked.get_peak(ch_type='eeg', tmin=time_window[0], tmax=time_window[1], mode='pos', return_amplitude=True)
        # print(f"Correct P300 Peak at {correct_peak_time*1000} ms with amplitude {correct_peak_amplitude} µV")
        # incorrect_peak_time, incorrect_peak_amplitude = incorrect_evoked.get_peak(ch_type='eeg', tmin=time_window[0], tmax=time_window[1], mode='pos', return_amplitude=True)
        # print(f"Incorrect P300 Peak at {incorrect_peak_time*1000} ms with amplitude {incorrect_peak_amplitude} µV")

        # # Plot Topomaps
        # correct_evoked.plot_topomap(times=correct_peak_time, ch_type='eeg', show=False)
        # incorrect_evoked.plot_topomap(times=incorrect_peak_time, ch_type='eeg', show=False)

        # Create data arrays: correct_evoked_data and incorrect_evoked_data are numpy arrays from evoked.get_data() for correct and incorrect conditions
        # Extract data as numpy arrays from the Evoked objects
        # evoked_data_correct = correct_evoked.get_data()
        # evoked_data_incorrect = incorrect_evoked.get_data()

        # all_evoked_correct and all_evoked_incorrect are lists of Evoked objects
        # data_correct = [e.get_data() for e in correct_evoked]  # List of arrays
        # data_incorrect = [e.get_data() for e in incorrect_evoked]  # List of arrays

        # Stack arrays to create a 3D array (n_epochs, n_channels, n_times)
        # data_correct = np.stack(data_correct)
        # data_incorrect = np.stack(data_incorrect)

        # Conduct the permutation cluster test
        # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(np.array([data_correct, data_incorrect]), n_permutations=1000, threshold=None, tail=0)
        # Observed Statistic
        # print(T_obs)
        # Cluster Information
        # print(clusters)
        # p-values for each cluster
        # print(cluster_p_values)
        # Permutation distribution of the max statistic
        # print(H0)

    # Use plt.show() to dislay all plots at once
    plt.show()

    print(f"plot {type} data")

