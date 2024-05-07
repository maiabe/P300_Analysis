import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.stats import permutation_cluster_test


channel_names = [ 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                'P6', 'P8', 'PO7', 'PO8', 'O1', 'O2', 'EOG']
channels_to_plot = ['Fp1', 'F1', 'Fz', 'FC1', 'FCz', 'C1', 'Cz', 'CP1', 'CPz', 'P3', 'P1', 'Pz', 'EOG']
referenced_scalings = { 'eeg': 100e-6 }
filtered_scalings = { 'eeg': 50e-6 }
cleaned_scalings = {'eeg': 50e-6}
epochs_scalings = { 'eeg': 50e-6 }


def plot_data(type, subject=None, session=None):

    ########## Plot Original
    if type == "original":
        original_sample = f"data/originalRaw/Data_S{subject}_Sess{session}_raw.fif"

        if os.path.exists(original_sample):
            # Plot original Data
            print("----------------- Plot Original Data -----------------")
            original_raw = mne.io.read_raw_fif(original_sample, preload=True)
            original_fig = original_raw.plot(scalings={ 'eeg': 200e-6 }, show=False)
            original_fig.set_size_inches(15, 5)
            print(original_raw.info)
        else:
            print(f"Missing original raw file for Subject {subject} Session {session}")


    ########## Plot Original vs Filtered EEG Data
    if type == "original_vs_filtered":
        original_sample = f"data/originalRaw/Data_S{subject}_Sess{session}_raw.fif"
        filtered_sample = f"data/preprocessed/bandpassFiltered/Data_S{subject}_Sess{session}_filtered_raw.fif"

        # if os.path.exists(original_sample):
        #     # Plot original Data
        #     print("----------------- Plot Original Data -----------------")
        #     original_raw = mne.io.read_raw_fif(original_sample, preload=True)
        #     original_fig = original_raw.plot(scalings=preprocess_scalings, show=False)
        #     original_fig.set_size_inches(15, 5)
        #     print(original_raw.info)

        # else:
        #     print(f"Missing original file for Subject {subject} Session {session}")

        if os.path.exists(filtered_sample):
            # Plot filtered Data
            print("----------------- Plot Filtered Data -----------------")
            filtered_raw = mne.io.read_raw_fif(filtered_sample, preload=True)
            filtered_fig = filtered_raw.plot(scalings={ 'eeg': 50e-6 }, show=False)
            filtered_fig.set_size_inches(15, 5)
            print(filtered_raw.info)
        else:
            print(f"Missing filtered file for Subject {subject} Session {session}")

    ########## Plot Filtered vs Avg Referenced EEG Data
    if type == "rereferenced":
        referenced_sample = f"data/preprocessed/rereferenced/Data_S{subject}_Sess{session}_referenced_raw.fif"

        if os.path.exists(referenced_sample):
            # Plot referenced Data
            print("----------------- Plot Referenced Data -----------------")
            referenced_raw = mne.io.read_raw_fif(referenced_sample, preload=True)
            referenced_fig = referenced_raw.plot(scalings=referenced_scalings, show=False, proj=True)
            referenced_fig.set_size_inches(15, 5)
            print(referenced_raw.info)
        else:
            print(f"Missing referenced file for Subject {subject} Session {session}")

    ########## Plot Filtered vs Cleaned EEG Data
    if type == "filtered_vs_cleaned":
        filtered_sample = f"data/preprocessed/bandpassFiltered/Data_S{subject}_Sess{session}_filtered_raw.fif"
        cleaned_sample = f"data/preprocessed/artifactRemoved/Data_S{subject}_Sess{session}_cleaned_raw.fif"

        if os.path.exists(filtered_sample):
            # Plot filtered Data
            print("----------------- Plot Filtered Data -----------------")
            filtered_raw = mne.io.read_raw_fif(filtered_sample, preload=True)
            filtered_fig = filtered_raw.plot(scalings=filtered_scalings, show=False)
            filtered_fig.set_size_inches(15, 5)
            print(filtered_raw.info)
        else:
            print(f"Missing filtered file for Subject {subject} Session {session}")

        if os.path.exists(cleaned_sample):
            # Plot cleaned Data
            print("----------------- Plot Cleaned Data -----------------")
            cleaned_raw = mne.io.read_raw_fif(cleaned_sample, preload=True)
            cleaned_fig = cleaned_raw.plot(scalings=cleaned_scalings, show=False)
            cleaned_fig.set_size_inches(15, 5)
            print(cleaned_raw.info)
        else:
            print(f"Missing cleaned file for Subject {subject} Session {session}")
    
    
    ########## Plot Filtered vs Cleaned EEG Data
    if type == "cleaned_vs_interpolated":
        cleaned_sample = f"data/preprocessed/artifactRemoved/Data_S{subject}_Sess{session}_cleaned_raw.fif"
        interpolated_sample = f"data/preprocessed/interpolated/Data_S{subject}_Sess{session}_interpolated_raw.fif"

        if os.path.exists(cleaned_sample):
            # Plot cleaned Data
            print("----------------- Plot Cleaned Data -----------------")
            cleaned_raw = mne.io.read_raw_fif(cleaned_sample, preload=True)
            cleaned_fig = cleaned_raw.plot(scalings=cleaned_scalings, show=False)
            cleaned_fig.set_size_inches(15, 5)
        else:
            print(f"Missing cleaned file for Subject {subject} Session {session}")

        if os.path.exists(interpolated_sample):
            # Plot interpolated Data
            print("----------------- Plot Interpolated Data -----------------")
            interpolated_raw = mne.io.read_raw_fif(interpolated_sample, preload=True)
            interpolated_fig = interpolated_raw.plot(scalings='auto', show=False)
            interpolated_fig.set_size_inches(15, 5)
        else:
            print(f"Missing interpolated file for Subject {subject} Session {session}")

    ########## Plots a sample epochs object (of a specified subject & session) as butterfly map with topomap at peak points
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
            epochs_fig = epochs.plot(n_epochs=5, title=f'S{subject} Sess{session} Epochs', events=True, event_color=event_colors, show=False, event_id=event_id)
            epochs_fig.set_size_inches(15, 5)

        else:
            print(f"Missing epoched file for Subject {subject} Session {session}")

    ########## Plots a sample evoked object (of a specified subject & session) as butterfly map with topomap at peak points
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

    ########## Plots the Grand Average of Correct vs Incorrect response waves and Difference waves
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
        # correct_evoked.plot(show=False, titles="Correct Response ERP", gfp=True)
        # incorrect_evoked.plot(show=False, titles="Incorrect Response ERP", gfp=True)
        # difference_evoked.plot(show=False, titles="Difference Waves", gfp=True)
        
        # correct_evoked.plot_joint(title="Correct Response ERP", show=False)
        # incorrect_evoked.plot_joint(title="Incorrect Response ERP", show=False)
        # difference_evoked.plot_joint(title="Difference Waves", show=False)

        mne.viz.plot_compare_evokeds(dict(correct=correct_evoked, incorrect=incorrect_evoked), 
                legend="upper left",
                show_sensors="upper right",)

    ########## Get peak channel, time and amplitude and plot topomap at peaks for Correct and Incorrect responses
    #============================== Should plot difference peaks? 
    if type == "peaks":
        grand_average_path = "data/grandAverage/grand_averages-ave.fif"

        # Load specific evoked object by comment
        correct_evoked = mne.read_evokeds(grand_average_path, condition='correct')
        incorrect_evoked = mne.read_evokeds(grand_average_path, condition='incorrect')
        difference_evoked = mne.combine_evoked([correct_evoked, incorrect_evoked], [1, -1])

        # Find Peaks
        time_window = (0.25, 0.59)  # 250 to 600 milliseconds
        correct_peak_result = correct_evoked.get_peak(ch_type='eeg', tmin=time_window[0], tmax=time_window[1], mode='pos', return_amplitude=True)
        print("-------------- Peak for Correct Feedback ---------------")
        correct_peak_channel = correct_peak_result[0]
        correct_peak_time = correct_peak_result[1]  # Convert to milliseconds
        correct_peak_amplitude = correct_peak_result[2]  # Convert to microvolts
        print(f"Correct Peak Channel at {correct_peak_channel}")
        print(f"Correct Peak at {correct_peak_time*1e3} ms with amplitude {correct_peak_amplitude*1e6} µV")
        
        incorrect_peak_result = incorrect_evoked.get_peak(ch_type='eeg', tmin=time_window[0], tmax=time_window[1], mode='pos', return_amplitude=True)
        print("-------------- Peak for Incorrect Feedback ---------------")
        incorrect_peak_channel = incorrect_peak_result[0]
        incorrect_peak_time = incorrect_peak_result[1]
        incorrect_peak_amplitude = incorrect_peak_result[2]
        print(f"Inorrect Peak Channel at {incorrect_peak_channel}")
        print(f"Inorrect Peak at {incorrect_peak_time*1e3} ms with amplitude {incorrect_peak_amplitude*1e6} µV")
        
        diff_peak_result = difference_evoked.get_peak(ch_type='eeg', tmin=time_window[0], tmax=time_window[1], mode='pos', return_amplitude=True)
        print("-------------- Peak for Difference Waves ---------------")
        diff_peak_channel = diff_peak_result[0]
        diff_peak_time = diff_peak_result[1]
        diff_peak_amplitude = diff_peak_result[2]
        print(f"Difference wave Peak Channel at {diff_peak_channel}")
        print(f"Difference wave Peak at {diff_peak_time*1e3} ms with amplitude {diff_peak_amplitude*1e6} µV")

        # print(f"Incorrect P300 Peak at {incorrect_peak_time*1000} ms with amplitude {incorrect_peak_amplitude} µV")

        # Plot Topomaps
        correct_topo_fig = correct_evoked.plot_topomap(times=correct_peak_time, ch_type='eeg', show_names=True, show=False)
        correct_topo_fig.suptitle("Peak Amplitude Topography Map for Correct Responses")
        incorrect_topo_fig = incorrect_evoked.plot_topomap(times=incorrect_peak_time, ch_type='eeg', show_names=True, show=False)
        incorrect_topo_fig.suptitle("Peak Amplitude Topography Map for Incorrect Responses")
        diff_topo_fig = difference_evoked.plot_topomap(times=diff_peak_time, ch_type='eeg', show_names=True, show=False)
        diff_topo_fig.suptitle("Peak Amplitude Topography Map for Correct vs Incorrect Responses")

        # Plot Animated Topomap
        times = np.arange(0.25, 0.6, 0.01)
        # fig, anim = difference_evoked.animate_topomap(ch_type="eeg", times=times, frame_rate=2, butterfly=True, blit=False, time_unit="ms")
        # Save Animation
        # anim.save('diff_peak_animation.mp4', writer='ffmpeg', fps=10)

    ########## Conduct Permutation Cluster Test for Correct and Incorrect Responses
    if type == "cluster_permutation":

        #-- Get data for each condition over each epoch (each observation)
        epochs_dir = "data/epochs"
        
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

        # Create a mask for significant clusters at p < 0.05
        p_threshold = 0.05
        significant_points = np.zeros_like(T_obs, dtype=bool)
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= p_threshold:
                significant_points[c] = True

        # Now plot the T-values with masking for non-significant points
        cluster_fig = t_evoked.plot_image(picks="all", exclude=['EOG'], titles='Significant T-values Between Correct vs Incorrect Responses', mask=significant_points, mask_style='green', time_unit='s', show_names=True)

        # Plot Topomaps
        time_window = (0.25, 0.59)  # 250 to 600 milliseconds
        tevoked_peak_result = t_evoked.get_peak(ch_type='eeg', tmin=time_window[0], tmax=time_window[1], mode='pos', return_amplitude=True)
        print("-------------- Peak for t-test evoked ---------------")
        tevoked_peak_time = tevoked_peak_result[1]  # Convert to milliseconds
        
        t_topo_fig = t_evoked.plot_topomap(times=tevoked_peak_time, ch_type='eeg', show_names=True, show=False)
        t_topo_fig.suptitle("T-values ")

        # Create Topomap animation showing differences in statistical significance
        times = np.arange(t_evoked.times[0], t_evoked.times[-1], t_evoked.times[1] - t_evoked.times[0])
        # fig, anim = t_evoked.animate_topomap(ch_type="eeg", times=times, frame_rate=2, butterfly=True, blit=False, time_unit="ms")
        # Save Animation
        # anim.save('tval_topo_animation.mp4', writer='ffmpeg', fps=10)

    # Use plt.show() to dislay all plots at once
    plt.show()

    print(f"plot {type} data")

