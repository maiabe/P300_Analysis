import pandas as pd
import numpy as np
import os
import mne


def filter_data():
        
    # Directory to original raw data
    original_raw_dir = 'data/originalRaw'

    # Set of channels
    channel_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                    'P6', 'P8', 'PO7', 'POz', 'O1', 'O2', 'EOG']
    channel_types = ['eog' if name == 'EOG' else 'eeg' for name in channel_names]

    channels_to_plot = ['Fp1', 'F1', 'Fz', 'FC1', 'FCz', 'C1', 'Cz', 'CP1', 'CPz', 'P3', 'P1', 'Pz', 'EOG']
    preprocess_plot_scalings = { 'eeg': 200e-6 }

    ##################################################################
    #                  Filter Data by Bandpass Filters      
    ##################################################################
    # Iterate Over Session Files to load each EEG session data (Make sure all sessions exist)
    session_files = ["{}/Data_S{:02d}_Sess{:02d}.csv".format(original_raw_dir, i, j)
                    for i in range(1, 26+1)
                    for j in range(1, 5+1)]

    # Filter settings
    sampling_rate = 200  #-- original data downsampled already to 200Hz
    low_freq, high_freq = 1, 40  #-- Frequency band for P300
    montage = mne.channels.make_standard_montage('standard_1020')  # For electrode locations

    # Define output directory and file to save filtered data
    output_dir = 'data/preprocessed/bandpassFiltered'

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define Scalings
    plot_scalings = { 'eeg': 1e-6 }

    first_iter = True
    for file_path in session_files:
        base_name = os.path.basename(file_path)   # Get the file name from full path
        filtered_base_name = base_name.replace('_raw.fif', '_filtered_raw.fif')   # Replace the extention
        output_file_path = os.path.join(output_dir, filtered_base_name)   # Construct the full output file path to save the filtered data to

        print(output_file_path)

        try:
            # Create a RawArray object
            raw = mne.io.read_raw_fif(file_path)

            # Set the montage (electrode locations)
            raw.set_montage(montage)

            print("-------- Original info --------")
            print(raw.info)

            # -------------------- Bandpass Filter --------------------
            # Filter the data
            raw_filtered = raw.copy().filter(low_freq, high_freq, method='fir', fir_design='firwin', verbose=True)

            # Examine the shape of the data
            print("-------- Bandpass filtered info --------")
            print(raw_filtered.get_data().shape)
            print(raw_filtered.info)

            # Save the preprocessed data to /Preprocessing/BandpassFiltered directory
            raw_filtered.save(output_file_path, fmt='double', overwrite=True)
            print(f"=========================> Saved filtered data to {output_file_path}")


            ################################## Plot Original & Filtered Data ##################################
            # if first_iter:
            #     # Plot Original Data
            #     original_fig = raw.plot(duration=10.0, start=350, n_channels=13, scalings=preprocess_plot_scalings, show=False, picks=channels_to_plot)
            #     original_fig.set_size_inches(15, 5)
            #     original_fig.show()

            #     # Plot Filtered Data
            #     filtered_fig = raw_filtered.plot(duration=10.0, start=350, n_channels=13, scalings=preprocess_plot_scalings, show=False, picks=channels_to_plot)
            #     filtered_fig.set_size_inches(15, 5)
            #     filtered_fig.show()

            #     print('----------------------------- filtered data -----------------------------')
            #     print(raw_filtered.get_data())

            #     # raw_filtered.plot(scalings='auto', show = True, title='Filtered EEG - Subject 01, Session 01')
            #     first_iter = False

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Reset the first iteration flag
    # first_iter = True






def remove_artifact():
    filtered_dir = "data/preprocessed/bandpassFiltered"
    cleaned_dir = "data/preprocessed/artifactRemoved"

    # Ensure cleaned directory exists
    if not os.path.exists(cleaned_dir):
        os.makedirs(cleaned_dir)
    

    ##################################################################
    #                  Remove EOG from EEG using ICA      
    ##################################################################    
    # Iterate Over BandpassFiltered Files to load Filtered EEG data (Iterate through each one to confirm the existence of each session file)
    filtered_file_paths = ["{}/Data_S{:02d}_Sess{:02d}_raw.fif".format(filtered_dir, i, j)
                    for i in range(1, 26+1)
                    for j in range(1, 5+1)]

    
    for filtered_file_path in filtered_file_paths:
        print('==============================================================================================')
        print('==============================================================================================')
        print(filtered_file_path)
        filtered_file_name = os.path.basename(filtered_file_path)
        print(filtered_file_name)

        # Load the preprocessed .fif file
        filtered_raw = mne.io.read_raw_fif(filtered_file_path, preload=True)

        # Initialize and fit ICA
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, method='fastica')
        ica.fit(filtered_raw)

        # Plot the components to visualize their time courses and topographies
        # print('--------------------- Plot Components ---------------------')
        # ica.plot_components()

        # Automatically find the EOG artifacts
        eog_indices, eog_scores = ica.find_bads_eog(filtered_raw)
        # print('--------------------- Plot Scores ---------------------')
        # ica.plot_scores(eog_scores)

        # Check if there are any EOG-related components
        # if eog_indices:
        #     # Plot the properties of EOG components to further inspect them
        #     print('--------------------- Plot Properties ---------------------')
        #     ica.plot_properties(filtered_raw, picks=eog_indices)
        # else:
        #     print('No EOG-related artifact componetns found.')

        # Exclude the identified components
        print('--------------------- EOG Indices ---------------------')
        print(eog_indices)
        ica.exclude = eog_indices

        # Apply ICA to remove EOG components
        corrected_raw = filtered_raw.copy()
        ica.apply(corrected_raw)

        # Save ICA cleaned Raw data
        cleaned_file_name = os.path.basename(filtered_file_name).replace('_raw.fif', '_cleaned_raw.fif')
        print(cleaned_file_name)
        cleaned_file_path = os.path.join(cleaned_dir, cleaned_file_name)
        print(cleaned_file_path)

        corrected_raw.save(cleaned_file_path, fmt='double', overwrite=True)

        print(f"=====================================> Processed and saved cleaned data to {cleaned_file_path}")



    print("remove artifact")