import pandas as pd
import numpy as np
import os
import mne
import matplotlib.pyplot as plt


all_data_dir = 'data/allData'
original_raw_dir = 'data/originalRaw'
filtered_dir = 'data/preprocessed/bandpassFiltered'
cleaned_dir = 'data/preprocessed/artifactRemoved'
referenced_dir = 'data/preprocessed/rereferenced'
interpolated_dir = 'data/preprocessed/interpolated'


# bads = ['POz', 'T7', 'T8', 'F7', 'F8', 'FT7', 'FT8', 'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8']
bads = ['POz', 'Fp1', 'Fp2', 'AF7', 'AF8']


def interpolate_bads():
    
    # Define output directory to save the rereferenced data
    output_dir = interpolated_dir
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate Over Session Files to load each EEG session data (Make sure all sessions exist)
    session_files = ["{}/Data_S{:02d}_Sess{:02d}_raw.fif".format(original_raw_dir, i, j)
                    for i in range(1, 26+1)
                    for j in range(1, 5+1)]
    
    
    for file_path in session_files:
        base_name = os.path.basename(file_path)   # Get the file name from full path
        result_name = base_name.replace(f'_raw.fif', '_interpolated_raw.fif')   # Replace the extention
        output_file_path = os.path.join(output_dir, result_name)   # Construct the full output file path to save the filtered data to

        print(file_path)

        try:
            print("--------- Original Raw Data ----------")
            # Create a RawArray object
            raw = mne.io.read_raw_fif(file_path, preload=True)


            # Step 1: Calculate statistical measures
            variance_per_channel = np.var(raw.get_data(), axis=1)  # Variance for each channel
            mean_per_channel = np.mean(np.abs(raw.get_data()), axis=1)     # Mean amplitude for each channel

            # Step 2: Compute signal-to-noise ratio (SNR)
            noise_level_per_channel = np.sqrt(variance_per_channel)  # Estimate noise level as standard deviation
            snr_per_channel = mean_per_channel / noise_level_per_channel

            # Step 3: Set threshold for SNR to identify bad channels
            snr_threshold = 0.7  # Adjust as needed
            bad_channels = [raw.info['ch_names'][i] for i, snr in enumerate(snr_per_channel) if snr < snr_threshold and 'EOG' not in raw.info['ch_names'][i]]

            print(bad_channels)

            # Step 4: Visualize SNR distribution
            # plt.figure(figsize=(8, 6))
            # plt.hist(snr_per_channel, bins=20, color='skyblue', edgecolor='black')
            # plt.axvline(x=snr_threshold, color='red', linestyle='--', label='SNR Threshold')
            # plt.xlabel('SNR')
            # plt.ylabel('Frequency')
            # plt.title('Signal-to-Noise Ratio (SNR) Distribution')
            # plt.legend()
            # plt.show()

            # Interpolate bad channels
            raw.info['bads'] = bad_channels
            raw_interpolated = raw.copy().interpolate_bads(reset_bads=False)

            # Set the average reference
            # raw_avg_ref = raw.copy().set_eeg_reference('average', projection=True)
            # print(raw.info)

            # Save the preprocessed data to /Preprocessing/BandpassFiltered directory
            raw_interpolated.save(output_file_path, fmt='double', overwrite=True)
            print(f"=========================> Saved interpolated_data data to {output_file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print('get bads')

def filter_data(low_freq, high_freq):

    ##################################################################
    #                  Filter Data by Bandpass Filters      
    ##################################################################
    # Iterate Over Session Files to load each EEG session data (Make sure all sessions exist)
    session_files = ["{}/Data_S{:02d}_Sess{:02d}_referenced_raw.fif".format(referenced_dir, i, j)
                    for i in range(1, 26+1)
                    for j in range(1, 5+1)]

    # low_freq, high_freq = 0.1, 40  #-- Frequency band for P300
    montage = mne.channels.make_standard_montage('standard_1020')  # For electrode locations

    # Define output directory and file to save filtered data
    output_dir = filtered_dir

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_path in session_files:
        base_name = os.path.basename(file_path)   # Get the file name from full path
        filtered_base_name = base_name.replace('_referenced_raw.fif', '_filtered_raw.fif')   # Replace the extention
        output_file_path = os.path.join(output_dir, filtered_base_name)   # Construct the full output file path to save the filtered data to

        print(file_path)

        try:
            # Create a RawArray object
            raw = mne.io.read_raw_fif(file_path, preload=True)

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

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Reset the first iteration flag
    # first_iter = True


def rereference(original_dir, identifier):

    # Define output directory to save the rereferenced data
    output_dir = referenced_dir
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate Over Session Files to load each EEG session data (Make sure all sessions exist)
    session_files = ["{}/Data_S{:02d}_Sess{:02d}_{}.fif".format(original_dir, i, j, identifier)
                    for i in range(1, 26+1)
                    for j in range(1, 5+1)]
    
    
    for file_path in session_files:
        base_name = os.path.basename(file_path)   # Get the file name from full path
        averaged_base_name = base_name.replace(f'_{identifier}.fif', '_referenced_raw.fif')   # Replace the extention
        output_file_path = os.path.join(output_dir, averaged_base_name)   # Construct the full output file path to save the filtered data to

        print(file_path)

        try:
            print("--------- Original Raw Data ----------")
            # Create a RawArray object
            raw = mne.io.read_raw_fif(file_path, preload=True)

            # Remove bad channels
            raw.info['bads'] = bads
            raw.drop_channels(ch_names=bads)

            # Set the average reference
            raw_avg_ref = raw.copy().set_eeg_reference('average', projection=True)
            print(raw.info)

            # Save the preprocessed data to /Preprocessing/BandpassFiltered directory
            raw_avg_ref.save(output_file_path, fmt='double', overwrite=True)
            print(f"=========================> Saved average referenced data to {output_file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


    print('average reference')


def remove_artifact(n_components, random_state, max_iter="auto"):

    # Ensure cleaned directory exists
    if not os.path.exists(cleaned_dir):
        os.makedirs(cleaned_dir)
    

    ##################################################################
    #                  Remove EOG from EEG using ICA      
    ##################################################################    
    # Iterate Over BandpassFiltered Files to load Filtered EEG data (Iterate through each one to confirm the existence of each session file)
    session_paths = ["{}/Data_S{:02d}_Sess{:02d}_filtered_raw.fif".format(filtered_dir, i, j)
                    for i in range(1, 26+1)
                    for j in range(1, 5+1)]

    
    for file_path in session_paths:
        print('==============================================================================================')
        print('==============================================================================================')
        print(file_path)
        file_name = os.path.basename(file_path)

        # Load the preprocessed .fif file
        raw = mne.io.read_raw_fif(file_path, preload=True)

        # Initialize and fit ICA
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, method='fastica', max_iter=max_iter)
        ica.fit(raw)

        # Plot the components to visualize their time courses and topographies
        # print('--------------------- Plot Components ---------------------')
        # ica.plot_components()

        # Automatically find the EOG artifacts
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG')
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
        corrected_raw = raw.copy()
        ica.apply(corrected_raw)

        # Save ICA cleaned Raw data
        cleaned_file_name = os.path.basename(file_name).replace('_filtered_raw.fif', '_cleaned_raw.fif')
        print(cleaned_file_name)
        cleaned_file_path = os.path.join(cleaned_dir, cleaned_file_name)
        print(cleaned_file_path)

        corrected_raw.save(cleaned_file_path, fmt='double', overwrite=True)

        print(f"=====================================> Processed and saved cleaned data to {cleaned_file_path}")



    print("remove artifact")