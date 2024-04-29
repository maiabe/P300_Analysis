import pandas as pd
import os
import mne

channel_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                    'P6', 'P8', 'PO7', 'POz', 'O1', 'O2', 'EOG']
channel_types = ['eog' if name == 'EOG' else 'eeg' for name in channel_names]
sampling_rate=200

def save_original_raw():
    original_dir = "data/allData/"

    # Define output directory and file to save original data as raw files
    output_dir = "data/originalRaw/"
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate Over Session Files to load each EEG session data (Make sure all sessions exist)
    session_files = ["{}/Data_S{:02d}_Sess{:02d}.csv".format(original_dir, i, j)
                    for i in range(1, 26+1)
                    for j in range(1, 5+1)]
    
    # Iterate through each session files in allData directory
    for session_path in session_files:
        file_name = os.path.basename(session_path)
        fif_file_name = file_name.replace('.csv', '_raw.fif')
        output_file_path = os.path.join(output_dir, fif_file_name)

        print(output_file_path)
        
        # Create a dataframe of the current session data
        df = pd.read_csv(session_path)
        
        # Convert DataFrame to a NumPy array and transpose it to match MNE's expected format (in Volts)
        data = (df[channel_names].to_numpy().T) / 1e6   # Convert microvolts to volts

        # Create an MNE Info structure (contains information about the data)
        info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types=channel_types)

        # Create a RawArray object
        raw = mne.io.RawArray(data, info)

        # Set channel locations
        raw.set_montage("standard_1020")

        raw.save(output_file_path, fmt="double", overwrite=True)
        print(f"=========================> Saved original data to {output_file_path}")

    print("save original raw completed")

