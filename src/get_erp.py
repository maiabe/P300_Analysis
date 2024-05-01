import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt


def create_epochs(sfreq):

    # Define preprocessed dir
    preprocessed_dir = 'data/preprocessed/artifactRemoved'
    # Define original csv dir
    original_dir = 'data/allData'
    # Define filepath of Feedback Labels
    feedback_labels_path = 'data/AllDataLabels.csv'
    
    # Define directory to save the epoched data
    epoch_dir = 'data/epochs'
    
    # Ensure cleaned directory exists
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

        
    # Load the feedback labels
    feedback_labels_df = pd.read_csv(feedback_labels_path)

    ### Iterate through each session for each subject
    for i in range(1, 26+1):   # Num Subjects
        for j in range(1, 5+1):    # Num Sessions
            preprocessed_file_path = f"{preprocessed_dir}/Data_S{i:02d}_Sess{j:02d}_cleaned_raw.fif"
            print(preprocessed_file_path)
            original_csv_path = f"{original_dir}/Data_S{i:02d}_Sess{j:02d}.csv"
            print(original_csv_path)

            # Check if both files exist before processing
            if os.path.exists(preprocessed_file_path) and os.path.exists(original_csv_path):
                print("===============================================================================================")
                print("===============================================================================================")
                print(f"Processing: {preprocessed_file_path} and {original_csv_path}")

                ### Retrieve Event timestamps
                session_df = pd.read_csv(original_csv_path)

                # Get corresponding Feedback labels for each feedback event in this session from AllDataLabels.csv
                subject_id = f"S{i:02d}"
                session_id = f"Sess{j:02d}"
                search_pattern = f"{subject_id}_{session_id}"
                labels_df = feedback_labels_df[feedback_labels_df['IdFeedBack'].str.contains(search_pattern)]

                if labels_df.empty:
                    print(f"No entries found for subject {subject_id} and session {session_id}.")
                else:
                    print(f"Found entries for subject {subject_id} and session {session_id}:")
                    print(labels_df)


                # Identify relevant columns
                event_times_seconds = session_df.loc[session_df['FeedBackEvent'] == 1, 'Time']

                # Convert Event times to samples
                event_samples = (event_times_seconds * sfreq).astype(int)

                # Create the Events array for MNE
                events = np.column_stack((
                    event_samples,
                    np.zeros(len(event_samples), dtype=int),
                    labels_df['Prediction'] + 1 # Add 1 to Prediction value, 0 or 1 to make IDs 1 (incorrect) and 2 (correct)
                ))

                # Create Epochs based on incorrect and correct feedbacks (set the baseline from -0.2 from the stimulus to 0)
                raw = mne.io.read_raw_fif(preprocessed_file_path, preload=True)
                event_id = { 'incorrect': 1, 'correct': 2 }
                epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True)

                # Plot the epochs
                # print(">>>>> Original epochs")
                # epoch_fig = epochs.plot(picks=channels_to_plot, scalings=scalings, n_epochs=5, n_channels=13, title=f'S{i:02d} Sess{j:02d} Epochs', show=False)
                # epoch_fig.set_size_inches(15, 20)
                # epoch_fig.show()

                # Reject any remaining artifacts
                reject_criteria = dict(eeg=150e-6)
                epochs_auto_rejected = epochs.copy().drop_bad(reject=reject_criteria)
                print(epochs_auto_rejected.drop_log)
                # epochs_auto_rejected.plot_drop_log()

                if len(epochs_auto_rejected) == 0:
                    print("No epochs left after rejection. Skipping plotting and saving.")
                else:
                    # Plot the epochs after removing remaining artifacts
                    # print(">>>>> Epochs after removing remaining artifacts")
                    # epoch_auto_rejected_fig = epochs_auto_rejected.plot(picks=channels_to_plot, scalings=scalings, n_epochs=5, n_channels=13, title=f'S{i:02d} Sess{j:02d} Epochs', show=False)
                    # epoch_auto_rejected_fig.set_size_inches(15, 20)
                    # epoch_auto_rejected_fig.show()

                    # Save epoched data
                    epoch_file_path = f"{epoch_dir}/Data_S{i:02d}_Sess{j:02d}_epochs_epo.fif"
                    epochs_auto_rejected.save(epoch_file_path, fmt="double", overwrite=True)
                    print(f"=====================================> Saved epoch data to {epoch_file_path}")


            else:
                print(f"Missing file for Subject {i}, Session {j}.")

    print("create epochs")



# def average_epochs():
#     # Define epochs dir
#     epoch_dir = f"data/epochs"
#     # Define evoked dir
#     evoked_dir = f"data/evoked/"


#     # Ensure correct evoked directory exists
#     if not os.path.exists(evoked_dir):
#         os.makedirs(evoked_dir)

#     # Iterate through epochs for each session of each subject
#     for i in range(1, 26+1):   # Num Subjects
#         for j in range(1, 5+1):    # Num Sessions
#             epoch_path = f"{epoch_dir}/Data_S{i:02d}_Sess{j:02d}_epochs_epo.fif"
#             print(epoch_path)

#             # Check if both files exist before processing
#             if os.path.exists(epoch_path):
#                 print("===============================================================================================")
#                 print("===============================================================================================")
#                 print(f"Processing: {epoch_path}")

#                 # Load the epochs
#                 epochs = mne.read_epochs(epoch_path, preload=True)

#                 # Get correct and incorrect feedback epochs
#                 correct_epochs = epochs['correct']
#                 incorrect_epochs = epochs['incorrect']

#                 evokeds = []

#                 if len(correct_epochs) == 0:
#                     print("No epochs left after rejection. Skipping plotting and saving.")
#                 else:
#                     # Compute the average ERP from the epochs for correct and incorrect events
#                     correct_evoked = correct_epochs.average()
#                     # Assign comment to evoked object
#                     correct_evoked.comment = "correct"
#                     evokeds.append(correct_evoked)                    
                    
#                 if len(incorrect_epochs) == 0:
#                     print("No epochs left after rejection. Skipping plotting and saving.")
#                 else:
#                     # Compute the average ERP from the epochs for correct and incorrect events
#                     incorrect_evoked = incorrect_epochs.average()
#                     # Assign comment to evoked object
#                     incorrect_evoked.comment = "incorrect"
#                     evokeds.append(incorrect_evoked)
                
#                 # Save the list of evoked objects to a FIF file
#                 evoked_file_path = f"{evoked_dir}/Data_S{i:02d}_Sess{j:02d}_evoked-ave.fif"
#                 mne.write_evokeds(evoked_file_path, evokeds)
#                 print(f"=====================================> Saved evoked data to {evoked_file_path}")

#             else:
#                 print(f"Missing Epoch file for Subject {i}, Session {j}.")


# Averages epochs by sessions per subject
def average_epochs(subject_id, session_id, event_type):
    print("average_epochs called")
    
    # Define epochs path
    epochs_path = f"data/epochs/Data_S{subject_id:02}_Sess{session_id:02}_epochs_epo.fif"

    # Load epochs
    epochs = mne.read_epochs(epochs_path, preload=True)
    print(epochs.info)
    print(event_type)

    # Select epochs based on the event type and average them
    print(len(epochs[event_type]))
    if len(epochs[event_type]) == 0:
        print(f"epochs read is empty")
    else:
        evoked = epochs[event_type].average()
        print("===============================================================")
        print("===============================================================")
        print(evoked.info)
        print(evoked.get_data())
        print(evoked.get_data().shape)

        return evoked
    
    return None



def create_grand_average():

    grand_average_dir = f"data/grandAverage"

    # Ensure cleaned directory exists
    if not os.path.exists(grand_average_dir):
        os.makedirs(grand_average_dir)

    all_evoked_correct = []
    all_evoked_incorrect = []

    # Loop over subjects and sessions
    for subject_id in range(1, 27):  # Iterate through 26 subjects
        correct_subject_evokeds = []
        incorrect_subject_evokeds = []
        for session_id in range(1, 6):  # Iterate through 5 sessions per subject
            try:
                evoked_correct = average_epochs(subject_id, session_id, 'correct')
                correct_subject_evokeds.append(evoked_correct)
                evoked_incorrect = average_epochs(subject_id, session_id, 'incorrect')
                incorrect_subject_evokeds.append(evoked_incorrect)

                if(evoked_correct):
                    print(evoked_correct.info)
                    all_evoked_correct.append(evoked_correct)
                if(evoked_incorrect):
                    print(evoked_incorrect.info)
                    all_evoked_incorrect.append(evoked_incorrect)

            except FileNotFoundError:
                print(f"Data not found for Subject {subject_id} Session {session_id}")

        # Save averages by subjects
        subject_average_path = f"data/evoked/bySubject/Data_S{subject_id:02d}_evokeds-ave.fif"
        correct_subject_evokeds = [item for item in correct_subject_evokeds if item is not None]
        correct_subject_average = mne.grand_average(correct_subject_evokeds)
        correct_subject_average.comment = 'correct'

        incorrect_subject_evokeds = [item for item in incorrect_subject_evokeds if item is not None]
        incorrect_subject_average = mne.grand_average(incorrect_subject_evokeds)
        incorrect_subject_average.comment = 'incorrect'
        mne.write_evokeds(subject_average_path, evoked=[correct_subject_average, incorrect_subject_average], overwrite=True)

    # Compute grand averages across all epochs
    grand_average_correct = mne.grand_average(all_evoked_correct)
    grand_average_correct.comment = 'correct'
    grand_average_incorrect = mne.grand_average(all_evoked_incorrect)
    grand_average_incorrect.comment = 'incorrect'

    print("---------------- Grand Average Correct ----------------")
    print(grand_average_correct.info)
    print(grand_average_correct.get_data().shape)
    print("---------------- Grand Average Incorrect ----------------")
    print(grand_average_correct.info)
    print(grand_average_correct.get_data())
    print(grand_average_correct.get_data().shape)

    # Save grand averages
    grand_average_path = f"{grand_average_dir}/grand_averages-ave.fif"
    mne.write_evokeds(grand_average_path, evoked=[grand_average_correct, grand_average_incorrect], overwrite=True)


