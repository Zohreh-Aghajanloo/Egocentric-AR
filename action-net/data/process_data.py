import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


def low_pass_filter(data, cutoff=5, fs=200):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data


def segment_data(df, segment_size=100):
    # List to store the segmented data
    segmented_data = []

    # for i, row in df.iterrows():
    #     if row['description'] not in labels_dict:
    #         continue
    #     start, stop = row['start'], row['stop']
    #     duration = stop - start
    #     num_segments = int(np.floor(duration / segment_size))
        
    #     for j in range(num_segments):
    #         seg_start = start + j * segment_size
    #         seg_stop = seg_start + segment_size
            
    #         segment = row.copy()
    #         segment['idx'] = i
    #         segment['start'] = seg_start
    #         segment['stop'] = seg_stop
    #         segment['label'] = labels_dict[row['description']]
            
    #         for arm in ['myo_left', 'myo_right']:
    #             readings = np.array(row[f'{arm}_readings'])
    #             timestamps = np.array(row[f'{arm}_timestamps'])
                
    #             # Segment the readings and timestamps
    #             mask = (timestamps >= seg_start) & (timestamps < seg_stop)
    #             segment[f'{arm}_readings'] = readings[mask]
    #             segment[f'{arm}_timestamps'] = timestamps[mask]
            
    #         # Handle the case where the number of the readings is different for the sensors
    #         min_len = min(segment['myo_left_readings'].shape[0], segment['myo_right_readings'].shape[0])
    #         segment['myo_left_timestamps'] = segment['myo_left_timestamps'][:min_len]
    #         segment['myo_right_timestamps'] = segment['myo_right_timestamps'][:min_len]
    #         segment['myo_left_readings'] = segment['myo_left_readings'][:min_len]
    #         segment['myo_right_readings'] = segment['myo_right_readings'][:min_len]
    #         segmented_data.append(segment)
    
    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        # Segment the left readings and timestamps
        left_readings = np.array(row['myo_left_readings'])
        left_timestamps = np.array(row['myo_left_timestamps'])
        
        # Segment the right readings and timestamps
        right_readings = np.array(row['myo_right_readings'])
        right_timestamps = np.array(row['myo_right_timestamps'])

        # Handle the case where the number of the readings is different for the sensors
        if left_timestamps.shape[0] < right_timestamps.shape[0]:
            min_len = left_timestamps.shape[0]
            final_timestamps = left_timestamps
        else:
            min_len = right_timestamps.shape[0]
            final_timestamps = right_timestamps
        
        # Determine how many segments can be created
        num_segments = min_len // segment_size
        
        for i in range(num_segments):
            segmented_data.append({
                'idx': idx,
                'label': labels_dict[row['description']],
                'start': final_timestamps[i * segment_size],
                'stop': final_timestamps[(i + 1) * segment_size - 1],
                'myo_left_readings': left_readings[i * segment_size: (i + 1) * segment_size],
                'myo_left_timestamps': left_timestamps[i * segment_size: (i + 1) * segment_size],
                'myo_right_readings': right_readings[i * segment_size: (i + 1) * segment_size],
                'myo_right_timestamps': right_timestamps[i * segment_size: (i + 1) * segment_size],
            })

    return pd.DataFrame(segmented_data, index=range(len(segmented_data)))


labels_dict = { 'Get/replace items from refrigerator/cabinets/drawers': 0,
                'Get items from refrigerator/cabinets/drawers': 0,
                'Peel a cucumber': 1,
                'Clear cutting board': 2,
                'Slice a cucumber': 3,
                'Peel a potato': 4,
                'Slice a potato': 5,
                'Slice bread': 6,
                'Spread almond butter on a bread slice': 7,
                'Spread jelly on a bread slice': 8,
                'Open/close a jar of almond butter': 9,
                'Open a jar of almond butter': 9,
                'Pour water from a pitcher into a glass': 10,
                'Clean a plate with a sponge': 11,
                'Clean a plate with a towel': 12,
                'Clean a pan with a sponge': 13,
                'Clean a pan with a towel': 14,
                'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 15,
                'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 16,
                'Stack on table: 3 each large/small plates, bowls': 17,
                'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 18,
                'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 19,
}

emg_data = {'S00_2': None,
            'S01_1': None,
            'S02_2': None,
            'S02_3': None,
            'S02_4': None,
            'S03_1': None,
            'S03_2': None,
            'S04_1': None,
            'S05_2': None,
            'S06_1': None,
            'S06_2': None,
            'S07_1': None,
            'S08_1': None,
            'S09_2': None
            }

samples = pd.DataFrame()
for s in emg_data:
    df =  pd.DataFrame(pd.read_pickle(f'emg\{s}.pkl'))
    emg_data[s] = segment_data(df)
    emg_data[s]['subject'] = s
    for i, row in emg_data[s].iterrows():
        # Rectification
        myo_left_abs = np.abs(row['myo_left_readings'])
        myo_right_abs = np.abs(row['myo_right_readings'])

        # Low-pass filtering
        myo_left_filtered = low_pass_filter(myo_left_abs)
        myo_right_filtered = low_pass_filter(myo_right_abs)

        # Normalization
        myo_left_normalized = normalize(myo_left_filtered)
        myo_right_normalized = normalize(myo_right_filtered)

        # # Rectification
        # myo_left_normalized_abs = np.abs(myo_left_normalized)
        # myo_right_normalized_abs = np.abs(myo_right_normalized)

        # # Sum across channels for forearm activation
        # forearm_activation_left = np.sum(myo_left_normalized_abs, axis=1)
        # forearm_activation_right = np.sum(myo_right_normalized_abs, axis=1)
        
        # # Smooth the summed signals
        # forearm_activation_left_smooth = low_pass_filter(forearm_activation_left)
        # forearm_activation_right_smooth = low_pass_filter(forearm_activation_right)

        # Store processed data
        emg_data[s].at[i, 'myo_left_readings'] = myo_left_normalized
        emg_data[s].at[i, 'myo_right_readings'] = myo_right_normalized
        # emg_data[s].at[i, 'myo_left_readings'] = forearm_activation_left_smooth
        # emg_data[s].at[i, 'myo_right_readings'] = forearm_activation_right_smooth
    
    samples = pd.concat([samples, emg_data[s]], ignore_index=True)

sub4_samples = samples[samples['subject'] == 'S04_1']

y = samples['label']
X_train, X_test, y_train, y_test = train_test_split(samples, y, test_size=0.2, stratify=y, random_state=42)
# Calculate loss weights for each class:
class_counts = X_train['label'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
print(class_weights / class_weights.mean())
X_train.to_pickle('train_set.pkl')
X_test.to_pickle('test_set.pkl')

y = sub4_samples['label']
X_train, X_test, y_train, y_test = train_test_split(sub4_samples, y, test_size=0.2, stratify=y, random_state=42)
# Calculate loss weights for each class:
class_counts = X_train['label'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
print(class_weights / class_weights.mean())
X_train.to_pickle('multimodal_train_set.pkl')
X_test.to_pickle('multimodal_test_set.pkl')