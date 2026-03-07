import pandas as pd
import numpy as np
import zipfile
import os

# --- PATH CONFIGURATION ---
zip_path = r"C:\Users\nolab\Downloads\SmartInsole_Dataset_v1_0_Public (3).zip"
save_path = r"C:\Users\nolab\Downloads\Gait_Analysis_Results.csv"

# Constant for Distance-based calculations
TOTAL_DISTANCE_M = 20.0


# --- HELPERS ---
def safe_round(val, decimals=4):
    """Safely rounds a value if it is not None."""
    return round(val, decimals) if val is not None else None


def get_event_onsets(df, column_name, label='HES'):
    """
    Finds the timestamps where an event STARTS.
    """
    if column_name not in df.columns:
        return []

    is_event = df[column_name] == label

    # Find "Rising Edge": Current is True AND Previous was False
    mask = is_event & (is_event.shift(1).fillna(False) != True)
    onsets = df[mask]

    time_col = None
    for t in ['# time', 'Time', 'time', 'Timestamp']:
        if t in df.columns:
            time_col = t
            break

    if time_col:
        return onsets[time_col].tolist()
    return []


# --- 1. AVERAGE STEP TIME ---
def calculate_avg_step_time(df):
    left_onsets = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'HES')
    right_onsets = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'HES')

    events = []
    for t in left_onsets:
        events.append({'Time': t, 'Side': 'Left'})
    for t in right_onsets:
        events.append({'Time': t, 'Side': 'Right'})

    timeline = sorted(events, key=lambda x: x['Time'])
    step_times = []

    for i in range(len(timeline) - 1):
        current_step = timeline[i]
        next_step = timeline[i + 1]
        if current_step['Side'] != next_step['Side']:
            dt = next_step['Time'] - current_step['Time']
            step_times.append(dt)

    return np.mean(step_times) if step_times else None


# --- 2. AVERAGE STRIDE TIME (GAIT CYCLE) ---
def calculate_avg_stride_time(df):
    all_strides = []
    left_onsets = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'HES')
    if len(left_onsets) > 1:
        all_strides.extend(np.diff(left_onsets))

    right_onsets = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'HES')
    if len(right_onsets) > 1:
        all_strides.extend(np.diff(right_onsets))

    return np.mean(all_strides) if all_strides else None


# --- 3. AVERAGE STANCE TIME ---
def calculate_avg_stance_time(df):
    time_col = None
    for t in ['# time', 'Time', 'time', 'Timestamp']:
        if t in df.columns:
            time_col = t
            break
    if not time_col: return None

    stance_labels = ['HES', 'FOF', 'HER']
    durations = []

    for side in ['Left', 'Right']:
        col_name = f'Event- Label level 2- {side} Foot'
        if col_name in df.columns:
            is_stance = df[col_name].isin(stance_labels)
            temp_df = df[[col_name, time_col]].copy()
            temp_df['grp'] = (is_stance != is_stance.shift()).cumsum()
            stance_groups = temp_df[is_stance].groupby('grp')[time_col]

            if not stance_groups.size().empty:
                block_durations = stance_groups.agg(np.ptp)
                valid_durations = block_durations[block_durations > 0]
                durations.extend(valid_durations.tolist())

    return np.mean(durations) if durations else None


# --- 4. AVERAGE SWING TIME ---
def calculate_avg_swing_time(df):
    """Time from Toe Off to Heel Strike of the SAME foot."""
    left_hes = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'HES')
    right_hes = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'HES')
    left_tof = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'TOF')
    right_tof = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'TOF')

    swing_times = []
    # Left foot
    for tof in left_tof:
        next_hes = [h for h in left_hes if h > tof]
        if next_hes: swing_times.append(next_hes[0] - tof)
    # Right foot
    for tof in right_tof:
        next_hes = [h for h in right_hes if h > tof]
        if next_hes: swing_times.append(next_hes[0] - tof)

    return np.mean(swing_times) if swing_times else None


# --- 5. AVERAGE SINGLE SUPPORT TIME ---
def calculate_avg_single_support_time(df):
    """Time from Toe Off of one foot until Heel Strike of the OTHER foot."""
    left_hes = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'HES')
    right_hes = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'HES')
    left_tof = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'TOF')
    right_tof = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'TOF')

    ss_times = []
    # Left TOF to Right HES
    for tof in left_tof:
        next_hes = [h for h in right_hes if h > tof]
        if next_hes: ss_times.append(next_hes[0] - tof)
    # Right TOF to Left HES
    for tof in right_tof:
        next_hes = [h for h in left_hes if h > tof]
        if next_hes: ss_times.append(next_hes[0] - tof)

    # Filter out unusually large anomalies (> 1.5 seconds)
    valid_ss = [t for t in ss_times if t < 1.5]
    return np.mean(valid_ss) if valid_ss else None


# --- 6. AVERAGE DOUBLE SUPPORT TIME ---
def calculate_avg_double_support_time(df):
    """Time from Heel Strike of one foot until Toe Off of the OTHER foot."""
    left_hes = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'HES')
    right_hes = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'HES')
    left_tof = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'TOF')
    right_tof = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'TOF')

    ds_times = []
    # Left HES to Right TOF
    for hes in left_hes:
        next_tof = [t for t in right_tof if t > hes]
        if next_tof: ds_times.append(next_tof[0] - hes)
    # Right HES to Left TOF
    for hes in right_hes:
        next_tof = [t for t in left_tof if t > hes]
        if next_tof: ds_times.append(next_tof[0] - hes)

    valid_ds = [t for t in ds_times if t < 1.0]
    return np.mean(valid_ds) if valid_ds else None


# --- MAIN EXECUTION ---
print(f"Opening zip file: {zip_path}...")
results = []

try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        target_files = [f for f in all_files if (f.endswith('.xlsx') or f.endswith('.csv')) and '__MACOSX' not in f]

        print(f"Found {len(target_files)} files in zip. Processing...")

        for file_name in target_files:
            try:
                with z.open(file_name) as f:
                    try:
                        if file_name.endswith('.csv'):
                            df = pd.read_csv(f)
                        else:
                            df = pd.read_excel(f)
                    except Exception:
                        with z.open(file_name) as f_retry:
                            df = pd.read_csv(f_retry)

                df.columns = df.columns.str.strip()

                # Fetch Time column name
                time_col = None
                for t in ['# time', 'Time', 'time', 'Timestamp']:
                    if t in df.columns:
                        time_col = t
                        break

                # Calculate Temporal Averages
                avg_step = calculate_avg_step_time(df)
                avg_stride = calculate_avg_stride_time(df)
                avg_stance = calculate_avg_stance_time(df)
                avg_swing = calculate_avg_swing_time(df)
                avg_single_sup = calculate_avg_single_support_time(df)
                avg_double_sup = calculate_avg_double_support_time(df)

                if avg_step is not None and avg_stride is not None and time_col is not None:

                    # --- Phase Percentages (%) ---
                    # Stride time is our Gait Cycle duration
                    stance_pct = (avg_stance / avg_stride) * 100 if avg_stance else None
                    swing_pct = (avg_swing / avg_stride) * 100 if avg_swing else None
                    single_sup_pct = (avg_single_sup / avg_stride) * 100 if avg_single_sup else None
                    double_sup_pct = (avg_double_sup / avg_stride) * 100 if avg_double_sup else None

                    # --- Distance, Velocity, and Frequency Metrics ---
                    left_hes = get_event_onsets(df, 'Event- Label level 2- Left Foot', 'HES')
                    right_hes = get_event_onsets(df, 'Event- Label level 2- Right Foot', 'HES')

                    total_steps = len(left_hes) + len(right_hes)

                    # Estimate total "Gait Time" by measuring from the first step to the last step
                    # (This prevents idle standing time from ruining the cadence calculation)
                    all_hes = sorted(left_hes + right_hes)
                    gait_time = (all_hes[-1] - all_hes[0]) if len(all_hes) >= 2 else (
                                df[time_col].max() - df[time_col].min())

                    if total_steps > 0 and gait_time > 0:
                        step_length = TOTAL_DISTANCE_M / total_steps
                        stride_length = step_length * 2  # A stride is two steps
                        step_freq = (total_steps / gait_time) * 60  # Cadence
                        gait_velocity = stride_length / avg_stride
                        walk_ratio = (step_length * 1000) / step_freq if step_freq > 0 else None
                    else:
                        step_length = stride_length = step_freq = gait_velocity = walk_ratio = None

                    # --- Extract Info ---
                    parts = file_name.split('/')
                    filename = parts[-1]
                    patient_id = parts[-2] if len(parts) > 2 else "Unknown"
                    group = parts[-3] if len(parts) > 3 else "Unknown"

                    # Append to results
                    results.append({
                        'Group': group,
                        'Patient ID': patient_id,
                        'File': filename,
                        'Avg Step Time (s)': safe_round(avg_step),
                        'Avg Stride Time (s)': safe_round(avg_stride),
                        'Avg Stance Time (s)': safe_round(avg_stance),
                        'Stance Phase (%)': safe_round(stance_pct, 2),  # Round % to 2 decimals
                        'Avg Swing Time (s)': safe_round(avg_swing),
                        'Swing Phase (%)': safe_round(swing_pct, 2),
                        'Avg Single Support (s)': safe_round(avg_single_sup),
                        'Single Support (%)': safe_round(single_sup_pct, 2),
                        'Avg Double Support (s)': safe_round(avg_double_sup),
                        'Double Support (%)': safe_round(double_sup_pct, 2),
                        'Step Length (m)': safe_round(step_length),
                        'Stride Length (m)': safe_round(stride_length),
                        'Gait Velocity (m/s)': safe_round(gait_velocity),
                        'Step Frequency (steps/min)': safe_round(step_freq, 2),
                        'Walk Ratio (mm/step/min)': safe_round(walk_ratio, 2)
                    })

            except Exception as e:
                pass

except FileNotFoundError:
    print(f"ERROR: Could not find zip file at {zip_path}")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# Save Results
if results:
    output_df = pd.DataFrame(results)
    output_df.to_csv(save_path, index=False)
    print("\nAll Done!")
    print(output_df.head(10))
    print(f"\nFull results saved to: {save_path}")
else:
    print("No valid data found.")