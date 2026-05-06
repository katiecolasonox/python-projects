import requests
import pandas as pd
import numpy as np
from datetime import datetime

def get_data(client_id, client_secret, refresh_token):
    # 1. Get new access token
    url = "https://www.strava.com/oauth/token"
    payload = {
        'client_id' : client_id,
        'client_secret' : client_secret,
        'refresh_token' : refresh_token,
        'grant_type' : 'refresh_token' # already have a master key, will request frequently 
    }

    response = requests.post(url, data=payload).json() # retrieving all token information
    new_token = response['access_token'] # storing new access token information 
    headers = {'Authorization' : f'Bearer {new_token}'}  # sending Strava server new token id

    # 2. Fetch the list of activites 
    start_timestamp = int(datetime(2025, 10, 1).timestamp())
    activities_url = "https://www.strava.com/api/v3/athlete/activities"
    params = {'after': start_timestamp, 'per_page': 200}
    r = requests.get(activities_url, headers=headers, params=params)
    activities = r.json()

    # 3. Loop through all activties and pull streams for each one
    all_stream_data = []

    for activity in activities:
        if activity['type'] == 'Run': # only pulling data for runs
            act_id = activity['id']
            stream_url = f"https://www.strava.com/api/v3/activities/{act_id}/streams"
            stream_params = {
                'keys': 'time,distance,altitude,grade_smooth,velocity_smooth,heartrate',
                'key_by_type': 'true'
            }
            
            s = requests.get(stream_url, headers=headers, params=stream_params)

            if s.status_code == 200:
                s_data = s.json()
                # create a temporary dataframe for this specific run's streams
                temp_df = pd.DataFrame({
                    'run_id': act_id,
                    'time': s_data['time']['data'], # in seconds 
                    'distance': s_data['distance']['data'], # in meters
                    'grade': s_data['grade_smooth']['data'], # calculated incline at exact point
                    'vel_ms': s_data['velocity_smooth']['data'], # speed
                    'alt_m': s_data['altitude']['data'], # in meters 
                    'heartrate': s_data['heartrate']['data'] # beats per min
                    #'temp': s_data['temp']['data']
                })
                all_stream_data.append(temp_df)
    
    master_streams = pd.concat(all_stream_data, ignore_index=True)
    activities_df = pd.DataFrame(activities)

    return master_streams, datetime.now().strftime("%Y-%m-%d %H:%M")

def build_grade_model(streams):
    # renaming columns to add in units suffix
    streams = streams.rename(columns={
    'distance': 'distance_m',
    'time' : 'time_s',
    'grade' : 'linear_grade',
    'alt_m' : 'elevation_m'
    }).copy()
    
    # calculate deltas and converting units 
    g = streams.groupby('run_id')
    streams['dist_mi_diff'] = g['distance_m'].diff() * 0.000621371 # converting to miles
    streams['elev_ft_diff'] = g['elevation_m'].diff() * 3.28084 # converting to ft
    streams['time_s_diff'] = g['time_s'].diff()

    # binning and aggregation
    # creating bin column for every 30 meters
    streams['dist_bin'] = (streams['distance_m'] / 30).astype(int)

    # filtering for "moving" only to avoid skewing the pace curve
    moving_streams = streams[streams['vel_ms'] > 0.5].copy()

    # aggregating metrics by run and bin
    streams_binned = moving_streams.groupby(['run_id', 'dist_bin']).agg({
    'time_s_diff' : 'sum',
    'dist_mi_diff' : 'sum',
    'elev_ft_diff' : 'sum',
    'heartrate' : 'mean'
    }).reset_index()

    # use a small placeholder to avoid division by zero
    epsilon = 1e-9 
    streams_binned['linear_grade_pct'] = (
        streams_binned['elev_ft_diff'] / (streams_binned['dist_mi_diff'] * 52.8 + epsilon)
    )
    streams_binned['pace_min_mi'] = (
        (streams_binned['time_s_diff'] / 60) / (streams_binned['dist_mi_diff'] + epsilon))

    # establishing grade curve 
    grade_curve = (
        streams_binned[
            (streams_binned['linear_grade_pct'].between(-30, 30)) &  # only linear grades between -30 and 30
            (streams_binned['pace_min_mi'] < 30) # pace has to be greater than 30 minutes per mile 
        ]
        .groupby(streams_binned['linear_grade_pct'].round())['pace_min_mi']
        .agg(['median', 'count'])
    )

    grade_curve = grade_curve['median']

    return grade_curve 

def run_silverton_prediction(course_df, grade_curve):
    epsilon = 1e-9 
    # deltas for elevation and distance
    course_df['delta_elev_ft'] = course_df['Elevation (feet)'].diff()
    course_df['delta_dist_mi'] = course_df['Distance (miles)'].diff()

    # linear grade calcuation
    course_df['linear_grade_pct'] = (course_df['delta_elev_ft'] / (course_df['delta_dist_mi'] * 52.8 + epsilon)).round()

    # adding in predicted pace from personal grade curve 
    course_df['base_predicted_pace'] = course_df['linear_grade_pct'].map(grade_curve)
    
    # calculating cumulative gain and loss
    course_df['cum_gain'] = course_df['delta_elev_ft'].clip(lower=0).cumsum()
    course_df['cum_loss'] = course_df['delta_elev_ft'].clip(upper=0).abs().cumsum()

    # calcuating the length of the sustained climb 
    course_df['is_up'] = course_df['linear_grade_pct'] > 0

    # Logic for sustained blocks
    blocks = (course_df['is_up'] != course_df['is_up'].shift()).cumsum()
    course_df['sustained_block_dist'] = course_df.groupby(blocks)['delta_dist_mi'].cumsum()

    # Dynamic fatigure & altitude scaling

    # Altitude Penalty: 1% slower for every 1000ft above Boulder (5300ft)
    alt_penalty = 1 + (np.maximum(0, course_df['Elevation (feet)'] - 5300) / 1000 * 0.015)

    # Fatigue Factors
    # general race fatigue: 1% slowdown for every 5 miles
    global_fatigue = 1 + (course_df['Distance (miles)'] / 5 * 0.01) 

    #  vertical fatigue: slow down 5% for every for every 2,000ft of cumulative gain 
    vert_fatigue = 1 + (course_df['cum_gain'] / 2000 * 0.05)

    # Sustained Penalties 
    # slow down for 5% for every continuous mile of uphill and 2% for every continuous mile of downhill
    sustained_mult = np.where(
        course_df['sustained_block_dist'] > 1.0,
        np.where(course_df['is_up'], 
                1 + (course_df['sustained_block_dist'] * 0.05), # Uphill
                1 + (course_df['sustained_block_dist'] * 0.02)), # Downhill
        1.0
    )

    # Final Fatigued Pace per race segement
    course_df['fatigued_pace'] = (
        course_df['base_predicted_pace'] * alt_penalty * global_fatigue * vert_fatigue * sustained_mult)
    
    # Predicted finish time
    course_df['segment_time_mins'] = course_df['fatigued_pace'] * course_df['delta_dist_mi']
    predicted_finish_time = course_df['segment_time_mins'].sum()

    return predicted_finish_time