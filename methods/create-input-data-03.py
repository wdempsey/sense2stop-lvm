# %%
# Create "mock" true latent times using dict_knitted_with_puffmarker
# Then we will test this first assuming that observed times come from Self-Report and/or Random EMA only

latent_data = {}

for participant in dict_knitted_with_puffmarker.keys():
    current_participant_dict = {}
    for days in dict_knitted_with_puffmarker[participant].keys():
        current_data = dict_knitted_with_puffmarker[participant][days]
        all_puff_time = []
        if len(current_data.index)==0:
            next
        else:
            # note that even if participant reported "Yes" smoked, 
            # delta is set to missing if in Self Report participant reported to have smoked "more than 30 minutes ago"
            # or in Random EMAs where participant reported to have smoked "more than 2 hours ago"
            current_data_yes = current_data[(current_data['smoke']=='Yes') & ~(np.isnan(current_data['delta']))]
            if len(current_data_yes)==0:
                next
            else:
                for this_row in range(0, len(current_data_yes.index)):
                    if current_data_yes['adjusted'].iloc[this_row]==1:
                        tmp = current_data_yes['puff_time_adjusted'].iloc[this_row]
                        # tmp is an array with 1 element, and this element is a dictionary
                        # hence, we need the next steps
                        tmp = tmp[0]
                        tmp_to_array = []
                        for pm_key in tmp.keys():
                            tmp_to_array.append(tmp[pm_key])
                        # we're out of the loop
                        all_puff_time.extend(tmp_to_array)
                    elif current_data_yes['adjusted'].iloc[this_row]==0:
                        all_puff_time.append(current_data_yes['puff_time_adjusted'].iloc[this_row])
                    else:
                        next

        # The output of the next line should be one number only, but it will be an array object
        # Hence, the .iloc[0] converts the array object into a float
        current_day_length = data_day_limits[(data_day_limits['participant_id']==participant) & (data_day_limits['study_day']==days)]['day_length'].iloc[0]
        new_dict = {'participant_id':participant, 
                    'study_day':days, 
                    'day_length':current_day_length, 
                    'latent_event_order': (np.arange(len(all_puff_time))),  # begins with zero (not 1)
                    'hours_since_start_day': np.array(all_puff_time)}
        current_participant_dict.update({days: new_dict})
    # Add this participant's data to dictionary
    latent_data.update({participant:current_participant_dict})




# %%
# Create "mock" observed data
clean_data = copy.deepcopy(dict_knitted_with_puffmarker)  # Keep dict_knitted_with_puffmarker untouched

for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        current_data = clean_data[participant][days]
        if len(current_data.index)>0:
            current_data = current_data.loc[:, ['assessment_type', 'hours_since_start_day', 'hours_since_start_day_shifted','smoke','when_smoke']]
            current_data = current_data.rename(columns = {'assessment_type':'assessment_type',
                                                          'hours_since_start_day':'assessment_begin', 
                                                          'hours_since_start_day_shifted':'assessment_begin_shifted',
                                                          'smoke':'smoke',
                                                          'when_smoke':'windowtag'})
            clean_data[participant][days] = current_data


# %%
# Let's simply use Self-Reports for now as out "mock" observed data
for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        current_data = clean_data[participant][days]
        if len(current_data.index)>0:
            # Create variable: order at which the participant initiated a particular assessment
            current_data['assessment_order'] = np.arange(len(current_data.index))  # begins with zero (not 1)
            current_data = current_data.loc[:, ['assessment_order','assessment_type','assessment_begin_shifted','assessment_begin', 'smoke', 'windowtag']]
            clean_data[participant][days] = current_data



# %%
# Now, let's convert each PERSON-DAY of clean_data into a dictionary
for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        current_data = clean_data[participant][days]
        if len(current_data.index)>0:
            current_dict = {'participant_id':participant,
                            'study_day': days,
                            'assessment_order': np.array(current_data['assessment_order']),
                            'assessment_type': np.array(current_data['assessment_type']),
                            'assessment_begin_shifted': np.array(current_data['assessment_begin_shifted']),
                            'assessment_begin': np.array(current_data['assessment_begin']),
                            'smoke': np.array(current_data['smoke']),
                            'windowtag': np.array(current_data['windowtag'])}
            clean_data[participant][days] = current_dict
        else:
            current_dict = {'participant_id':participant,
                            'study_day': days,
                            'assessment_order': np.array([]),
                            'assessment_type': np.array([]),
                            'assessment_begin_shifted': np.array([]),
                            'assessment_begin': np.array([]),
                            'smoke': np.array([]),
                            'windowtag': np.array([])}
            clean_data[participant][days] = current_dict



