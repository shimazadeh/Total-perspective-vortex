import pyedflib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci

def read_process_data(subjects, run_execution, run_imagery):
    
    raw_files = []
    for subject in subjects:
        for i, j in zip(run_execution, run_imagery):
            raw_files_execution = [read_raw_edf(file, preload=True, stim_channel='auto') for file in eegbci(subject, i)]
            raw_files_imagery = [read_raw_edf(file, preload=True, stim_channel='auto') for file in eegbci(subject, j)]

            raw_actions = concatenate_raws(raw_files_execution)
            raw_imagine = concatenate_raws(raw_files_imagery)

            events, _ = mne.events_from_annotations(raw_actions, event_id=dict(T0=1, T1=2, T2=3))
            mapping = {1:'rest', 2: 'do/feet', 3: 'do/hands'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, 
                sfreq=raw_actions.info['sfreq'],
                orig_time=raw_actions.info['meas_date'])
            raw_actions.set_annotations(annot_from_events)

            events, _ = mne.events_from_annotations(raw_imagine, event_id=dict(T0=1, T1=2, T2=3))
            mapping = {1:'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, 
                sfreq=raw_imagine.info['sfreq'],
                orig_time=raw_imagine.info['meas_date'])
            raw_imagine.set_annotations(annot_from_events)

            raw_files.append(raw_actions)
            raw_file.append(raw_imagery)
    
    raw_data = concatenate_raws(raw_files)
    event, event_dict = mne.events_from_annotations(raw_data)
    data = raw_data.get_data()

    return event, event_dict, data, raw_data

def filter_eye_artifacts(raw, method, fit_params=None):
    raw_corrected = raw.copy()
    n_components = 20
    
    ica = ICA(n_components=20, method=method, fit_params=fit_params, random_state=97)
    t0 = time()
    ica.fit(raw_corrected, picks=picks)
    fit_time = time() - t0
    ica.plot_components()
    plt.show()
    
    eog_indicies, scores= ica.find_bads_eog(raw, ch_name='Fpz', threshold=1.5)
    ica.plot_scores(scores, exclude=eog_indicies)
    ica.exclude.extend(eog_indicies)
    raw_corrected = ica.apply(raw_corrected, n_pca_components = n_components, exclude = ica.exclude)
    
    return raw_corrected

def main():
    with open(sys.argv[1], 'r') as yaml_file:
        params = yaml.safeload(yaml_file)

    subjects = params['subjects']
    run_execution = params['action_tasks']
    run_imaginary = params['imaginary_tasks']

    if len(subjects) == 0 or len(run_execution) == 0 or len(run_execution) != len(run_imaginary):
        print("Error: ")

    #read and process data
    event, event_dict, data, raw_data = read_process_data(subjects, run_execution, run_imaginary)

    #filter bad channels , set montage image to the data
    picks = pick_types(raw_data.info, msg-False, eeg=True, stim=False, eog=False, exclude='bads')

    biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    biosemi_montage.plot()
    plt.show()

    eegbci.standardize(raw_data)
    montage = make_standard_montage('standard_1005')
    raw_data.set_montage(montage)

    #visualize raw_data vs filtered
    raw_data.plot_psd(average=False)
    plt.save_fig('raw_data.png')

    fig = mne.viz.plot_events(event, sfreq=raw_data.info['sfreq'], first_stamp=raw_data.first_samp, event_id=event_dict)
    fig.subplots_adjust(right=0.7)
    fig.save_fig('raw_data_events.png')

    #apply bandpass filter
    raw_data.filter(5., 40., fir_design='firwin', skip_by_annotation='edge')
    raw_data.plot_psd(average=False)
    plt.save_fig('filtered_data.png')

    #exclude eye artifacts
    raw_fastica = filter_eye_artifacts(raw_data, 'fastica')


    #isolate event and convert them to epoches
    event_id = {'do/feet': 1, 'do/hands': 2, 'imagine/feet': 3, 'imagine/hands': 4}
    tmin, tmax = -1.0, 4.0

    events, event_dict = mne.events_from_annotations(raw_data, event_id=event_id)
    epochs = Epochs(raw_data, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

    #training
    y = epochs.events[:, -1] - 1
    X = epochs.get_data() #3d array: the epoch, channels, time points

        #using CSP transformers
    csp1 = CSP(n_components=10, reg=None, log=True)
    csp2 = CSP(n_components=10, reg=None, log=True)
    csp3 = CSP(n_components=10, reg=None, log=True)

    pipeline_creation(X, y, csp1, csp2, csp3)
        
        #using Spoc transformers
    Spoc1 = SPoC(n_components=15, reg='oas', log=True, rank='full')
    Spoc2 = SPoC(n_components=15, reg='oas', log=True, rank='full')
    Spoc3 = SPoC(n_components=15, reg='oas', log=True, rank='full')

    pipeline_creation(X, y, Spoc1, Spoc2, Spoc3)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: Correct use of program: main.py config.yaml")
        exit(-1)

    main()
   