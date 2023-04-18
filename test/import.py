import mne
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT_DIR, 'test')

def statistics(data,info):
    start_time = int((1500 / 1000) * info['sfreq'])
    end_time = int((1800 / 1000) * info['sfreq'])
    motor_channels = ['C3..', 'Cz..', 'C4..']
    data = mne.io.read_raw_edf(data)

    dataset = data.get_data(picks=motor_channels).T
    df_describe = pd.DataFrame(dataset)
    df_describe.describe()
def plots(data):
    rawdata= mne.io.read_raw_edf(data)
    Oz_unit = rawdata.__dict__['_orig_units']['Oz..']
    fig, ax = plt.subplots(figsize=[15, 5])
    start_time = 15
    end_time = 16

    plt.ylabel(Oz_unit, fontsize=18)
    plt.xlabel('Time', fontsize=18)
    ax.plot(data.get_data(picks='Oz..', tmin=start_time, tmax=end_time).T)
    y_fmt = tick.FormatStrFormatter('%5.0e')
    ax.yaxis.set_major_formatter(y_fmt)
    plt.show()
def importdata(data):
    data = mne.io.read_raw_edf(data)
    rawgetdata = data.get_data()
    # you can get the metadata included in the file and a list of all channels:
    info = data.info
    channels = data.ch_names
    return data,rawgetdata,info,channels
if __name__ == "__main__":
    datafile="S001R01.edf"
    rawdata,rawgetdata,info,channels=importdata(datafile)
    #print(raw_data,info,channels)
    #statistics(data,info)
    #plots(data)
    low_cut=1
    hi_cut=30
    raw2=rawdata.copy()
    raw_filt = raw2.filter(low_cut, hi_cut)
    #raw_filt.plot_psd(fmax=10)
