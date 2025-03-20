import numpy as np
import pydarn
import datetime as dt
import bz2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
import os
import gc
'''
a = [r"C:\masters\fitacf files\1995\july\19950713.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950713.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950713.0800.00.han.fitacf.bz2"]
b = [r"C:\masters\fitacf files\1995\july\19950714.0000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.0200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.0800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.1400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.1600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.1800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.2000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\july\19950714.2200.00.han.fitacf.bz2"]
c = [r"C:\masters\fitacf files\1995\august\19950811.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\august\19950811.1400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\august\19950811.1600.00.han.fitacf.bz2"]
d = [r"C:\masters\fitacf files\1995\august\19950814.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\august\19950814.0800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\august\19950814.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\august\19950814.1200.00.han.fitacf.bz2"]
f = [r"C:\masters\fitacf files\1995\august\19950815.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\august\19950815.0800.00.han.fitacf.bz2"]
g = [r"C:\masters\fitacf files\1995\september\19950911.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\september\19950911.1400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\september\19950911.1600.00.han.fitacf.bz2"]
h = [r"C:\masters\fitacf files\1995\september\19950924.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\september\19950924.1400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\september\19950924.1600.00.han.fitacf.bz2"]
i = [r"C:\masters\fitacf files\1995\october\19951013.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951013.0800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951013.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951013.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951013.1400.00.han.fitacf.bz2"]
j = [r"C:\masters\fitacf files\1995\october\19951014.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951014.0800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951014.0900.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951014.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951014.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951014.1400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1995\october\19951014.1600.00.han.fitacf.bz2"]
k = [r"C:\masters\jan\19960103.0800.00.han.fitacf.bz2", r"C:\masters\jan\19960103.1000.00.han.fitacf.bz2", r"C:\masters\jan\19960103.1200.00.han.fitacf.bz2"]
l = [r"C:\masters\jan\19960118.0600.00.han.fitacf.bz2", r"C:\masters\jan\19960118.0800.00.han.fitacf.bz2", r"C:\masters\jan\19960118.1000.00.han.fitacf.bz2", r"C:\masters\jan\19960118.1200.00.han.fitacf.bz2"]
m = [r"C:\masters\jan\19960119.0600.00.han.fitacf.bz2", r"C:\masters\jan\19960119.0800.00.han.fitacf.bz2", r"C:\masters\jan\19960119.1000.00.han.fitacf.bz2", r"C:\masters\jan\19960119.1200.00.han.fitacf.bz2"]
n = [r"C:\masters\jan\19960125.0800.00.han.fitacf.bz2", r"C:\masters\jan\19960125.1000.00.han.fitacf.bz2", r"C:\masters\jan\19960125.1200.00.han.fitacf.bz2"]
o = [r"C:\masters\jan\19960126.0800.00.han.fitacf.bz2", r"C:\masters\jan\19960126.1000.00.han.fitacf.bz2"]
p = [r"C:\masters\jan\19960127.0800.00.han.fitacf.bz2", r"C:\masters\jan\19960127.1000.00.han.fitacf.bz2"]
q = [r"C:\masters\fitacf files\1996\19960208.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960208.0800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960208.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960208.1200.00.han.fitacf.bz2"]
r = [r"C:\masters\fitacf files\1996\19960319.1200.00.han.fitacf.bz2"]
s = [r"C:\masters\fitacf files\1996\19960403.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960403.0800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960403.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960403.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960403.1400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960403.1600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960403.1800.00.han.fitacf.bz2"]
t = [r"C:\masters\fitacf files\1996\19960404.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960404.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960404.1200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960404.1400.00.han.fitacf.bz2"]
u = [r"C:\masters\fitacf files\1996\19960405.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960405.0600.00.han.fitacf.bz2"]
v = [r"C:\masters\fitacf files\1996\19960407.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960407.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960407.0800.00.han.fitacf.bz2"]
w = [r"C:\masters\fitacf files\1996\19960408.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960408.0800.00.han.fitacf.bz2"]
x = [r"C:\masters\fitacf files\1996\19960512.0800.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960512.1000.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960512.1200.00.han.fitacf.bz2"]
y = [r"C:\masters\fitacf files\1996\19960514.0200.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960514.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960514.0600.00.han.fitacf.bz2"]
z = [r"C:\masters\fitacf files\1996\19960802.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960802.0800.00.han.fitacf.bz2"]
alpha = [r"C:\masters\fitacf files\1996\19960803.0800.00.han.fitacf.bz2"]
beta = [r"C:\masters\fitacf files\1996\19960804.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960804.0600.00.han.fitacf.bz2"]
gamma = [r"C:\masters\fitacf files\1996\19960806.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960806.0600.00.han.fitacf.bz2"]
delta = [r"C:\masters\fitacf files\1996\19960914.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960914.0600.00.han.fitacf.bz2"]
epsilon = [r"C:\masters\fitacf files\1996\19960924.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960924.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960924.0800.00.han.fitacf.bz2"]
zeta = [r"C:\masters\fitacf files\1996\19960925.0400.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960925.0600.00.han.fitacf.bz2", r"C:\masters\fitacf files\1996\19960925.0800.00.han.fitacf.bz2"]
'''
jan = r"C:\masters\jan"

class Event:
    yes = 1
    no = 0

data = []
files = [f for f in os.listdir(jan) if f.endswith('.bz2')]
if not files:
    print("No files found in the directory.")
print(f"Found {len(files)} files in the directory.")
    
for file in files:
        fitacf_file = os.path.join(jan, file)
        try:
            with bz2.open(fitacf_file) as fp:
                fitacf_stream = fp.read()
            reading = pydarn.SuperDARNRead(fitacf_stream, True)
            records = reading.read_fitacf()
            #data.append(records)
            for record in records:
                if 'slist' not in record or len(record['slist']) == 0:
                    continue
                record_time = dt.datetime(
                        record['time.yr'],
                        record['time.mo'],
                        record['time.dy'],
                        record['time.hr'],
                        record['time.mt'],
                        record['time.sc'],
                        int(record['time.us'] / 1000)  # microseconds --> milliseconds
                    )
                common_data = {
                    'time': record_time,
                    'bmnum': record.get('bmnum', np.nan),
                    'channel': record.get('channel', np.nan),
                    'cp': record.get('cp', np.nan),
                    'nrang': record.get('nrang'),
                    'frang': record.get('frang'),
                    'rsep': record.get('rsep'),
                    'stid': record['stid'],
                }
                slist = record['slist']
                for idx, gate in enumerate(slist):
                    gate_data = common_data.copy()
                    gate_data.update({
                        'range_gate': gate,
                        'p_l': record['p_l'][idx],
                        'v': record['v'][idx],
                        'w_l': record['w_l'][idx],
                        'gflg': record['gflg'][idx] if 'gflg' in record else np.nan
                    })
                    data.append(gate_data)
            del records, 
            gc.collect()
        except Exception as e:
            print(f"Error reading file {file}: {e}")
print(f"Read {len(data)} records from {len(files)} files.")
df = pd.DataFrame(data)
    
df.set_index(['time', 'range_gate'], inplace=True)
df.sort_index(inplace=True)

duplicates = df.index[df.index.duplicated()]
if not duplicates.empty:
    df = df[~df.index.duplicated(keep='first')]

    # fill missing
df['p_l'] = df['p_l'].fillna(-9999)
df['v']   = df['v'].fillna(-9999)
df['w_l'] = df['w_l'].fillna(-9999)

#df[['p_l', 'v']] = MMS().fit_transform(df[['p_l', 'v']].astype(float))

df['event'] = Event.no

df_filtered = df.loc[df['bmnum'] == 5]
df_filtered_actual = df_filtered[['p_l', 'v', 'bmnum', 'event']]
#print(df_filtered[['p_l', 'v', 'bmnum', 'event']])
df_filtered_actual.to_csv(r"C:\masters\machine learning\january.csv")

