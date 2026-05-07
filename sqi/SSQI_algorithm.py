import numpy as np 
import pandas as pd 
import glob 

csv_files = glob.glob(r"D:\Study\Lab\fingerTests\*.csv")


#PPG_csv = pd.read_csv("D:\Study\Lab\day1test3640.csv")

#PPG_np = PPG_csv.to_numpy() 

#PPG_time = PPG_np[:,0]
#PPG_reading = PPG_np[:,1]

def Ssqi( reading_array):
    mu = reading_array.mean() 
    stddev = reading_array.std(ddof=0)
    return (np.mean(((reading_array - mu)/stddev) ** 3))

for file in csv_files:
    df = pd.read_csv(file)
    data_np = df.to_numpy()
    score = Ssqi(data_np[:,1])
    print(file, score)

