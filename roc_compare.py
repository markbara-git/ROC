import pandas as pd

#path to input csv
input = "input\\input-0-nr.csv"
output = "output\\"

#read CSV
df = pd.read_csv(input,usecols=['geopixel','device_total_security_score'])

#group by geopixel and calculate the arithmetic mean of the device_total_security_score
df = df.groupby(by='geopixel').mean().reset_index()

#save data to CSV
df.to_csv(output+"compare.csv",index=False)