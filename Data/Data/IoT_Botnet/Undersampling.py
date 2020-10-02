import pandas as pd
import csv
import types
df = pd.read_csv("/home/bench/notebooks/data/IoT_Botnet/UNSW_2018_IoT_Botnet_Dataset_1.csv",header = None)
df.columns = ["pkSeqID","stime","flgs","proto","saddr","sport","daddr","dport","pkts","bytes","state","ltime","seq","dur","mean","stddev","smac","dmac","sum","min","max","soui","doui","sco","dco","spkts","dpkts","sbytes","dbytes","rate","srate","drate","attack","category","subcategory"]
df.to_csv("/home/bench/notebooks/data/IoT_Botnet/Complete.csv")
i=2
while(i<74): 
    file_to_work = "/home/bench/notebooks/data/IoT_Botnet/UNSW_2018_IoT_Botnet_Dataset_" + str(i) +".csv"
    df_to_add = pd.read_csv(file_to_work,quoting=csv.QUOTE_NONE,header=None)
    df_to_add.columns = ["pkSeqID","stime","flgs","proto","saddr","sport","daddr","dport","pkts","bytes","state","ltime","seq","dur","mean","stddev","smac","dmac","sum","min","max","soui","doui","sco","dco","spkts","dpkts","sbytes","dbytes","rate","srate","drate","attack","category","subcategory"]
    
    df = pd.read_csv("/home/bench/notebooks/data/IoT_Botnet/Complete.csv")
    df = pd.concat([df,df_to_add])
    print(df.shape)
    df.to_csv("/home/bench/notebooks/data/IoT_Botnet/Complete.csv")
    i+=1

df = pd.read_csv("/home/bench/notebooks/data/IoT_Botnet/Complete.csv")

print("OUI !!!!")
df = df.fillna(0)

print("NaN handled")
for i in df.saddr:
    if(type(i) == type("")):
        values = i.strip('"').split(".")
        if(len(values) == 4):
            df["saddr_1"] = int(values[0])
            df["saddr_2"] = int(values[1])
            df["saddr_3"] = int(values[2])
            df["saddr_4"] = int(values[3])
        else:
            print(i)
            df["saddr_1"] = 0
            df["saddr_2"] = 1
            df["saddr_3"] = 2
            df["saddr_4"] = 3
    else:
        print(type(i))
        df["saddr_1"] = 0
        df["saddr_2"] = 1
        df["saddr_3"] = 2
        df["saddr_4"] = 3

print("start df with string")
df_with_string = df.copy()
for col in df_with_string.columns:
    if(df_with_string[col].dtype == "object"):
        dic_of_value={}
        i = 0
        for x in pd.unique(df_with_string[col]):
            if (type(x) == type('')):
                dic_of_value[x.strip(" ").strip("'")]=i
            else:
                dic_of_value[x]=i
            i+=1
        for key,value in dic_of_value.items():
            df_with_string[col][df_with_string[col]==key] = value

print("start df with float")
df_with_float = df_with_string.copy()
for col in df_with_float.columns:
    if(df_with_float[col].dtype == "float64" and col!='stime' and col != 'attack'):
        df_with_float[col] = df_with_float[col]*1000000
        df_with_float[col] = df_with_float[col].astype('int')

print("Finished and saving in complete_processed")
df_with_float.to_csv("Complete_processed.csv")
