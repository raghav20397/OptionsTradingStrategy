import pickle
from pickle_helper import *
import json

def save_as_json(filename,file):
    with open(filename, "w") as outfile:
        json.dump(file, outfile)
    return

def recur(d):
    if isinstance(d, dict):
        di = {}
        for i in d:
            di[str(i)] = recur(d[i])
        return(di)
    elif type(d) is list:
        di = []
        for i in d:
            di.append(str(i))
        return(di)
    else:
        return(str(d))

file_accr = "1"
data = loadp(file_accr+".pkl")
data_str = recur(data)
save_as_json(file_accr+".json",data_str)
