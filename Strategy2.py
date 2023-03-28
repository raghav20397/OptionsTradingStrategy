#2023-01-03 02:35:28
import pickle as pkl
from datetime import datetime, timedelta
import pickle
import os
import math
from matplotlib import pyplot
import numpy as np
import copy
import json
from scipy import stats
import csv
from tqdm import tqdm
import subprocess

class Contract:
    vol = 1
    def __init__(self, price, vol, imp_v, delta, theta, vega, gamma, rho, time):
        self.price = price
        #can be multiple, will implement later
        self.vol = vol
        self.imp_v = imp_v
        self.delta = delta
        self.theta = theta
        self.vega = vega
        self.gamma = gamma
        self.rho = rho
        self.time = time

class spotpair():
    def __init__(self, time, spot):
        self.time = time
        self.spot = spot

class prempair():
    def __init__(self, time, premium):
        self.time = time
        self.premium = premium

class impvpair():
    def __init__(self,time,imp_v):
        self.time = time
        self.imp_v = imp_v

def load_json(path):
    with open(path,"rb") as json_file:
        data = json.load(json_file)
        return data

def addCon(con1, con2, timeGap):
    # adding all parameters of two contracts
    price = con1.price + con2.price
    vol = con1.vol + con2.vol
    imp_v = con1.imp_v + con2.imp_v
    delta = con1.delta + con2.delta
    theta = con1.theta +con2.theta 
    vega = con1.vega + con2.vega
    gamma = con1.gamma + con2.gamma
    rho = con1.rho + con2.rho
    time = con2.time +timedelta(seconds=timeGap)
    res = Contract(price, vol, imp_v, delta, theta, vega, gamma, rho, time)
    return res

def load_pkl(path):
    with open(path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
        return data

def getTimeList(year, month, date, hourTo, minuteTo, secondTo, hourFrom, minuteFrom, secondFrom, windowSize):
    timeList = []

    endTime = datetime(year, month, date, hourTo, minuteTo, secondTo)
    startTime = datetime(year, month, date, hourFrom, minuteFrom, secondFrom)

    def_endTime = datetime(year, month, date, 15, 30, 0)
    def_startTime = datetime(year, month, date, 9, 15, 0)

    lenBefore = 0
    lenAfter = 0
    startCpy = startTime
    endCpy = endTime

    while def_startTime < startCpy:  
        def_startTime += timedelta(seconds=1)
        lenBefore += 1

    while endCpy < def_endTime:  
        endCpy += timedelta(seconds=1)
        lenAfter+= 1

    startTime += timedelta(seconds=windowSize)

    while startTime <= endTime:  
        timeList.append(startTime)
        startTime += timedelta(seconds=1)
    # returning the seconds list for the interval for plotting purposes
    # and number of seconds before the time starts and after time ends for plotting purposes
    return timeList, lenBefore, lenAfter

def computeAverage(buffer, windowSize):
    Price = 0
    Vol = 0
    Imp_v = 0
    Delta = 0
    Theta = 0
    Vega = 0
    Gamma = 0
    Rho = 0  
    for contract in buffer:
        Price += contract.price/windowSize
        Vol += contract.vol/windowSize
        Imp_v += contract.imp_v/windowSize
        Delta += contract.delta/windowSize
        Theta += contract.theta/windowSize
        Vega += contract.vega/windowSize
        Gamma += contract.gamma/windowSize
        Rho += contract.rho/windowSize
    return Price, Vol, Imp_v, Delta, Theta, Vega, Gamma, Rho

def computeExpectedRegress(buffer, windowSize, timeGap):
    parameters = ["imp_v","price","vol", "delta", "theta", "vega", "gamma", "rho"]

    Price = [contract.price for contract in buffer]
    Vol = [contract.vol for contract in buffer]
    Imp_v = [contract.imp_v for contract in buffer]
    Delta = [contract.delta for contract in buffer]
    Theta = [contract.theta for contract in buffer]
    Vega = [contract.vega for contract in buffer]
    Gamma = [contract.gamma for contract in buffer]
    Rho = [contract.rho for contract in buffer]  
    window = windowSize-1

    t_list = []
    for i in range(1, windowSize+1):
        t_list.append(i)
        
    for parameter in parameters:
        if(parameter == "delta"):
            m1, c1, r1, p1, se1 = stats.linregress(t_list, Delta)
            # mymodel = np.poly1d(np.polyfit(t_list, Delta,3))

        if parameter == "theta":
            m2, c2, r2, p2, se2 = stats.linregress(t_list, Theta)

        if parameter == "price":
            m3, c3, r3, p3, se3 = stats.linregress(t_list, Price)

        if parameter == "imp_v":
            m4, c4, r4, p4, se4 = stats.linregress(t_list, Imp_v)

        if parameter == "vega":
            m5, c5, r5, p5, se5 = stats.linregress(t_list, Vega)

        if parameter == "gamma":
            m6, c6, r6, p6, se6 = stats.linregress(t_list, Gamma)
            
        if parameter == "rho":
            m7, c7, r7, p7, se7 = stats.linregress(t_list, Rho)

        if(parameter == "vol"):
            m8, c8, r8, p8, se8 = stats.linregress(t_list, Vol)

    # return Price*timeGap, Vol*timeGap, Imp_v*timeGap, Delta*timeGap, Theta*timeGap, Vega*timeGap, Gamma*timeGap, Rho*timeGap
    return m3*timeGap, m8*timeGap, m4*timeGap, m1*timeGap, m2*timeGap, m5*timeGap, m6*timeGap, m7*timeGap

def computeExpected(buffer, windowSize, timeGap):
    # changes in parameters
    Price = 0
    Vol = 0
    Imp_v = 0
    Delta = 0
    Theta = 0
    Vega = 0
    Gamma = 0
    Rho = 0  
    window = windowSize-1
    for i in range(1, len(buffer)):
        Price += (buffer[i].price - buffer[i-1].price)/window
        Vol += (buffer[i].vol - buffer[i-1].vol)/window
        Imp_v += (buffer[i].imp_v - buffer[i-1].imp_v)/window
        Delta += (buffer[i].delta - buffer[i-1].delta)/window
        Theta += (buffer[i].theta - buffer[i-1].theta)/window
        Vega += (buffer[i].vega - buffer[i-1].vega)/window
        Gamma += (buffer[i].gamma - buffer[i-1].gamma)/window
        Rho += (buffer[i].rho - buffer[i-1].rho)/window
    # print(Price)
    

    return Price*timeGap, Vol*timeGap, Imp_v*timeGap, Delta*timeGap, Theta*timeGap, Vega*timeGap, Gamma*timeGap, Rho*timeGap

def plotForParameter(parameter, actualValues, expectedValues, averageValues, timeList, targetStrike, windowSize, timeGap, date, month, year, lenBefore, lenAfter, estimation_type, type1):
    type1 = str(type1)
    stry = "actual"
    # to get lists of specific parameters from the contract objects for plotting purposes
    #Ac = Actual, Av = Average in window, Ex = Expected
    if(parameter == "delta"):
        Ac = [ac.delta for ac in actualValues]
        Ex = [ex.delta for ex in expectedValues]
        Av = [av.delta for av in averageValues]

    if parameter == "theta":
        Ac = [ac.theta for ac in actualValues]
        Ex = [ex.theta for ex in expectedValues]
        Av = [av.theta for av in averageValues]

    if parameter == "price":
        Ac = [ac.price for ac in actualValues]
        Ex = [ex.price for ex in expectedValues]
        Av = [av.price for av in averageValues]

    if parameter == "imp_v":
        Ac = [ac.imp_v for ac in actualValues]
        Ex = [ex.imp_v for ex in expectedValues]
        Av = [av.imp_v for av in averageValues]
        parameter = "Implied Volatility"

    if parameter == "vega":
        Ac = [ac.vega for ac in actualValues]
        Ex = [ex.vega for ex in expectedValues]
        Av = [av.vega for av in averageValues]

    if parameter == "gamma":
        Ac = [ac.gamma for ac in actualValues]
        Ex = [ex.gamma for ex in expectedValues]
        Av = [av.gamma for av in averageValues]
        
    if parameter == "rho":
        Ac = [ac.rho for ac in actualValues]
        Ex = [ex.rho for ex in expectedValues]
        Av = [av.rho for av in averageValues]
    
    #change path here
    location = f"D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}\\"
    # location = f"X:\\NXBLOCK\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}\\"
    
    # print(location)
    j=0
    try:
        if(os.path.exists(location)):
            pass
        else:
            os.mkdir(location)

    except OSError as error:
        j+=1
        # print(error)
    # location = f"X:\\NXBLOCK\\Reports\\{targetStrike}_{windowSize}_{timeGap}_{year}_{month}_{date}_{str}\\"

    # print(lenBefore, lenAfter, len(timeList), (lenBefore+lenAfter+len(timeList)-len(Ac))==0 )
    pyplot.plot(timeList[timeGap+5:], Ac[lenBefore+timeGap+5: lenBefore + len(timeList)], linestyle = "dashed", label = f"Actual {parameter} Values")
    pyplot.plot(timeList[timeGap+5:], Av[lenBefore+timeGap+5: lenBefore + len(timeList)], label = f"Average {parameter} Values")
    pyplot.plot(timeList[timeGap+5:], Ex[lenBefore+timeGap+5: lenBefore + len(timeList)], label = f"Expected {parameter} Values")
    pyplot.legend()
    pyplot.title(f"Correlation between Actual, Avg and Expected {parameter} values for {year}-{month}-{date}, Strike = {targetStrike}")
    pyplot.xlabel(f"""Time (MM-DD HH)
    {type1}_{estimation_type}""")
    
    if not os.path.exists(location):
        os.makedirs(location)
    pyplot.savefig(location + f"{parameter} ComparisionChart", bbox_inches="tight")
    pyplot.close()

def plotForParameterError(parameter, actualValues, expectedValues, averageValues, timeList, targetStrike, windowSize, timeGap, date, month, year, lenBefore, lenAfter, estimation_type, type1):
    type1 = str(type1)
    stry = "error"
    #Ac = Actual, Av = Average in window, Ex = Expected
    # to get lists of specific parameters from the contract objects for plotting purposes
    if(parameter == "delta"):
        Ac = [ac.delta for ac in actualValues]
        Ex = [ex.delta for ex in expectedValues]
        Av = [av.delta for av in averageValues]

    if parameter == "theta":
        Ac = [ac.theta for ac in actualValues]
        Ex = [ex.theta for ex in expectedValues]
        Av = [av.theta for av in averageValues]

    if parameter == "price":
        Ac = [ac.price for ac in actualValues]
        Ex = [ex.price for ex in expectedValues]
        Av = [av.price for av in averageValues]

    if parameter == "imp_v":
        Ac = [ac.imp_v for ac in actualValues]
        Ex = [ex.imp_v for ex in expectedValues]
        Av = [av.imp_v for av in averageValues]
        parameter = "Implied Volatility"


    if parameter == "vega":
        Ac = [ac.vega for ac in actualValues]
        Ex = [ex.vega for ex in expectedValues]
        Av = [av.vega for av in averageValues]

    if parameter == "gamma":
        Ac = [ac.gamma for ac in actualValues]
        Ex = [ex.gamma for ex in expectedValues]
        Av = [av.gamma for av in averageValues]
        
    if parameter == "rho":
        Ac = [ac.rho for ac in actualValues]
        Ex = [ex.rho for ex in expectedValues]
        Av = [av.rho for av in averageValues]
        
    #change path here

    # location = f"X:\\NXBLOCK\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}\\"
    location = f"D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}\\"
    j=0
    try:    
        if(os.path.exists(location)):
            pass
        else:
            os.mkdir(location)

    except OSError as error:
        j+=1
        # print(error)
    
    #calculating the percentage change from actual and expected value at each second 
    plotted = np.subtract(np.array(Ex), np.array(Ac))
    plotted = [plotted[i]/Ac[i] for i in range(len(plotted))]
    plotted = np.array(plotted)
    plotted = plotted*100
    sumAbs = 0
    for i in plotted[lenBefore: lenBefore + len(timeList)]:
        sumAbs += abs(i)

    mean_er = sum(plotted[lenBefore: lenBefore + len(timeList)]) / len(plotted[lenBefore: lenBefore + len(timeList)])
    mean_abs_error = sumAbs / len(plotted[lenBefore: lenBefore + len(timeList)])

    pyplot.plot(timeList[5:], plotted[lenBefore+5: lenBefore + len(timeList)] , label = f"% change in expected {parameter} ValueValues after {timeGap} secondss")
    # pyplot.plot(timeList[5:], Ac[lenBefore+5: lenBefore + len(timeList)] ,linestyle = 'dotted' ,label = f"Actual {parameter} Values")
    # pyplot.plot(timeList, Ex, label = f"Expected {parameter} Values")
    pyplot.ylim([-7.5, 7.5])
    # pyplot.ylim([min(plotted[lenBefore: lenBefore + len(timeList)]), max(plotted[lenBefore: lenBefore + len(timeList)])])
    pyplot.title(f"""Mean error % for {parameter} is {round(mean_er, 5)} %
    Mean absoute error % for {parameter} is {round(mean_abs_error,5)} %""")
    pyplot.legend()
    pyplot.xlabel(f"""Time (MM-DD HH)
    For {year}-{month}-{date}, Strike: {targetStrike}
    {type1}_{estimation_type}""")

    # print(f"Mean error % for {parameter} is {mean_er} %, max error % is {max(plotted[lenBefore: lenBefore + len(timeList)])}")
    # print(f"Mean absoute error % for {parameter} is {mean_abs_error} %")
    # print(f"Mean absoute error % for {parameter} is {mean_abs_error} %, max error % is {max(abs(plotted[lenBefore+timeGap: lenBefore + len(timeList)]))}")
    
    if not os.path.exists(location):
        os.makedirs(location)
    pyplot.savefig(location + f"{parameter} ChangeChart", bbox_inches="tight")
    pyplot.close()

def computePlotActual(actualValues, expectedValues, averageValues, windowSize, timeGap, targetStrike,hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, date, month,year, estimation_type, type):
    # parameters = ["price", "imp_v", "delta", "theta", "vega", "gamma", "rho"]
    # parameters = ["delta"]
    parameters = ["imp_v", "delta", "theta", "vega", "gamma", "rho"]

    # print(len(actualValues))
    # print(len(averageValues))
    # print(len(expectedValues))
    # print(len(actualValues))
    timeList, lenBefore, lenAfter = getTimeList(year, month, date, hourTo, minuteTo, secondTo, hourFrom, minuteFrom, secondFrom, windowSize)

    for param in parameters:
        plotForParameter(param, actualValues, expectedValues, averageValues, timeList, targetStrike, windowSize, timeGap, date, month, year, lenBefore, lenAfter, estimation_type, type)

def computePlotError(actualValues, expectedValues, averageValues, windowSize, timeGap, targetStrike,hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, date, month, year, estimation_type, type):
    # parameters = ["price", "imp_v", "delta", "theta", "vega", "gamma", "rho"]
    # parameters = ["delta"]
    parameters = ["imp_v", "delta", "theta", "vega", "gamma", "rho"]
    timeList, lenBefore, lenAfter = getTimeList(year, month, date, hourTo, minuteTo, secondTo, hourFrom, minuteFrom, secondFrom, windowSize)
    
    for param in parameters:
        plotForParameterError(param, actualValues, expectedValues, averageValues, timeList, targetStrike, windowSize, timeGap, date, month, year, lenBefore, lenAfter, estimation_type, type)

def AverageChangeRegress(buffer, windowSize):
    t_list = []
    for i in range(1, windowSize+1):
        t_list.append(i)

    spots = [pair.spot for pair in buffer]

    m_spot, c, r, p, se = stats.linregress(t_list, spots)

    return m_spot

def AverageChange(buffer, windowSize):
    sum = 0
    for i in range(1,len(buffer)):
        sum+=(buffer[i].spot - buffer[i-1].spot)/windowSize
    return sum

def computeSpotsActual(year,month,day,spotData):
    myDate = datetime(year,month,day)
    # print("Spot JSON loading..")
    # data = load_json(path)
    # print("Spot JSON loaded ")
    data = spotData

    datetime_start = datetime(myDate.year,myDate.month,myDate.day,9,15,0)
    spotArr = []
    startiter = datetime_start
    datetime_end = datetime(myDate.year,myDate.month,myDate.day,15,30,1)
    while(startiter < datetime_end):
        try:
            spotArr.append(spotpair(startiter, float(data[str(startiter)])))
            # print(float(data[str(startiter)]))
            # startiter+=timedelta(seconds=1)
        except KeyError :
            spotArr.append(spotpair(startiter, spotArr[len(spotArr)-1].spot))

        startiter+=timedelta(seconds=1)

    return spotArr

# def computeExpectedSpotChanges(year,month,day, data,windowSize,timeGap, type):
#     SpotPrices = computeSpotsActual(year,month,day,data)
#     PredictedSpotChanges = []
#     datetime_start = datetime(year,month,day,9,15,0)
#     datetime_end = datetime(year,month,day,15,30,1)
#     iter = datetime_start
#     j=0
#     buffer = []
#     # Initialization of buffer
#     while(iter < datetime(year,month,day,9,15,windowSize)):
#         buffer.append(SpotPrices[j])
#         # print(iter)
#         iter+=timedelta(seconds=1)
#         j+=1
#     iter = iter - timedelta(seconds=1)
#     # print(".",iter, timeGap)
#     next_iter = iter + timedelta(seconds=timeGap)


#     while(next_iter < datetime_end):
#         if(type=="simple"):
#             PredictedSpotChange = timeGap*AverageChange(buffer, windowSize)

#         if(type=="regress"):
#             PredictedSpotChange = timeGap*AverageChangeRegress(buffer, windowSize)
#         # print(PredictedSpotChange)
#         PredictedSpotChanges.append(spotpair(next_iter,PredictedSpotChange))
#         iter+=timedelta(seconds=1)
#         next_iter+=timedelta(seconds=1)
#         buffer.append(SpotPrices[j])
#         del buffer[0]
#         j+=1
#     return PredictedSpotChanges, SpotPrices

def computeExpectedSpotChanges(year,month,day, data,windowSize,timeGap, type, smoothingFactor):
    SpotPrices = computeSpotsActual(year,month,day,data)
    PredictedSpotChanges = []
    spots = []
    datetime_start = datetime(year,month,day,9,15,0)
    datetime_end = datetime(year,month,day,15,30,1)
    iter = datetime_start
    j=0
    buffer = []
    # Initialization of buffer
    while(iter < datetime(year,month,day,9,15,windowSize)):
        buffer.append(SpotPrices[j])
        spots.append(SpotPrices[j].spot)
        # print(iter)
        iter+=timedelta(seconds=1)
        j+=1
    iter = iter - timedelta(seconds=1)
    # print(".",iter, timeGap)
    next_iter = iter + timedelta(seconds=timeGap)


    while(next_iter < datetime_end):
        if(type=="simple"):
            PredictedSpotChange = timeGap*AverageChange(buffer, windowSize)

        if(type=="regress"):
            PredictedSpotChange = timeGap*AverageChangeRegress(buffer, windowSize)
        
        if(type=="ema"):
            PredictedSpotChange = EMA(spots,len(buffer),smoothingFactor,timeGap)
        # print(PredictedSpotChange)
        PredictedSpotChanges.append(spotpair(next_iter,PredictedSpotChange))
        iter+=timedelta(seconds=1)
        next_iter+=timedelta(seconds=1)
        buffer.append(SpotPrices[j])
        spots.append(SpotPrices[j].spot)
        del buffer[0]
        del spots[0]
        j+=1

    return PredictedSpotChanges, SpotPrices

def plotPremiumActual(exptectedPremiums, actualValues,timeList, targetStrike, windowSize,timeGap, year, month, date, lenBefore, estimation_type,type1):
    type1 = str(type1)
    stry="actual"
    parameter="Premium"
    location = f"X:\\NXBLOCK\\OptionsTradingStrategy\\Reports\\{targetStrike}_{windowSize}_{timeGap}_{year}_{month}_{date}_{str}\\"
    # location = f"X:\\NXBLOCK\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}\\"

    # location = f"D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}"
    j=0
    try:
        if(os.path.exists(location)):
            pass
        else:
            os.mkdir(location)

    except OSError as error:
        j+=1
        # print(error)
    #calculating the percentage change from actual and expected value at each second 


    Ac = [ac.price for ac in actualValues]
    Ex = [ex.premium for ex in exptectedPremiums]

    pyplot.plot(timeList[timeGap+5:], Ac[lenBefore+timeGap+5: lenBefore + len(timeList)], linestyle = "dashed", label = f"Actual {parameter} Values")
    pyplot.plot(timeList[timeGap+5:], Ex[lenBefore+timeGap+5: lenBefore + len(timeList)], label = f"Expected {parameter} Values after {timeGap} seconds")
    pyplot.legend()
    pyplot.title(f"Correlation between Actual and Expected {parameter} values for {year}-{month}-{date}, Strike = {targetStrike}")
    pyplot.xlabel(f"""Time (MM-DD HH)
    {type1}_{estimation_type}""")
    
    if not os.path.exists(location):
        os.makedirs(location)
    pyplot.savefig(location + f"{parameter} ComparisionChart", bbox_inches="tight")
    pyplot.close()

def plotValFit(windowSize,TimeGap,data,year,month,date,targetStrike):
    SpotPrices = computeSpotsActual(year,month,date,data)
    # PredictedSpotChanges = []
    endstr = "plotsforCheckingFit"
    
    datetime_start = datetime(year,month,date,9,15,0)
    datetime_end = datetime(year,month,date,15,30,1)
    iter = datetime_start
    j=0
    buffer = []
    # Initialization of buffer
    while(iter < datetime(year,month,date,9,15,windowSize+TimeGap)):
        buffer.append(SpotPrices[j].spot)
        # print(iter)
        iter+=timedelta(seconds=1)
        j+=1
    iter = iter - timedelta(seconds=1)
    # print(".",iter, timeGap)
    next_iter = iter + timedelta(seconds=TimeGap)
    numPlots = 0
    k=0
    while(next_iter < datetime_end and numPlots <= 10):
        k+=1
        if(k%1800==0):
            location = f"D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{TimeGap}\\{endstr}"
            # location = f"X:\\NXBLOCK\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{endstr}\\"

            X_Vals1 = np.arange(0,windowSize)
            X_Vals2 = np.arange(0,windowSize + TimeGap)

            Y_Vals1 = np.array(buffer[0:windowSize])
            m, c, r, p, se = stats.linregress(X_Vals1,Y_Vals1)

            pyplot.xlim(0,windowSize + 2*TimeGap) 
            # pyplot.ylim(-1000,1000)

            pyplot.plot(X_Vals2,np.array(buffer),color="red", label= "Actual Spot vals")
            # pyplot.plot([j+timeGap],SpotPrices[j+TimeGap].spot,color="red")
            pyplot.plot(X_Vals2,m*X_Vals2+buffer[0],color="blue",linestyle="dashed", label= "best fit expected line")
            # pyplot.plot([j+timeGap],m*TimeGap+buffer[len(buffer)-1],color="green", label= "expected spot")

            pyplot.legend()

            nameofPlot = str(k)+"_" + str(numPlots) + ".jpg"
            if not os.path.exists(location):
                os.mkdir(location)
            
            pyplot.savefig(location + nameofPlot)
            pyplot.close()
            numPlots+=1
        del buffer[0]
        buffer.append(SpotPrices[j].spot)
        j+=1
        next_iter+=timedelta(seconds=1)
    
def EMA(arr,windowSize,b_Factor,timeGap):

    coefficient_arr = [math.exp(i*b_Factor) for i in range(1, timeGap+1)]

    avg_buffer = []
# if(windowSize - timeGap - timeGap >=1):
    for i in range(0, windowSize - timeGap):
        average_change=0
        for j in range(0,timeGap):
            average_change += (arr[i+j+1]-arr[i+j])*coefficient_arr[j]
            average_change /= sum(coefficient_arr)

        avg_buffer.append(average_change)
    # try:
    fin = sum(avg_buffer)/len(avg_buffer)
    # else:
    #     print("prev window size is big")
    #     fin=0
    # except:
    #     fin = 0
    return fin

def EMA_pr(arr,windowSize,smoothingFactor,numSeconds):

    beta = smoothingFactor/(windowSize+1)

    ans = 0

    starting_index = 0
    ending_index = numSeconds

    try:
        prev_ema = arr[ending_index-1] - arr[starting_index]
    except:
        prev_ema = arr[len(arr)-1] - arr[0]

    for i in range(0,int(windowSize/numSeconds)):

        # curr_val = averageofArray(arr[starting_index:ending_index])
        curr_val = arr[ending_index-1] - arr[starting_index]
        ans+=beta*curr_val+(1-beta)*prev_ema
        prev_ema = ans
        starting_index = ending_index
        ending_index += numSeconds
    
    return ans
        
def computeEMA(buffer,windowSize,smoothingFactor,numSeconds):

    PriceBuffer = []
    VolBuffer = []
    Imp_VBuffer = []
    DeltaBuffer = []
    ThetaBuffer = []
    VegaBuffer = []
    GammaBuffer = []
    RhoBuffer = []

    for i in range(windowSize):
        PriceBuffer.append(buffer[i].price)
        VolBuffer.append(buffer[i].vol)
        Imp_VBuffer.append(buffer[i].imp_v)
        DeltaBuffer.append(buffer[i].delta)
        ThetaBuffer.append(buffer[i].theta)
        VegaBuffer.append(buffer[i].vega)
        GammaBuffer.append(buffer[i].gamma)
        RhoBuffer.append(buffer[i].rho)
    
    price = PriceBuffer[len(PriceBuffer)-1] + EMA(PriceBuffer,len(PriceBuffer),smoothingFactor,numSeconds)
    vol = VolBuffer[len(VolBuffer)-1] + EMA(VolBuffer,len(VolBuffer),smoothingFactor,numSeconds)
    imp_v = Imp_VBuffer[len(Imp_VBuffer)-1] + EMA(Imp_VBuffer,len(Imp_VBuffer),smoothingFactor,numSeconds)
    delta = DeltaBuffer[len(DeltaBuffer)-1] + EMA(DeltaBuffer,len(DeltaBuffer),smoothingFactor,numSeconds)
    delta /= 2
    theta = ThetaBuffer[len(ThetaBuffer)-1] + EMA(ThetaBuffer,len(ThetaBuffer),smoothingFactor,numSeconds)
    vega = VegaBuffer[len(VegaBuffer)-1] + EMA(VegaBuffer,len(VegaBuffer),smoothingFactor,numSeconds)
    gamma = GammaBuffer[len(GammaBuffer)-1] + EMA(GammaBuffer,len(GammaBuffer),smoothingFactor,numSeconds)
    rho = RhoBuffer[len(RhoBuffer)-1] + EMA(RhoBuffer,len(RhoBuffer),smoothingFactor,numSeconds)

    return price,vol,imp_v,delta,theta,vega,gamma,rho

def plotPremiumError(exptectedPremiums, actualValues, timeList, targetStrike, windowSize,timeGap, year, month, date, lenBefore, estimation_type,type1):
    type1 = str(type1)
    stry="error"
    parameter="Premium"
    #calculating the percentage change from actual and expected value at each second 
    location = f"D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}"
    # location = f"X:\\NXBLOCK\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\{stry}\\{type1}_{estimation_type}\\"
    
    j=0
    try:
        if(os.path.exists(location)):
            pass
        else:
            os.mkdir(location)

    except OSError as error:
        j+=1

    Ac = [ac.price for ac in actualValues]
    Ex = [ex.premium for ex in exptectedPremiums]


    plotted = np.subtract(np.array(Ex), np.array(Ac))
    plotted = [plotted[i]/Ac[i] for i in range(len(plotted))]
    plotted = np.array(plotted)
    plotted = plotted*100
    sumAbs = 0
    for i in plotted[lenBefore: lenBefore + len(timeList)]:
        sumAbs += abs(i)

    mean_er = sum(plotted[lenBefore: lenBefore + len(timeList)]) / len(plotted[lenBefore: lenBefore + len(timeList)])
    mean_abs_error = sumAbs / len(plotted[lenBefore: lenBefore + len(timeList)])

    pyplot.plot(timeList[5:], plotted[lenBefore+5: lenBefore + len(timeList)] , label = f"% error in expected {parameter} Values after {timeGap} seconds")
    # pyplot.plot(timeList[5:], Ac[lenBefore+5: lenBefore + len(timeList)] ,linestyle = 'dotted' ,label = f"Actual {parameter} Values")
    # pyplot.plot(timeList, Ex, label = f"Expected {parameter} Values")
    pyplot.ylim([-7.5, 7.5])
    # pyplot.ylim([min(plotted[lenBefore: lenBefore + len(timeList)]), max(plotted[lenBefore: lenBefore + len(timeList)])])
    pyplot.title(f"""Mean error % for {parameter} is {round(mean_er, 5)} %
    Mean absoute error % for {parameter} is {round(mean_abs_error,5)} %""")
    pyplot.legend()
    pyplot.xlabel(f"""Time (MM-DD HH)
    For {year}-{month}-{date}, Strike: {targetStrike}
    {type1}_{estimation_type}""")
    # print(f"Mean error % for {parameter} is {mean_er} %, max error % is {max(plotted[lenBefore: lenBefore + len(timeList)])}")
    # print(f"Mean absoute error % for {parameter} is {mean_abs_error} %")
    # print(f"Mean absoute error % for {parameter} is {mean_abs_error} %, max error % is {max(abs(plotted[lenBefore+timeGap: lenBefore + len(timeList)]))}")
    
    if not os.path.exists(location):
        os.makedirs(location)
    pyplot.savefig(location + f"{parameter} ErrorChart", bbox_inches="tight")
    pyplot.close()

def ccomputePremiumExpectedDelta(actualValues, expectedValues, expectedSpotChanges, windowSize, timeGap, targetStrike,hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, date, month, year, estimation_type,type="now"):
    expectedPremiums = []

    timeList, lenBefore, lenAfter = getTimeList(year, month, date, hourTo, minuteTo, secondTo, hourFrom, minuteFrom, secondFrom, windowSize)
    
    # print(actualValues[0].time,expectedValues[timeGap].time,expectedSpotChanges[timeGap].time)
    for i in range(len(actualValues)):
        if(i<timeGap):
            expectedPremiums.append(prempair(actualValues[i].time, 0))
        else:
            #   MX  +  C 
            if(type=="predict"):
            # delta for t+5
                expectedPremiums.append(prempair(expectedSpotChanges[i].time, actualValues[i-timeGap].price +  expectedValues[i].delta*(expectedSpotChanges[i].spot)))

            if(type=="now" and i<len(expectedSpotChanges)):
            # delta for t
                try:
                    expectedPremiums.append(prempair(expectedSpotChanges[i].time, actualValues[i-timeGap].price +  actualValues[i-timeGap].delta*expectedSpotChanges[i].spot))
                except:
                    print(i, len(expectedSpotChanges))
    # print(len(expectedSpotChanges),len(actualValues),len(expectedPremiums))
    # print(expectedSpotChanges[0].time,actualValues[0].time,expectedPremiums[0].time)
    # print(actualValues[-6].price, expectedValues[-1].delta, spotsExpected[-1].spot, expectedPremiums[-1].premium)
    # plotPremiumError(expectedPremiums,actualValues, timeList, targetStrike, windowSize, timeGap, year, month, date, lenBefore, estimation_type, type)
    # plotPremiumActual(expectedPremiums,actualValues, timeList, targetStrike, windowSize, timeGap, year, month, date, lenBefore, estimation_type, type)
    return expectedPremiums

#expected premiums for vega 
def computePremiumExpectedVega(actualValues, expectedValues, expectedImpVChanges, windowSize, timeGap, targetStrike,hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, date, month, year, estimation_type,type="now"):
    expectedPremiums = []

    timeList, lenBefore, lenAfter = getTimeList(year, month, date, hourTo, minuteTo, secondTo, hourFrom, minuteFrom, secondFrom, windowSize)
    
    # print(actualValues[0].time,expectedValues[timeGap].time,expectedSpotChanges[timeGap].time)
    for i in range(len(actualValues)-timeGap):
        if(i<timeGap):
            expectedPremiums.append(prempair(actualValues[i].time, 0))
        else:
            #   MX  +  C 
            if(type=="predict"):
            # delta for t+5
                expectedPremiums.append(prempair(expectedImpVChanges[i].time, actualValues[i-timeGap].price +  expectedValues[i].vega*(expectedImpVChanges[i].imp_v)))

            if(type=="now"):
            # delta for t
                # try:
                expectedPremiums.append(prempair(expectedImpVChanges[i].time, actualValues[i-timeGap].price +  actualValues[i-timeGap].vega*expectedImpVChanges[i].imp_v))
                # except:
                    # print(i, len(expectedImpVChanges), len(actualValues), i-timeGap)
    # print(actualValues[-6].price, expectedValues[-1].delta, spotsExpected[-1].spot, expectedPremiums[-1].premium)
    # plotPremiumError(expectedPremiums,actualValues, timeList, targetStrike, windowSize, timeGap, year, month, date, lenBefore, estimation_type, type)
    # plotPremiumActual(expectedPremiums,actualValues, timeList, targetStrike, windoCSpots, expectedSpots, actualPremiums, expectedValues,timeGap, windowSize, optionType ,brockerage, targetStrike, date, month, year, hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, greek_type, estimation_type, smoothingFactor):

    # timeList, len
    # print(len(expectedImpVChanges),len(actualValues),len(expectedPremiums))
    # print(expectedImpVChanges[0].time,actualValues[0].time,expectedPremiums[0].time)
    return expectedPremiums

def computeExpectedImpvChanges(actualImpvs,windowSize,timeGap,year,month,day,type):
    # SpotPrices = computeSpotsActual(year,month,day,data)
    PredictedImpVChanges = []
    impvs = []
    datetime_start = datetime(year,month,day,9,15,0)
    datetime_end = datetime(year,month,day,15,30,1)
    iter = datetime_start
    j=0
    buffer = []
    # Initialization of buffer
    while(iter < datetime(year,month,day,9,15,windowSize)):
        buffer.append(actualImpvs[j])
        impvs.append(actualImpvs[j].imp_v)
        # print(iter)
        iter+=timedelta(seconds=1)
        j+=1
    iter = iter - timedelta(seconds=1)
    # print(".",iter, timeGap)
    next_iter = iter + timedelta(seconds=timeGap)

    PredictedImpVChange = 0
    while(next_iter < datetime_end):
        if(type=="simple"):
            PredictedImpVChange = timeGap*AverageChange(buffer, windowSize)

        if(type=="regress"):
            PredictedImpVChange = timeGap*AverageChangeRegress(buffer, windowSize)
        
        if(type=="ema"):
            PredictedImpVChange = EMA(impvs,len(buffer),smoothingFactor,timeGap)
        # print(PredictedSpotChange)
        PredictedImpVChanges.append(impvpair(next_iter,PredictedImpVChange))
        iter+=timedelta(seconds=1)
        next_iter+=timedelta(seconds=1)
        buffer.append(actualImpvs[j])
        impvs.append(actualImpvs[j].imp_v)
        del buffer[0]
        del impvs[0]
        j+=1

    return PredictedImpVChanges


def ProfitorLossforaDay(grk, pathtocreate,expectedPremiums, actualImpVs, expectedImpVs, actualSpots, expectedSpots, actualPremiums, expectedValues,timeGap, windowSize, optionType ,brockerage, targetStrike, date, month, year, hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, greek_type, estimation_type, smoothingFactor):

    timeList, lenBefore, lenAfter = getTimeList(year, month, date, hourTo, minuteTo, secondTo, hourFrom, minuteFrom , secondFrom, windowSize)
    
    # location = f"D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\Report_{greek_type}_{estimation_type}.csv"
    # location = f"X:\\NXBLOCK\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{targetStrike}\\{windowSize}_{timeGap}\\Report_{greek_type}_{estimation_type}_{smoothingFactor}_{optionType}.csv"
    
    os.chdir(pathtocreate)
    location = pathtocreate +"\\"+str(year)
    if(os.path.exists(location)==False):
        os.mkdir(location)
    os.chdir(location)

    location += "\\"+str(month)
    if(os.path.exists(location)==False):
        os.mkdir(location)
    os.chdir(location)

    location += "\\"+str(date)
    if(os.path.exists(location)==False):
        os.mkdir(location)

    location += "\\"+str(targetStrike)
    if(os.path.exists(location)==False):
        os.mkdir(location)

    location += "\\" + f"{windowSize}_{timeGap}" +"\\"
    if(os.path.exists(location)==False):
        os.mkdir(location)

    # if "Reports" not in os.listdir():
    #     location += "\\Reports"
    #     os.mkdir(location)

    # os.chdir(pathtocreate+"\\"+str(year)+"\\"+str(month)+"\\"+str(date)+"\\Reports")
    os.chdir(location)
    # location = pathtocreate+"\\"+str(year)+"\\"+str(month)+"\\"+str(date)+"\\"

    if(grk == "vega"):
        location = f"Report_{greek_type}_{estimation_type}_{smoothingFactor}_{optionType}_vega.csv"
    if(grk == "delta"):
        location = f"Report_{greek_type}_{estimation_type}_{smoothingFactor}_{optionType}_delta.csv"

    time_iter = lenBefore + timeGap
    end_iter = lenAfter
    pnl = 0
    rows= []
    if(grk == "vega"):
        rows.append(["time t","actual price at t",f"expected price at t + {timeGap}",f"actual price at t + {timeGap}","vega at t", f"actual vega at t + {timeGap}","imp_v at t",f"actual imp_v at t + {timeGap}",f"time t + {timeGap}", "trade pnl"])
    
    if(grk == "delta"):
        rows.append(["time t","actual price at t",f"expected price at t + {timeGap}",f"actual price at t + {timeGap}","delta at t", f"actual delta at t + {timeGap}","spot at t",f"actual spot at t + {timeGap}",f"time t + {timeGap}", "trade pnl"])

    # rows.append(["time t","actual price at t",f"expected price at t + {timeGap}",f"actual price at t + {timeGap}","delta at t",f"expected delta at t+{timeGap}", f"actual delta at t + {timeGap}","spot at t",f"expected spot at t+{timeGap}",f"actual spot at t + {timeGap}",f"time t + {timeGap}", "trade pnl"])
    
    p,q,r,s = 0,0,0,0
    while time_iter < lenBefore + len(timeList)-timeGap:
        indiv_pnl = 0
        # print(len(expectedPremiums), time_iter+timeGap)
        if time_iter+timeGap< len(expectedPremiums):
            if( (abs(expectedPremiums[time_iter+timeGap].premium - actualPremiums[time_iter].price) / actualPremiums[time_iter].price) * 100  >100*0.02):
            # if( ((abs(expectedPremiums[time_iter+timeGap].premium - actualPremiums[time_iter].price) / actualPremiums[time_iter].price) * 100  >100*brockerage) and (0.7>=actualPremiums[time_iter + timeGap].delta >= 0.3 or -0.7<=actualPremiums[time_iter + timeGap].delta <= -0.3)):
                # expected pnl>0 ,  actual pnl>0, brock -
                if expectedPremiums[time_iter+timeGap].premium-actualPremiums[time_iter].price >= 0 and actualPremiums[time_iter+timeGap].price  - actualPremiums[time_iter].price  >= 0 :
                    indiv_pnl = (actualPremiums[time_iter+timeGap].price-actualPremiums[time_iter].price - brockerage*actualPremiums[time_iter].price)
                    # indiv_pnl = expectedSpots[time_iter + timeGap]-(actualPremiums[time_iter].strike)-actualPremiums[time_iter].price - brockerage*actualPremiums[time_iter].price)

                    pnl += indiv_pnl
                    p +=1
                
                # expected pnl>0 ,  actual pnl<0, brock+
                elif expectedPremiums[time_iter+timeGap].premium-actualPremiums[time_iter].price  >= 0 and actualPremiums[time_iter+timeGap].price  - actualPremiums[time_iter].price  <= 0:
                    indiv_pnl = -1*(actualPremiums[time_iter].price - actualPremiums[time_iter+timeGap].price + brockerage*actualPremiums[time_iter].price)
                    pnl += indiv_pnl
                    q+=1
                
                # expected pnl<0 ,  actual pnl<0, brock -
                elif expectedPremiums[time_iter+timeGap].premium - actualPremiums[time_iter].price  <= 0 and actualPremiums[time_iter+timeGap].price  - actualPremiums[time_iter].price  <= 0:
                    indiv_pnl = (actualPremiums[time_iter].price-actualPremiums[time_iter+timeGap].price - brockerage*actualPremiums[time_iter].price)
                    pnl += indiv_pnl
                    r+=1

                # expected pnl<0 ,  actual pnl>0. brock+
                # predicted decrease, actually increases
                elif expectedPremiums[time_iter+timeGap].premium - actualPremiums[time_iter].price  <= 0 and actualPremiums[time_iter+timeGap].price - actualPremiums[time_iter].price  >= 0:
                    indiv_pnl =-1*(actualPremiums[time_iter+timeGap].price-actualPremiums[time_iter].price + brockerage*actualPremiums[time_iter].price)
                    pnl += indiv_pnl
                    s+=1
                
                # time t, actual price at t, delta at t, expected price at t + timeGap, actual at t + timeGap    
                # rows.append([actualPremiums[time_iter].time.strftime("%d/%m/%Y, %H:%M:%S"), actualPremiums[time_iter].price,    expectedPremiums[time_iter+timeGap].premium,   actualPremiums[time_iter+timeGap].price,     actualPremiums[time_iter].delta ,      expectedValues[time_iter+timeGap].delta ,    actualPremiums[time_iter+timeGap].delta,     actualSpots[time_iter].spot ,  expectedSpots[time_iter+timeGap].spot,  actualSpots[time_iter+timeGap].spot,    expectedPremiums[time_iter+timeGap].time.strftime("%d/%m/%Y, %H:%M:%S"), indiv_pnl])
                if(grk == "vega"):
                    rows.append([actualPremiums[time_iter].time.strftime("%d/%m/%Y, %H:%M:%S"), actualPremiums[time_iter].price,    expectedPremiums[time_iter+timeGap].premium,   actualPremiums[time_iter+timeGap].price,     actualPremiums[time_iter].vega ,       actualPremiums[time_iter+timeGap].vega,     actualImpVs[time_iter].imp_v ,   actualImpVs[time_iter+timeGap].imp_v,    expectedPremiums[time_iter+timeGap].time.strftime("%d/%m/%Y, %H:%M:%S"), indiv_pnl])
                if(grk == "delta"):
                    rows.append([actualPremiums[time_iter].time.strftime("%d/%m/%Y, %H:%M:%S"), actualPremiums[time_iter].price,    expectedPremiums[time_iter+timeGap].premium,   actualPremiums[time_iter+timeGap].price,     actualPremiums[time_iter].delta ,       actualPremiums[time_iter+timeGap].delta,     actualSpots[time_iter].spot ,   actualSpots[time_iter+timeGap].spot,    expectedPremiums[time_iter+timeGap].time.strftime("%d/%m/%Y, %H:%M:%S"), indiv_pnl])
                
        time_iter+=1
    try:
        rows[0].extend([ "    "," Total PnL:"])
        rows[1].extend([ "    ",pnl])
    except:
        pass
    profit_i = 0
    loss_i = 0 
    for i in range(1, len(rows)):
        row=rows[i]
        if float(row[9]) >= 0:
            profit_i +=1
        else:
            loss_i +=1
    try:
        rows[3].extend(["   ","positive:",profit_i])
        rows[4].extend(["   ","negative:",loss_i])
        rows[6].extend(["   ","EPAP:",p])
        rows[7].extend(["   ","EPAL:",q])
        rows[8].extend(["   ","ELAP:",s])
        rows[9].extend(["   ","ELAL:",r])
    except:
        pass

    predict_good = (p + r)/(p+q+r+s)

    # if(os.path.exists(location)==False):
    #     os.mkdir(location)

    with open(location  , "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return pnl, predict_good


def getStrikePrices(path,optionType,year,month,day):
    startDate = datetime(year,month,day,9,15,0)
    data = load_pkl(path)
    strikePrices = []
    len_ex=0
    ex = []
    for expiry_date in data:
        len_ex+=1
        ex.append(expiry_date)
    if len_ex==1:
        for expiry_date in data:
            if optionType in data[expiry_date][startDate]:
                for strike_price in data[expiry_date][startDate][optionType]:
                    strikePrices.append(int(strike_price))
    if len_ex==2:
        if optionType in data[ex[1]][startDate]:
                for strike_price in data[expiry_date][startDate][optionType]:
                    strikePrices.append(int(strike_price))
    

    return strikePrices

def computeGreeks(grk, pathoriginal,pathtocreate,fileName, spotData, windowSize, timeGap, targetStrike, optionType, smoothingFactor,date=1, month=12, year=2020, hourFrom=9, minuteFrom=15, secondFrom=00, hourTo=15, minuteTo=30, secondTo=00, estimation_type="simple", greek_use="predict"):
    # loading the pickle data
    # pkl_file_location = os.path.join(path, fileName)
    data = load_pkl(fileName)   
    
    # date_today = datetime(year, month, date, hourFrom, minuteFrom, secondFrom)
    print(year,"-",month,"-",date,targetStrike)

    averageValues = []
    expectedValues = []
    # the buffer with specified window size
    windowBuffer = []
    actualValues = []
    seconds_checker = 0
    exps = []
    imp_vpairs = []
    #contract_index = {'strike':0,'price':1','vol':2,'oi':3,'imp_v':4, 'delta':5 ,'theta':6'vega':7 ,'gamma':8,'rho':9}
    for expiry_date in data:
        exps.append(expiry_date)

    for expiry_date in data:
        # exps.append(expiry_date)
        temp=0
        if len(exps)==2:
            temp = exps[0]
        #     exps[1] = exps[0]
        #     exps[0] = temp
        # expiry_date = exps[0]
        if expiry_date!=temp:
            for trade_time in data[expiry_date]:

                # since there can be multiple contracts at a particular second
                # we find an equivalent contract for the second
                secondBuffer = []
                seconds_checker+=1
                for option_type in data[expiry_date][trade_time]:
                    if(option_type == optionType):
                        found=0
                        for strike_price in data[expiry_date][trade_time][option_type]:
                            
                            if strike_price != targetStrike:
                                continue
                            elif strike_price == targetStrike:
                                #if a contract exists for the given target strike at that second
                                found+=1
                                # all contracts for the given target strike at that second
                                strike_dict = data[expiry_date][trade_time][option_type][strike_price]
                                for contract in strike_dict:
                                    price, vol, oi, imp_v, delta, theta, vega, gamma, rho = contract["price"], contract["volume"], contract["oi"], contract["imp_v"], contract["delta"], contract["theta"], contract["vega"],contract["gamma"], contract["rho"]
                            #sabka price * vol ka weighted average and simple prices/num of contracts
                                    thisContract =  Contract(price, vol, imp_v, delta, theta, vega, gamma, rho, trade_time)                          
                                    secondBuffer.append(thisContract)

                        if found>0:
                            secondPrice = 0
                            secondVol = 0
                            secondImp_v = 0
                            secondDelta = 0
                            secondTheta = 0
                            secondVega = 0
                            secondGamma = 0
                            secondRho = 0
                            secondLen = len(secondBuffer)
                            # Weighted price variables
                            prices_buffer = []
                            vol_buffer = []
                            for secondContracts in secondBuffer:
                                # secondPrice += secondContracts.price/secondLen
                                # secondVol += secondContracts.vol/secondLen
                                prices_buffer.append(secondContracts.price)
                                vol_buffer.append(secondContracts.vol)
                                secondImp_v += secondContracts.imp_v/secondLen
                                secondDelta += secondContracts.delta/secondLen
                                secondTheta += secondContracts.theta/secondLen
                                secondVega += secondContracts.vega/secondLen
                                secondGamma += secondContracts.gamma/secondLen
                                secondRho += secondContracts.rho/secondLen
                            
                            for vol in vol_buffer:
                                secondVol+=vol

                            for i in range(len(prices_buffer)):
                                secondPrice+=(prices_buffer[i]*vol_buffer[i])/secondVol
                            
                            secondVol/=len(vol_buffer)
                            # this is the equivalent average-values contract for the second
                            secondEquivalent = Contract(secondPrice, secondVol, secondImp_v, secondDelta, secondTheta, secondVega, secondGamma, secondRho, trade_time);
                            # this becomes the actual value for the second (assumption)
                            actualValues.append(secondEquivalent)
                            # print(secondEquivalent.time)
                            # print(trade_time,secondImp_v)
                            imp_vpairs.append(impvpair(trade_time,secondImp_v))
                            
                        

                            
                            if len(windowBuffer) < windowSize:
                                # if window is currently < required windowSize
                                windowBuffer.append(secondEquivalent)
                            elif len(windowBuffer) == windowSize:
                                # push and pop next second's contract to keep the window equal
                                # to the window size required
                                windowBuffer.pop(0)
                                windowBuffer.append(secondEquivalent)
                        else:         

                            # if there was no contract available for that time with given 
                            # strike and type then we just duplicate and push the last found contract (assumption)
                            if len(windowBuffer) < windowSize:       
                                # print(windowBuffer, len(windowBuffer), expiry_date)   
                                windowBuffer.append(windowBuffer[len(windowBuffer)-1])  
                            elif len(windowBuffer) == windowSize:
                                windowBuffer.append(windowBuffer[len(windowBuffer)-1])
                                windowBuffer.pop(0)
                            try:
                                dup = copy.deepcopy(actualValues[len(actualValues)-1])
                                dup.time += timedelta(seconds=1) 
                                actualValues.append(dup)
                            except:
                                print("empty")
                            

                            # expectedValues.append(dup)
                            found=0
                            # print(trade_time, secondImp_v)
                            imp_vpairs.append(impvpair(trade_time,imp_vpairs[len(imp_vpairs)-1].imp_v))
                        if len(windowBuffer) == windowSize:
                            #average values
                            # average values for the current window duration
                            priceAverage, vol_Av, imp_vAverage, deltaAverage, thetaAverage, vegaAverage, gammaAverage, rhoAverage = computeAverage(windowBuffer, windowSize)
                            nextValues = Contract(priceAverage, 1, imp_vAverage, deltaAverage, thetaAverage, vegaAverage, gammaAverage, rhoAverage, trade_time)

                            #expected values
                            # expected values for the next 'timeGap' seconds
                            if(estimation_type == "simple"):
                                priceExpected, volEx, imp_vExpected, deltaExpected, thetaExpected, vegaExpected, gammaExpected, rhoExpected = computeExpected(windowBuffer, windowSize, timeGap)
                            
                            if(estimation_type == "regress"):
                                priceExpected, volEx, imp_vExpected, deltaExpected, thetaExpected, vegaExpected, gammaExpected, rhoExpected = computeExpectedRegress(windowBuffer, windowSize, timeGap)
                            
                            if(estimation_type == "ema"):
                                priceExpected, volEx, imp_vExpected, deltaExpected, thetaExpected, vegaExpected, gammaExpected, rhoExpected = computeEMA(windowBuffer,windowSize,smoothingFactor,timeGap)

                            expValues = Contract(priceExpected, 1, imp_vExpected, deltaExpected, thetaExpected, vegaExpected, gammaExpected, rhoExpected, trade_time)

                            averageValues.append(nextValues)
                            expectedValues.append(expValues)

                    if(len(data[expiry_date][trade_time])==1):
                        if optionType not in data[expiry_date][trade_time]:

                            # print(option_type, trade_time)
                            dup1 = copy.deepcopy(actualValues[len(actualValues)-1])
                            dup1.time += timedelta(seconds=1) 
                            actualValues.append(dup1)
                            dup2 = copy.deepcopy(averageValues[len(averageValues)-1])
                            dup2.time += timedelta(seconds=1) 
                            averageValues.append(dup2)
                            dup3 = copy.deepcopy(expectedValues[len(expectedValues)-1])
                            dup3.time += timedelta(seconds=1) 
                            expectedValues.append(dup3)


        i=0
        e = []
        # exp = [25, 26..]
        # ac = [20,21, 22...]
        # e = [0 ,0,0,0,0, 25th]
        # shifting the expected values by 'timeGap' seconds and 
        # interpolating to get the expected value after 'timeGap' seconds
        actualValues = actualValues[windowSize -1:]
        # print(expectedValues[0].time)
        for ex in range(len(expectedValues)):              
            if i<(timeGap):
                # since we have no prediction for the first (windowSize + timeGap) seconds
                # cur_time = expectedValues[ex].time
                noneContract = Contract(0,0,0,0,0,0,0,0,0)
                e.append(noneContract)
            else:   
                ###   
                # adding expected change and current actual contract values
                ###     mx+c
                k=0
                if(expectedValues[ex].time == actualValues[ex].time):
                    # print(True)
                    e.append(addCon(actualValues[ex-timeGap], expectedValues[ex-timeGap], timeGap))        
                # if ex < 50 : print(ex)
                else:
                    # print(expectedValues[ex-1].time , actualValues[ex-1].time)
                    k+=1
                # print("Some data was missing")
            i+=1
        expectedValues = e

        spotsActual = []
        expectedSpotsChanges = []
        # print(len(imp_vpairs))
        # spotsActual = computeSpotsActual(year, month, date, spotPath)
        
        expectedSpotsChanges, spotsActual = computeExpectedSpotChanges(year, month, date, spotData, windowSize, timeGap, estimation_type, smoothingFactor)
        expectedImpVChanges = computeExpectedImpvChanges(imp_vpairs,windowSize,timeGap,year,month,date,'ema')

        spotsActual = spotsActual[windowSize - 1:]
        imp_vpairs = imp_vpairs[windowSize - 1:]
        # print(spotsActual[0].time, expectedSpotsChanges[0].time)
        # break
        es = []
        for i in range(timeGap):
            es.append(spotpair(0,0))
        
        for spotchange in expectedSpotsChanges:
            es.append(spotchange)
        

        expectedSpotsChanges = es

        print(f"Plotting for {estimation_type} {greek_use} smoothingFactor: {smoothingFactor}, {windowSize}_{timeGap}")

        # computePlotAct ual(actualValues, expectedValues, averageValues, windowSize, timeGap, targetStrike, hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, date, month, year, estimation_type, greek_use)

        # computePlotError(actualValues, expectedValues, averageValues, windowSize, timeGap, targetStrike, hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, date, month, year, estimation_type, greek_use)                

        # grk = "vega"
                
        if(grk == "vega"):
            expectedPremiumsvega = computePremiumExpectedVega(actualValues,expectedValues,expectedImpVChanges,windowSize,timeGap,targetStrike,hourFrom,minuteFrom,secondFrom,hourTo,minuteTo,secondTo,date,month,year,estimation_type,greek_use)
            pnl = ProfitorLossforaDay(grk, pathtocreate,expectedPremiumsvega, imp_vpairs, expectedImpVChanges,spotsActual, expectedSpotsChanges, actualValues, expectedValues,timeGap, windowSize, optionType,0.01, targetStrike, date, month, year, hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, greek_use, estimation_type, smoothingFactor)
        
        if(grk == "delta"):
            expectedPremiums = ccomputePremiumExpectedDelta(actualValues, expectedValues,  expectedSpotsChanges, windowSize, timeGap, targetStrike, hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, date, month, year,estimation_type, greek_use)
            pnl = ProfitorLossforaDay(grk, pathtocreate,expectedPremiums, imp_vpairs, expectedImpVChanges,spotsActual, expectedSpotsChanges, actualValues, expectedValues,timeGap, windowSize, optionType,0.01, targetStrike, date, month, year, hourFrom, minuteFrom, secondFrom, hourTo, minuteTo, secondTo, greek_use, estimation_type, smoothingFactor)

        print("PNL: ", pnl)
        strikePnl = [targetStrike, pnl]

        # plotValFit(windowSize,timeGap,spotData,year,month,date,targetStrike)
        # location1 = f"D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{greek_use}_{estimation_type}_strikeReports.txt "
        # location1 = f"X:\\NXBLOCK\\OptionsTradingStrategy\\Reports\\{year}_{month}_{date}\\{greek_use}_{estimation_type}_strikeReports.txt "

        # with open(location1  , "a") as f:
        os.chdir(pathtocreate)
        pathtocreate += "\\"+str(year)
        if(os.path.exists(pathtocreate)==False):
            os.mkdir(pathtocreate)
        os.chdir(pathtocreate)

        pathtocreate += "\\"+str(month)
        if(os.path.exists(pathtocreate)==False):
            os.mkdir(pathtocreate)
        os.chdir(pathtocreate)

        pathtocreate += "\\"+str(date)
        if(os.path.exists(pathtocreate)==False):
            os.mkdir(pathtocreate)
        os.chdir(pathtocreate)

        # location1 = pathtocreate + "\\" + f"{estimation_type}_{greek_use}_strikeReports.txt"
        location1 =  pathtocreate + "\\" +f"{estimation_type}_{greek_use}_strikeReports.txt"
        # if(os.path.exists(location1)==False):
        #     os.mkdir(location1)
        
        file = open(location1, "a")
        file.write(str(strikePnl[0])+" " +f"{windowSize}_{timeGap}" + f"{optionType}" +f"{smoothingFactor}"+"---->" + str(strikePnl[1]) +"\n" )
        file.close()
        os.chdir(pathoriginal)
        return pnl
#plot shows expected change not actual value
#expected premium not calculated from greeks

if __name__ == "__main__":
    # path = "D:\\Desktop\\College Documents\\ProjectExtramarks2\\OptionsTradingStrategy"
    # path = "X:\\NXBLOCK\\OptionsTradingStrategy"
    # fileName = "10.pkl"

    # spotPath = "D:\\College Documents\\ExtramarksOptionsTrading\\OptionsTradingStrategy\\BANKNIFTY_spot_seconds_till_2022_10_12.json"
    spotPath = "X:\\NXBLOCK\\BANKNIFTY_spot_seconds_till_2022_10_12.json"
    
    print("Spot JSON loading..")
    spotData = load_json(spotPath)
    print("Spot JSON loaded ")
    # pathtopkl = "D:\\College Documents\\ExtramarksOptionsTrading\\OptionsTradingStrategy\\BANKNIFTY" # add the path
    pathtopkl = "X:\\NXBLOCK\\OptionsTradingStrategy\\BANKNIFTY" # add the path
    # pathtocreate = "D:\\College Documents\\ExtramarksOptionsTrading\\OptionsTradingStrategy\\Reports" # add the path
    pathtocreate = "X:\\NXBLOCK\\OptionsTradingStrategy\\AllReports" # add the path
    monthtonum = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}

    # change path to save graphs and csv
    # --------------------------------------------------------------
    # windowSize = int(input("Size of window: "))
    # timeGap = int(input("Gap of time: "))
    # targetStrike = int(input("Strike price: "))
    # optionType = input("Option Type: (CE/PE) ")
    # computeGreeks(path, fileName,windowSize , timeGap, targetStrike, optionType)
    # --------------------------------------------------------------
    # <      Target strike should lie in first second's data!     >
    # --------------------------------------------------------------
    # type of greeks used to predict premium at time t+5:  
    # 1. greek_use = "now"  --->  actual greek at time = t
    # 2. greek_use = "predict" ---> predicted greek at time = t
    # --------------------------------------------------------------
    # type of estimation for values
    # 1. estimation_type = "regress" ---> regression used to interpolate values atb t+5
    # 2. estimation_type = "simple" ---> average of changes used to interpolate values at t+5
    # 3. estimation_type = "ema"  ---> expected moving average
    # --------------------------------------------------------------
    os.chdir(pathtopkl)

    grk = "delta"

    for year in os.listdir():
        os.chdir(pathtopkl+"\\"+year)
        yearNum = int(year)
        for month in os.listdir():
            os.chdir(pathtopkl+'\\'+year+'\\'+month)
            monthNum = monthtonum[month]
            month_pnl = 0
            for date in os.listdir():
                # os.chdir(pathtopkl+"\\"+year+"\\"+month+"\\"+date)
                dateNum = int(date[:-4])
                
                for greek_use_i in ["now"]:
                    for estimation_type_i in["ema"]:
                        # targetStrike = 28700
                        for prev_windowSize in [ 3]:
                            for timeGap in [2]:
                                optionType = "PE"
                                # optionType = input('Enter the option type (in caps):')
                                # optionType = "PE"
                                strikePrices = getStrikePrices(date,optionType,yearNum,monthNum,dateNum)
                                if(estimation_type_i == "ema"):
                                    
                                    for smoothingFactor in [0.01, 0.03, 0.3, 0.4]:
                                        pg_list = []
                                        pnl_fac = 0
                                        for targetStrike in strikePrices:
                                            pnl, hit_rate = computeGreeks(grk, pathtopkl+"\\"+year+"\\"+month,pathtocreate,date, spotData, prev_windowSize, timeGap, targetStrike, optionType, smoothingFactor,dateNum, monthNum, yearNum, hourFrom=9, minuteFrom=15, secondFrom=0, hourTo=15, minuteTo=30, secondTo=0, estimation_type=estimation_type_i, greek_use=greek_use_i)
                                            pnl_fac += pnl
                                            pg_list.append(hit_rate)
                                        
                                        av_hitrate = sum(pg_list)/len(pg_list)

                                        location1 =  pathtocreate + "\\" + f"{year}" +"\\"+ f"{monthtonum[month]}" +"\\"+f"{date[:-4]}"+ "\\" +f"{estimation_type_i}_{greek_use_i}_{grk}_DayReport.txt"
                                        file = open(location1, "a")
                                        file.write(f"{prev_windowSize}_{timeGap} " + f"{optionType} " +f"{smoothingFactor} "+"----> " + str(pnl_fac) + " "+ "Average hit rate = " + " " + str(av_hitrate) +"\n" )
                                        file.close()
                                    

                                else:
                                    for targetStrike in [30000]:
                                            computeGreeks(grk, pathtopkl+"\\"+year+"\\"+month+"\\"+date,pathtocreate,date, spotData, prev_windowSize, timeGap, targetStrike, optionType, smoothingFactor,dateNum, monthNum, yearNum, hourFrom=9, minuteFrom=15, secondFrom=0, hourTo=15, minuteTo=30, secondTo=0, estimation_type=estimation_type_i, greek_use=greek_use_i)



