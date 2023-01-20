def averageofArray(arr):

    return sum(arr)/len(arr)

def EMA(arr,windowSize,smoothingFactor,numSeconds):

    beta = smoothingFactor/(windowSize+1)

    ans = 0
    prev_ema = 0

    starting_index = 0
    ending_index = numSeconds

    for i in range(0,windowSize/numSeconds):

        # curr_val = averageofArray(arr[starting_index:ending_index])
        curr_val = arr[ending_index-1] - arr[starting_index]
        ans+=beta*curr_val+(1-beta)*prev_ema
        prev_ema = ans
        starting_index = ending_index
        ending_index += numSeconds
    
    return ans


A = []

for i in range(1,21):
    A.append(i)

X = EMA(A,20,2,5)
print(X)
