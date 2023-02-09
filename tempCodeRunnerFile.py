:
    for i in range(0, windowSize - timeGap):
        average_change=0
        for j in range(0,timeGap):
            average_change += (arr[i+j+1]-arr[i+j])*coefficient_arr[j]
            average_change /= sum(coefficient_arr)

        avg_buffer.append(average_change)
    # try:
    fin = sum(avg_buffer)/len(avg_buffer)