#!/usr/bin/python3
import sys
import math
def geomean(input_list):
    temp = 1
    for i in range(0, len(input_list)) :
        temp = temp * input_list[i]
    temp2 = (float)(math.pow(temp, (1 / len(input_list))))
    res = (float)(temp2)
    return res

if len(sys.argv) < 2:
    print('Need to specify a cache replacement policy!')
    print('Example usage: ./print_results.py my_policy')
    sys.exit(0)

policy = sys.argv[1]
lru_policy = "no"

print('Replacement Policy Performance')
print('(IPC, MPKI, name of benchmark)')
print('------------------------------')
mpki_list = []
ipc_list = []
lru_mpki_list = []
lru_ipc_list = []
trace_list = open('sim_list/traces_mem_intensive.txt', 'r')
for trace in trace_list:
    trace = trace[:-1]
    trace_output = open('out/'+policy+'/'+trace+'.txt', 'r')
    lru_output = open('out/'+lru_policy+'/'+trace+'.txt', 'r')
    mpki = 0.0
    ipc = 0.0
    lru_mpki, lru_ipc = 0.0, 0.0
    for line in trace_output:
        if line.startswith('Core_0_IPC'):
            ipc = float(line.split()[1])
    for line in lru_output:
        if line.startswith('Core_0_IPC'):
            lru_ipc = float(line.split()[1])
    mpki_list.append(mpki)
    ipc_list.append(ipc)
    lru_mpki_list.append((trace, lru_mpki))
    lru_ipc_list.append(lru_ipc)
    #print(trace)
    if ipc == 0:
        print("Missing trace:", trace)
    #print(trace, '%.4f' % ((ipc/lru_ipc - 1)*100))
    #print(trace, mpki)

#avg_mpki = sum(mpki_list)/len(mpki_list)
avg_ipc = sum(ipc_list)/len(ipc_list)
#print('%.5f' % (avg_ipc))
speedup = [ipc_list[i]/lru_ipc_list[i] for i in range(len(ipc_list))]
geomean_speedup = geomean(speedup)
#print("len: ", len(speedup))
print('GEOMEAN:', (geomean_speedup - 1) * 100.0)
