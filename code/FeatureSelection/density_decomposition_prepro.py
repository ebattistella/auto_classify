### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Density friendly decomposition; Interface;
### Description: <Python ingterface to the C++ code of density friendly deocmposition>
### Input: G: a weigted graph, filename: the path where to save the C++ code input, iter: number of iterations, ncpu: number of cpus
### Ouput: files with the decomposition and the densities
###################################

import numpy as np
import os, sys
import subprocess

# Execute the C++ code
def launch_decompo(filename, iter, ncpu):
    startupinfo = None
    if sys.platform.startswith("win"):
        # this startupinfo structure prevents a console window from popping up on Windows
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    command = ' '.join(['./density_decomposition', str(ncpu), str(iter), filename + ".txt", filename + "rates.txt", filename + "pavafit.txt",
         filename + "cuts.txt", filename + "exact.txt"])
    completed_process = subprocess.run(command, shell=True, capture_output=True, startupinfo=startupinfo)
    print('Return code : ')
    print(completed_process.returncode)
    return completed_process

# Write the input graph in a file
def write_file(graph, filename):
    with open(filename + ".txt", "w") as f:
        for i in range(len(graph)):
            for j in range(i):
                for _ in range(int(graph[i][j])):
                    f.write(str(i) + ' ' + str(j) + '\n')


# Read the generates files with the densities and rates per node to retrieve the decomposition
def read_decompo(filename):
    s_ranks = []
    n_ranks = []
    try:
        with open(filename + "exact.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                s_ranks.append(int(line.split(" ")[0]))
        s_ranks.sort(reverse=True)
        with open(filename + "rates.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                n_ranks.append(int(line.split(" ")[0]))
    except:
        return []
    S = []
    counter = 0
    for num in s_ranks:
        S.append(n_ranks[counter:counter+num])
        counter += num
    return S

# Launch a simple toy test
if __name__ == "__main__":
    iter = 100
    ncpu = 2
    G = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 0],
                  [0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
    directory = "./"
    filename = directory + "test"
    write_file(G, filename)
    completed_process = launch_decompo(filename, iter, ncpu)
    if completed_process.returncode == 0:
        S = read_decompo(filename)
        print(S)
    else:
        print("Error in decomposition", str(completed_process.returncode))
