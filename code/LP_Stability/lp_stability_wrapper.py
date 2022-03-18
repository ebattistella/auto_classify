#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import struct
import subprocess
import os, sys
import time

def generate_simple_test(n = 15):
    '''
        Generate a random test with 3 cluster
    '''
    x1 = np.random.rand(n,2)
    x2 = 2+np.random.rand(n,2)
    x3 = 4+np.random.rand(n,2)
    
    X = np.concatenate((x1,x2,x3))
    
    return X


def distance(X,i,j):
    '''
        Calcul the distance between points to use in the clustering algorithm
    '''
    return np.sqrt( (X[i,0] - X[j,0])**2 + (X[i,1] - X[j,1])**2 )

def create_inputfile(X,n_points,input_f,dist = None,penalty = None,max_iter=50):
    '''
        Create the inputfile of the C++ LP stability algorithm 
        The function struct.pack is used to write bytes in a text file
        If distance is not an input, algorithm use euclidian distance
        If penalty is not an input, use 3 * median(dist)
    '''
    #print('Open input file')
    fileID = open(input_f,'wb')
    
    indices = np.arange(n_points)
    
    pairs = itertools.permutations(indices,2)
    pairs = np.array(list(pairs))
    
    #print('Calculating distances')
    
    if dist is None:
        #print('Using euclidian distances')
        dist = np.array([distance(X,pairs[i,0],pairs[i,1]) for i in range(len(pairs))]) 
    #else:
        #print('Using pre-calculated distances')
        
    if penalty is None:
        penalty = 3*np.median(dist)*np.ones(n_points)
        
    npairs = len(pairs)
    
    #print('Write bytes in the input file')
    # Conversion to bytes
    fileID.write(struct.pack('i',npairs))
    
    # First columns of pairs
    pairs0_list = pairs[:,0].tolist()
    bytes_string = struct.pack(str(npairs)+'i', *pairs0_list) # * force l'unpacking
    fileID.write(bytes_string)
      
    # Second column of pairs
    pairs1_list = pairs[:,1].tolist()
    bytes_string = struct.pack(str(npairs)+'i', *pairs1_list) 
    fileID.write(bytes_string)
    
    # Distance
    dist = dist.tolist()
    bytes_string = struct.pack(str(npairs)+'f', *dist)
    fileID.write(bytes_string)

    
    # Penalty
    penalty = penalty.tolist()
    bytes_string = struct.pack(str(n_points)+'f',*penalty)
    fileID.write(bytes_string)
    
    fileID.write(struct.pack('i',max_iter))
    
    fileID.close()

def create_big_inputfile(X,n_points,input_f,dist = None,penalty = None,max_iter=50):
    '''
        Create the inputfile of the C++ LP stability algorithm 
        The function struct.pack is used to write bytes in a text file
        because the input file is big, you do two for loop instead of writing a big bits 
        It's longer but better for memory issue
    '''   
    #print('Open input file')
    fileID = open(input_f,'wb')
    
    #print('Calculating distances')
    
    if dist is None:
        #print('distance must be pre-calculated')
        return None
    #else:
        #print('Using pre-calculated distances')
        

    penalty = penalty * np.median(dist)*np.ones(n_points)
    npairs = n_points * n_points
    
    percentage = 5
    percentage_number = int(n_points*percentage/100)
    #print('Write bytes in the input file')
    # Conversion to bytes
    fileID.write(struct.pack('i',npairs))
    
    # First columns of pairs
    #print('Writing first pairs')
    #start = time.time()
    for i in range(n_points):
        for j in range(n_points):
            bytes_string = struct.pack('i', i) 
            fileID.write(bytes_string)
        #print_update_message(i,percentage,percentage_number,start)
      
    # Second column of pairs
    #print('Writing second pairs')
    for i in range(n_points):
        for j in range(n_points):
            bytes_string = struct.pack('i', j) 
            fileID.write(bytes_string)
         #print_update_message(i,percentage,percentage_number,start)
    #print('{} s needed'.format(time.time() - start))
    
    # Distance
    #print('Writing distances')
    #start = time.time()
    for i in range(n_points):
        for j in range(n_points):
            bytes_string = struct.pack('f', dist[i][j]) 
            fileID.write(bytes_string)
        #print_update_message(i,percentage,percentage_number,start)
    #print('{} s needed'.format(time.time() - start))

    # Penalty
    #print('Writing penalty')
    start = time.time()
    for i in range(n_points):
        bytes_string = struct.pack('f', penalty[i]) 
        fileID.write(bytes_string)
        #print_update_message(i,percentage,percentage_number,start)
        
    #print('{} s needed'.format(time.time() - start))
    fileID.write(struct.pack('i',max_iter))
    
    fileID.close()

def execute_cpp(solver,input_f,output_f):
    '''
        Execution of the C++ code
    '''
    startupinfo=None
    if sys.platform.startswith("win"):
        # this startupinfo structure prevents a console window from popping up on Windows
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    
    #print('Running LP stability C++ code')
    command = solver + ' ' + input_f + ' ' + output_f
    completed_process = subprocess.run(command,shell = True, capture_output=True,startupinfo=startupinfo)
    """print('Return code : ')
    print(completed_process.returncode)"""
    return completed_process

def read_outputfile(output_f,npoints):
    '''
        Read the output file of the C++ code
    '''
    #print('Open output file')
    fileID = open(output_f,'rb')
    
    #print('Read bytes')
    #best_num_exemplars 1 int    
    best_num_exemplars = fileID.read(4)
    best_num_exemplars = struct.unpack("i",best_num_exemplars)[0]
    
    # best_exemplars best_num_exemplars int
    best_exemplars = fileID.read(4*best_num_exemplars)
    temp_list = []
    for i in range(best_num_exemplars):
        temp_list.append(struct.unpack('i',best_exemplars[0+4*i:4+4*i])[0])
    
    best_exemplars = np.array(temp_list)
    
    # Nombre iteration 1 int
    iters = fileID.read(4)
    iters = struct.unpack('i',iters)[0]
    
    # primal_function.size primal_function_size int
    primal_function = fileID.read(4*iters)
    temp_list = []
    for i in range(iters):
        temp_list.append(struct.unpack('f',primal_function[0+4*i:4+4*i])[0])
    
    primal_function = np.array(temp_list)
    
    # dual_function.size = primal_function_size int
    dual_function = fileID.read(4*iters)
    temp_list = []
    for i in range(iters):
        temp_list.append(struct.unpack('f',dual_function[0+4*i:4+4*i])[0])

    dual_function = np.array(temp_list)
    
    #flabel
    flabel = fileID.read(4*npoints)
    temp_list = []
    for i in range(npoints):
        temp_list.append(struct.unpack('i',flabel[0+4*i:4+4*i])[0])
    flabel = np.array(temp_list)
    
    fileID.close()
    #print('Close...')
    

    return (best_num_exemplars,best_exemplars,iters,
            primal_function,dual_function,flabel)

def clear_folder(input_f,output_f):
    os.remove(input_f)
    os.remove(output_f)
    
def plot_cluster(X,flabel,best_exemplars):
    plt.scatter(X[:,0],X[:,1],c=flabel,s=50)
    
    plt.scatter(X[best_exemplars,0],X[best_exemplars,1],s=50,c='red',marker='+')
    
def LP_stability(X,file_location,solver, input_f,output_f):
    
    solver = file_location + solver
    input_f = file_location + input_f
    output_f = file_location + output_f
    
    npoints = len(X)
    
    #create_inputfile(X,input_f)
    execute_cpp(solver,input_f,output_f)
    (best_num_exemplars,best_exemplars,primal_function_size,
     primal_function,dual_function,flabel) = read_outputfile(output_f,npoints)
    plot_cluster(X,flabel,best_exemplars)
    clear_folder(input_f,output_f)
    
    return (best_num_exemplars,best_exemplars,primal_function_size,
     primal_function,dual_function,flabel) 
    
def print_update_message(i,percentage,percentage_number,start):
    if i % percentage_number == 0:
        print(' {:d} % done. Calculation time : {:.2f} '.format(
                percentage * (i // percentage_number) 
              , time.time() - start))
#%%

if __name__ == '__main__':
    
    file_location = os.path.abspath(os.path.dirname(__file__))

    solver = 'clustering/clustering.exe'
    input_f = 'input.txt'
    output_f = 'output.txt'

    n = 20
    X = generate_simple_test(n=n)
    
    (best_num_exemplars,best_exemplars,primal_function_size,
     primal_function,dual_function,flabel)  = LP_stability(X,file_location,solver, input_f,output_f)
