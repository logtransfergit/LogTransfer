#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import argparse
import csv
import re

def readDir(rootdir,template_all,template_path_all,seq_path_all,mode):
    """
    Processing folder.If dir continue to find file,if file check mode and handle the file.
    
    Arguments
    ------
        rootdir: str, folder path 
        template_all: list, all templates list
        seq_path_all: list, all sequence files'path list
        template_path_all: list, all template files' path list
        mode: str, "template" or "template_path" or "seq_path"      
    
    Returns
    -----
        None
    
    """
    filelist = sorted(os.listdir(rootdir))
    for i in filelist:
        path = os.path.join(rootdir,i)
        if os.path.isdir(path):
            readDir(path,template_all,template_path_all, seq_path_all,mode)
        else:
            if mode=='template':
                readFileTemplate(path,template_all)
            elif mode == 'template_path':
                readFilePathTemplate(path,template_path_all)
            else:
                readFilePathSeq(path,seq_path_all)

def readFilePathSeq(path,seq_path_all):
    """
    Add file's path to seq_path_all if it is endswith 'seq'.
    
    Arguments
    ------
        path: str, file path     
    
    Returns
    -----
        None
    
    """
    if path.endswith('seq'):
        seq_path_all.append(path) 

def readFilePathTemplate(path,template_path_all):
    """
    Add file's path to template_path_all if it is endswith 'order'.
    
    Arguments
    ------
        path: str, file path     
    
    Returns
    -----
        None
    
    """
    if path.endswith('_order'):
        template_path_all.append(path)

def readFileTemplate(path,template_all):
    """
    Add all templates to template_all.
    
    Arguments
    ------
        path: str, files path     
    
    Returns
    -----
        None
    
    """
    if path.endswith('_order'):
        with open(path,'r') as f:
            while True:
                line = f.readline()
                if line:
                    template_all.append(line.strip())
                else:
                    break

def readFileLog(rootdir,log_all):
    """
    Processing folder.If dir continue to find file,if file check name and add all log file path to log_all.
    
    Arguments
    ------
        rootdir: str, folder path    
        log_all: list, all log files' path list 
    
    Returns
    -----
        None
    
    """    
    filelist = sorted(os.listdir(rootdir))
    for i in filelist:
        path = os.path.join(rootdir,i)
        if os.path.isdir(path):
            readFileLog(path, log_all)
        else:
            if path.endswith('.log'):
                log_all.append(path)

def getFileLogPath(rootdir):
    """
    Processing folder.Find all log path.
    
    Arguments
    ------
        rootdir: str, folder path    
   
    Returns
    -----
        log_path_all: list, all log path
    
    """    
    log_path_all = []
    readFileLog(rootdir, log_path_all)
    return log_path_all

def readFileMonthSeg(path):
    """
    Processing file to find two consecutive logs are in different months and the latter one is not in the first day of the month.
    
    Arguments
    ------
        path: str, file path
    
    Returns
    -----
        monthseg_list: list, example:[[m,n],[],..[]] m is the start line and n is the end line
    
    """ 
    monthseg_list = []
    if path.endswith('log'):
        start = 1
        end = -1
        count = 0
        month_pre = 'NuLL'
        with open(path,'r') as f:
            while True:
                line = f.readline()
                if line:
                    count = count + 1
                    month_res = re.search('Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec', line)
                    if month_res:
                        if month_res.group() != month_pre:
                            end = end + 1
                            if end != 0:
                                if line.strip().split()[1] != '1':
                                    monthseg_list.append([start,end])
                                    start = end + 1
                            month_pre = month_res.group()
                        else:
                            end = end + 1
                else:
                    break
    monthseg_list.append([start,end])
    return monthseg_list

def getSeqPathAll(rootdir):
    """
    Get the path of the seq file under the folder.
    
    Arguments
    ------
        rootdir: str, folder path
    
    Returns
    -----
        seq_path_list: list, seq file path
    
    """ 
    seq_path_list = []
    mode = 'seq_path'
    readDir(rootdir,[],[],seq_path_list,mode)
    return seq_path_list

def getTemplatePathAll(rootdir):
    """
    Get the path of the template file under the folder.
    
    Arguments
    ------
        rootdir: str, folder path
    
    Returns
    -----
        template_path_list: list, template file path
    
    """ 
    template_path_list = []
    mode = 'template_path'
    readDir(rootdir,[],template_path_list,[],mode)
    return template_path_list
 
def getTemplateAll(rootdir,template_all_path):
    """
    Get all templates under the folder and write them to files.
    
    Arguments
    ------
        rootdir: str, folder path
        template_all_dir: str, save templates' path
    
    Returns
    -----
      None
    
    """ 
    template_all = []
    mode = 'template'
    readDir(rootdir,template_all,[],[],mode) 
    template_all = list(set(template_all))
    with open(template_all_path, 'w') as w:
        for i,template in enumerate(template_all):
            w.write(str(template) + '\n')

def getVecTemplate(vector_path):
    """
    Get all templates under the folder and write them to files.
    
    Arguments
    ------
        vector_path: str, vector file path
    
    Returns
    -----
        dic_template_vec: dict, 'key': str, template 'val': list, template embedding vector
    
    """ 
    dic_template_vec = {}
    with open(vector_path,'r') as file:
        for l in file:
            template = l.split('//')[0]
            vec = l.split('//')[1].replace("[","").replace("]","")
            vec_list_str = vec.split()
            vec_list = list(map(float,vec_list_str))
            dic_template_vec[template] = vec_list
    return dic_template_vec

def readSeqFile(path):
    """
    Get sequences number in seq file.
    
    Arguments
    ------
        path: str, seq file path
    
    Returns
    -----
        numseq: list, sequences number
    
    """ 
    numseq = []
    with open(path, 'r') as f:
        for l in f:
            numseq.append(int(l.split()[-1]))
    return numseq

def readTemplateFile(path):
    """
    Get templates in single template file.
    
    Arguments
    ------
        path: str, templates file path
    
    Returns
    -----
        template: list, example: [[a,b,c....x],[],...[]]
    
    """ 
    template = []
    with open(path, 'r') as f:
        for line in f:
        	template.append(line.strip())
    return template

def getSingleFileInput(single_step, window_size, num_seq, Template, dic_template_vec,line_range_list):
    """
    Format input data.
    
    Arguments
    ------
        single_step: int, step_size
        window_size: int, window_size
        num_seq: list, input sequences
        Template: list, template sequence
        dic_template_vec: dict, template embedding vector
        line_range_list: list, right lines number without month fault
    
    Returns
    -----
        single_file_matrix: list, example: [[[],[]...[]],[],...[]] n*window_size*template_embedding_size(n is depended on file size)
    
    """ 
    single_file_matrix = []
    for line_range in line_range_list:
        start = int(line_range[0])
        end = int(line_range[1])
        n_lines = end - start + 1
        if n_lines < window_size + 1:
            continue
        else:
            current_lines = num_seq[start - 1: end]
            for i in range(0, len(current_lines) - window_size, single_step):
                single_window_matrix = []
                for j in range(window_size):
                    single_window_matrix.append(dic_template_vec[Template[current_lines[i+j]-1].lower()])
                single_file_matrix.append(single_window_matrix)
    print('Single Input generation down, length of the input is:{0}'.format(len(single_file_matrix)))
    return single_file_matrix

def getSingleFileOutput(single_step, window_size, IP_addr, label_dir,line_range_list):
    """
    Format input label.
    
    Arguments
    ------
        single_step: int, step_size
        window_size: int, window_size
        IP_addr: str, IP address
        label_dir: str, label file dir
        line_range_list: list, right lines number without month fault
    
    Returns
    -----
        Single_file_output: list, labels
        Single_file_IP_Chunk: list, example:[[IP,line_number],[]...[]] for backtracking
    
    """ 
    Single_file_output = []
    Single_file_IP_Chunk = []
    output_temp = []
    filelist = os.listdir(label_dir)
    for i in filelist:
        File_IP = MatchIP(i)
        if File_IP == IP_addr:
            label_path = os.path.join(label_dir,i)
            with open(label_path, 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        output_temp.append(line.strip().split()[0])
                    else:
                        break
    for line_range in line_range_list:
        start = int(line_range[0])
        end = int(line_range[1])
        n_lines = end - start + 1
        if n_lines < window_size + 1:
            continue
        else:
            current_lines = output_temp[window_size + start - 1: end]
            for i in range(0, len(current_lines),single_step):
                Single_file_output.append(current_lines[i])
                Single_file_IP_Chunk.append([IP_addr,start + window_size + i + 1])
    print('length of the output {0} is {1} '.format(IP_addr,len(Single_file_output)))
    return Single_file_output , Single_file_IP_Chunk

def MatchIP(path):
    """
    Find the IP that matches the rule.
    
    Arguments
    ------
        path: str, file path
    
    Returns
    -----
        IP: str, IP address
    
    """ 
    #path = path.split('\\')#for windows path
    path = path.split('/')
    result = re.match('\d{2,3}_\d{2,3}',path[-1])
    if result:
        IP = result.group()
    else:
        print(path)
        print('can not match IP! exit')
        IP = '0'
    return IP


def getSelectedIOdata(single_step, window_size, rootdir, label_dir, vector_path, selected_switchid_list):
    """
    Format data.
    
    Arguments
    ------
        single_step: int, step_size
        window_size: int, window_size
        rootdir: str, logs(train data) dir
        label_dir: str, labels dir
        vector_path: str, template vector path
        selected_switchid_list: list, selected file number

    Returns
    -----
        dataX: list, example: [[[],[]...[]],[],...[]] n*window_size*template_embedding_size(n is depended on file size) input data
        dataY: list, list, labels
        data_IP_Chunk: list, example:[[IP,line_number],[]...[]] for backtracking
    
    """
    dataX = []
    dataY = []
    data_IP_Chunk = []
    seq_file_path = getSeqPathAll(rootdir)
    Template_file_path = getTemplatePathAll(rootdir)
    dic_template_vec = getVecTemplate(vector_path)
    Log_path = getFileLogPath(rootdir)
    stra = ''.join(selected_switchid_list)
    stra = re.sub('\[|\]|,',' ',stra)
    stra = stra.strip().split(' ')
    #stra = selected_switchid_list
    for i in stra:
        i = int(i)
        IP_addr = MatchIP(seq_file_path[i])
        print(IP_addr)
        if IP_addr != '0':
            print('Current IP address is {0}'.format(IP_addr))
            num_seq = readSeqFile(seq_file_path[i])
            print('seq_file_path is:{0}'.format(seq_file_path[i]))
            template = readTemplateFile(Template_file_path[i])
            print('Template_file_path is:{0}'.format(Template_file_path[i]))
            line_range_list = readFileMonthSeg(Log_path[i])
            dataX.extend(getSingleFileInput(single_step,window_size, num_seq,template, dic_template_vec,line_range_list))
            Y, IP_Chunk = getSingleFileOutput(single_step,window_size, IP_addr, label_dir,line_range_list)
            dataY.extend(Y)
            data_IP_Chunk.extend(IP_Chunk)
        else:
            continue
    print('length of dataX is {0}, length of dataY is {1}'.format(len(dataX), len(dataY)))
    return dataX, dataY, data_IP_Chunk

def GetFailureSE(FailureCSV):
    """
    Get anomaly IP and lines.
    
    Arguments
    ------
        FailureCSV: str, anomaly file path      
    
    Returns
    -----
        dic_IP_SE: list, example:  dic_IP_SE[IP_address] = [[a,b...],[c,d...]] a and c are the same anomaly' corresponding start and end line number
    
    """
    IP_list = []
    dic_IP_SE = {}
    with open(FailureCSV, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            IP_address = row[0]
            IP = IP_address.split('.')
            Type = row[3]
            Start = row[4]
            if len(Start) == 0 or Start == 'NULL':
                Start = '0'
            End = row[5]
            if len(End) == 0 or End == 'NULL':
                End = '0'
            if IP_address not in IP_list:
                IP_list.append(IP_address)
                dic_IP_SE[IP_address] = [[int(Start)],[int(End)]]
            else:
                dic_IP_SE[IP_address][0].append(int(Start))
                dic_IP_SE[IP_address][1].append(int(End))
    return dic_IP_SE

def load(para):
    """
    Format data according to parameters.
    
    Arguments
    ------
        para: dict, parameters
        para['single_step']: int, step_size
        para['window_size']: int, window_size
        para['rootdir']: str, logs(train data) dir
        para['label_dir']: str, labels dir
        para['vector_path'] str, template vector path
        para['selected_switchid_list']: list, selected file number

    Returns
    -----
        X: list, example: [[[],[]...[]],[],...[]] n*window_size*template_embedding_size(n is depended on file size) input data
        Y: list, list, labels
        IP_Chunk: list, example:[[IP,line_number],[]...[]] for backtracking
    
    """
    single_step = para['single_step']
    window_size = para['window_size']
    rootdir = para['rootdir']
    label_dir = para['label_dir']
    vector_path = para['vector_path']
    selected_switchid_list = para['selected_switchid_list']
    X, Y, IP_Chunk = getSelectedIOdata(single_step,window_size, rootdir, label_dir, vector_path, selected_switchid_list)
    return X,Y,IP_Chunk

def createDir(path):
    """
    If dir does not exists, make the dir
    
    Arguments
    ------
        path: str, path to be made
    
    Returns
    -----
        None
    
    """
    import os
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)

if __name__ == '__main__':
    B6220_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    D2020_list = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    S5500_list = [36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]
    single_step = 3
    window_size = 20
    rootdir = '../Logs'
    label_dir = '../labels/S5/'
    vector_path = '../Logs/template_vec.dat'
    dataX, dataY, data_IP_Chunk = getSelectedIOdata(single_step, window_size, rootdir, label_dir, vector_path, S5500_list)
    print('length of B6220_dataX is {0}'.format(len(dataX)))
