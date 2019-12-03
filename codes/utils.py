#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import argparse
import csv
import re

def readDir(rootdir,template_all,template_path_all,seq_path_all,mode):
    """ 
    read dir, if dir then continue to find file, if file then check mode and handle the file
    
    Arguments
    ----------
        rootdir: str, root file dir
        template_all: list, return all templates list
        seq_path_all: list, return all sequence files list
        template_path_all: list, return all template files list
        mode: str, 'template' or 'sequence'
    
    Returns
    ----------
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
    find file which name endwith 'seq' in a dir

    Arguments
    ----------
        path: str, check dest path
        seq_path_all: list, return all sequence files list

    Returns
    ----------
        None

    """
    if path.endswith('seq'):
        seq_path_all.append(path)

def readFilePathTemplate(path,template_path_all):
    """ 
    find file which name endwith '_Template' in a dir

    Arguments
    ----------
        path: str, check dest path
        template_path_all: list, return all template files list

    Returns
    ----------
        None

    """
    if path.endswith('_order'):
        template_path_all.append(path)

def readFileTemplate(path,template_all):
    """ 
    find all template in one file

    Arguments
    ----------
        path: str, check dest path
        template_all: list, return all templates list

    Returns
    ----------
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


def getSeqPathAll(rootdir):
    """ 
    get all files' path which end up with 'seq'

    Arguments 
    ----------
        rootdir: str, root file dir

    Returns
    ----------
        seq_path_list: list, needed files' path

    """
    seq_path_list = []
    mode = 'seq_path'
    readDir(rootdir,[],[],seq_path_list,mode)
    return seq_path_list

def getTemplatePathAll(rootdir):
    """ 
    get all files' path which end up with 'template'

    Arguments 
    ----------
        rootdir: str, root file dir

    Returns
    ----------
        template_path_list: list, needed files' path

    """
    template_path_list = []
    mode = 'template_path'
    readDir(rootdir,[],template_path_list,[],mode)
    return template_path_list


def getTemplateAll(rootdir,template_all_dir):
    """ 
    get all files' path which end up with '_Template'

    Arguments 
    ----------
        rootdir: str, root file dir

    Returns
    ----------
        template_all_dir: list template files    
        write 'serial number +  template' to file

    """
    template_all = []
    mode = 'template'
    readDir(rootdir,template_all,[],[],mode) 
    template_all = list(set(template_all))
    with open(template_all_dir, 'w') as w:
        for i,template in enumerate(template_all):
            w.write(str(template) + '\n')

def getVecTemplate(vector_path):
    """ 
    fine templates's vector 

    Arguments 
    ----------
        vector_path: str, output of glove test; template vectors

    Returns
    ----------
        Dic_Template_Vec: dict, keys = template and values = template vectors

    """
    Total_Vec_Template = []
    Dic_Template_Vec = {}
    with open(vector_path,'r') as file:
        while True:
            line = file.readline()
            if line:
                template = line.split('//')[0]
                vec = line.split('//')[1].replace("[","").replace("]","")
                vec_list_str = vec.split()
                vec_list = list(map(float,vec_list_str))
                Dic_Template_Vec[template] = vec_list
                Total_Vec_Template.append(vec_list)
            else:
                break
    return Dic_Template_Vec

def readSeqFile(path):
    """ 
    get num list in seq file 

    Arguments 
    ----------
        path: str, input file path

    Returns
    ----------
        numseq: list, sequence number list

    """
    numseq = []
    with open(path, 'r') as f:
        for l in f:
            numseq.append(int(l.split()[-1]))
    return numseq
   
def readTemplateFile(path):
    """ 
    read template file

    Arguments 
    ----------
        path: str, input file path

    Returns
    ----------
        Template: list, template file content

    """
    Template = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line:
                Template.append(line.strip())
            else:
                break
    return Template

def getSingleFileInput(single_step, window_size, num_seq, Template, dic_template_vec):
    """ 
    read template file 

    Arguments 
    ----------
        single_steps: int, step length
        window_size: int, window size of the input
        num_seq: list, sequence of the log file
        Template: list, template of the log file
        dic_template_vec: dict, relationship between template and the vector

    Returns
    ----------
        single_file_matrix: list, all the input matrix from one single log file

    """
    single_file_matrix = []
    for i in range(0, len(num_seq) - window_size, single_step):
        single_window_matrix = []
        for j in range(window_size):
            single_window_matrix.append(dic_template_vec[Template[num_seq[i+j]-1].lower()])
        single_file_matrix.append(single_window_matrix)
    print('Single Input generation down, length of the input is :{0}'.format(len(single_file_matrix)))
    return single_file_matrix

def getSingleFileOutput(single_step, window_size, IP_addr, label_dir):
    """
    read template file 

    Arguments 
    ----------
        single_steps: int, step length
        window_size: int, window size of the input
        IP_addr: str, last 2 digits of Target IP
        label_dir: str, dir of label files

    Returns
    ----------
        Single_file_matrix: list
        
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
    for i in range(window_size, len(output_temp),single_step):
        Single_file_output.append(output_temp[i])
        Single_file_IP_Chunk.append([IP_addr, i + 1])
    print('length of the output {0} is {1} '.format(IP_addr,len(Single_file_output)))
    return Single_file_output , Single_file_IP_Chunk

def MatchIP(path):
    path = path.split('/')
    result = re.match('\d{2,3}_\d{2,3}',path[-1])
    if result:
        IP = result.group()
    else:
        print('can not match IP! exit')
        IP = '0'
    return IP

def getSelectedIOdata(single_step, window_size, rootdir, label_dir, vector_path, switch_start, switch_end):
    dataX = []
    dataY = []
    data_IP_Chunk = []
    seq_file_path = getSeqPathAll(rootdir)
    Template_file_path = getTemplatePathAll(rootdir)
    dic_template_vec = getVecTemplate(vector_path)
    for i in range(switch_start-1,switch_end):
        IP_addr = MatchIP(seq_file_path[i])
        if IP_addr != '0':
            num_seq = readSeqFile(seq_file_path[i])
            template = readTemplateFile(Template_file_path[i])
            dataX.extend(getSingleFileInput(single_step,window_size, num_seq,template, dic_template_vec))
            Y, IP_Chunk = getSingleFileOutput(single_step,window_size, IP_addr, label_dir)
            dataY.extend(Y)
            data_IP_Chunk.extend(IP_Chunk)
        else:
            continue
    print('length of dataX is {0}, length of dataY is {1}'.format(len(dataX), len(dataY)))
    return dataX, dataY, data_IP_Chunk

def GetFailureSE(FailureCSV):
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
    single_step = para['single_step']
    window_size = para['window_size']
    rootdir = para['rootdir']
    label_dir = para['label_dir']
    vector_path = para['vector_path']
    switch_start = para['switch_start']
    switch_end = para['switch_end']
    X, Y, IP_Chunk = getSelectedIOdata(single_step,window_size, rootdir, label_dir, vector_path, switch_start, switch_end)
    return X,Y,IP_Chunk

def createDir(path):
    import os
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='unit_test_mode.', type=str, default= 'getTemplatePathAll')
    args = parser.parse_args()
    if args.mode == 'getTemplatePathAll':
        print('testing getTemplatePathAll module:')
        rootdir = '../Logs'
        TemplatePathAll = getTemplatePathAll(rootdir)
        print(TemplatePathAll)
        print('done')
    elif args.mode == 'getSeqPathAll':
        print('testing getTemplatePathAll module:')
        rootdir = '../Logs'
        SeqPathAll = getSeqPathAll(rootdir)
        print(SeqPathAll)
        print('done')
    elif args.mode == 'getTemplateAll':
        print('testing getTemplatePathAll module:')
        rootdir = '../Logs'
        template_all_dir = '../Logs/Template_All'
        getTemplateAll(rootdir, template_all_dir)
        print('done')
    elif args.mode == 'getVecTemplate':
        print('testing getVecTemplate module:')
        vector_path = '../Logs/template_vec.dat'
        template_vec_dic = getVecTemplate(vector_path)
        print(template_vec_dic)
    elif args.mode == 'readSeqFile':
        print('testing readSeqFile module:')
        rootdir = '../Logs'
        SeqPathAll = getSeqPathAll(rootdir)
        path = SeqPathAll[0]
        num_seq = readSeqFile(path)
        print(num_seq)
    elif args.mode == 'readTemplateFile':
        print('testing readSeqFile module:')
        rootdir = '../Logs'
        TemplatePathAll = getTemplatePathAll(rootdir)
        path = TemplatePathAll[0]
        Template = readTemplateFile(path)
        print(Template)
    elif args.mode == 'getSingleFileInput':
        print('testing getSingleFileInput module:')       
        rootdir = '../Logs'
        window_size = 10
        vector_path = '../Logs/template_vec.dat'
        template_vec_dic = getVecTemplate(vector_path)
        SeqPathAll = getSeqPathAll(rootdir)
        path = SeqPathAll[0]
        num_seq = readSeqFile(path)
        TemplatePathAll = getTemplatePathAll(rootdir)
        path = TemplatePathAll[0]
        Template = readTemplateFile(path)
        single_file_matrix = getSingleFileInput(window_size, num_seq, Template, template_vec_dic)
        print(single_file_matrix)
    elif args.mode == 'getSelectedIOdata':
        print('testing getSelectedIOdata module:')  
        window_size = 10
        rootdir = '../Logs'
        vector_path = '../Logs/template_vec.dat'
        label_dir = '../labels/D2020/'
        switch_start = 1
        switch_end = 5
        X,y,IP_Chunk = getSelectedIOdata(window_size, rootdir, label_dir, vector_path, switch_start, switch_end)
        print('done')
    else:
        print('mode does match')
