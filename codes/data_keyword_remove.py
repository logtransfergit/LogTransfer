#!/usr/bin/python
# # -*- coding: utf-8 -*-
"""
remove some logs which don't relate to anomalies
"""

import os
import re
from utils import MatchIP
import argparse

def GetLogPathList(rootdir, LogPathList):
    filelist = sorted(os.listdir(rootdir))
    for i in filelist:
        path = os.path.join(rootdir,i)
        if os.path.isdir(path):
            GetLogPathList(path, LogPathList)
        else:
            ReadLogPath(path, LogPathList)
            
def ReadLogPath(path, LogPathList):
    if path.endswith('log'):
        LogPathList.append(path)
        
def GetLabelPathList(labeldir, LabelPathList):
    filelist = sorted(os.listdir(labeldir))
    for i in filelist:
        path = os.path.join(labeldir,i)
        print(path)
        labelpathlist = sorted(os.listdir(path))
        for j in labelpathlist:
            LabelPath = os.path.join(path,j)
            LabelPathList.append(LabelPath)

def PreprocessSingleLog(logfile, labelfile, removal):
    logs = []
    labels = []
    with open(logfile, 'r') as f1, open(labelfile,'r') as f2:
        for line1, line2 in zip(f1,f2):
            res = re.search(removal, line1)
            if res == None:
                logs.append(line1.strip())
                labels.append(line2.strip())
            else:
                continue
    with open(logfile, 'w') as f1:
        for line in logs:
            f1.write(line + '\n')
    with open(labelfile, 'w') as f2:
        for line in labels:
            f2.write(line + '\n')

def PreprocessLogs(rootdir, labeldir, removal):
    LogPathList = []
    LabelPathList = []  
    GetLogPathList(rootdir, LogPathList)
    GetLabelPathList(labeldir, LabelPathList)
    for logfile, labelfile in zip(LogPathList, LabelPathList):
        print('logfile is {0}'.format(logfile))
        print('labelfile is {0}'.format(labelfile))
        LogIP = MatchIP(logfile)
        LabelIP = MatchIP(labelfile)
        if LogIP == LabelIP:
            print('IP is OK, current IP is {0}'.format(LogIP))
            PreprocessSingleLog(logfile, labelfile, removal)
        else:
            print('IP is not equal, logIP is {0}; labelIP is {0}'.format(LogIP, LabelIP))
            continue
         
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', help='dir of the root.', type=str, default= '../Logs')
    parser.add_argument('--labeldir', help='dir of the label fir.', type=str, default= '../labels')
    parser.add_argument('--removal', help='str which need to be search in the log.', type=str, default= r'logined the switch|logouted from the switch')
    args = parser.parse_args()   
    PreprocessLogs(args.rootdir,args.labeldir,args.removal)
    