import argparse
import sys,os,time
import subprocess
import re
import settings

caffebin = settings.OPTIMIZED_CAFFE_BIN
original_caffebin = settings.ORIGINAL_CAFFE_BIN
DEBUG = True 

config_file_home=settings.CONFIG_FILE_HOME

def get_average_time(filename, is_total=True):
    file = open(filename, "r")
    if is_total:
        search_str = 'Average Backward pass:'
        #search_str = 'Average Forward-Backward: '
    else:
        search_str = 'Average Forward pass: '
    err_str = 'Data layer prefetch queue empty'
    for line in file.readlines():
        if re.search(err_str, line):
            return None
        if re.search(search_str, line):
            start_idx = line.find(search_str)+len(search_str)
            end_idx = line.find('ms.')
            ms = float(line[start_idx:end_idx].strip())
            file.close()
            return ms
    file.close()
    return 0 

def execute(config_file, gpu_id='0', bin=original_caffebin, caffeid='0'):
    logfile = '%s.%s.log'%(config_file, caffeid)
    cmd = '%s time -model=%s/%s -gpu=%s -iterations=2>&%s'%(bin, config_file_home, config_file, gpu_id, logfile)
    if DEBUG:
        print cmd
    niter = 5 
    attempt_num = 5
    nvalid = 0
    m = 0.0; fm=0.0
    for i in range(niter):
        while True:
            os.system(cmd)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()
            ms = get_average_time(logfile)
            forward_ms = get_average_time(logfile, is_total=False)
            if ms and forward_ms:
                break
            if attempt_num > 10:
                ms = 0; forward_ms = 0
                break
            attempt_num+=1
        if ms != 0 and forward_ms != 0:
            m+=ms
            fm+=forward_ms
            nvalid += 1
    if nvalid == 0:
        return 0, 0
    return (m+fm)/nvalid, fm/nvalid 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark script')
    parser.add_argument('-d', '--gpu_id', help='GPU ID used', default='0')
    parser.add_argument('-b', '--original_caffe', help='Benchmark original (0) or optimized (1)', default='0')
    parser.add_argument('-D', '--debug', help='Debug mode', default='0')
    parser.add_argument('-l', '--num_hidden', help='Number of hidden layers', default='2')
    parser.add_argument('-s', '--dataset', help='Dataset: synthetic (sy) or mnist (mn)', default='sy')
    p = parser.parse_args()
    DEBUG = p.debug == '1'
    #hiddens = [[2048, 2048]]
    hiddens = [[4096, 4096]]
    if p.dataset == 'sy':
        batches = [128, 256, 512, 1024, 2048, 4096]
    else:
        batches = [128, 256, 512, 1024, 2048, 4096, 8192, 8192*2]
    bin = original_caffebin 
    caffeid = p.original_caffe
    if p.original_caffe == '1':
        bin = caffebin 
    for hidden in hiddens:
        h1 = hidden[0]
        h2 = hidden[1]
        for batch in batches:
            # Create prototxt
            time.sleep(1)
            config_file_home = '%s/%s' % (settings.REPOS_HOME, '/benchmarks')
            config_file = 'fcn5%s%sl-b%d.prototxt' % (p.dataset, p.num_hidden, batch)
            cmd = 'batch_size=%d num_hidden=%s ./gen-fcn5-%s.sh' % (batch, p.num_hidden, p.dataset)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()
            ms, forward_ms = execute(config_file, p.gpu_id, bin, caffeid)
            print ','.join([str(p.num_hidden), str(batch), str(ms), str(forward_ms)])
