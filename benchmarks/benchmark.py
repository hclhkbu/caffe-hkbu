import argparse
import sys,os,time
import subprocess
import re
import settings


caffebin = settings.OPTIMIZED_CAFFE_BIN
original_caffebin = settings.ORIGINAL_CAFFE_BIN

config_file_home=settings.CONFIG_FILE_HOME

def get_average_time(filename):
    file = open(filename, "r")
    search_str = 'Average Forward-Backward: '
    for line in file.readlines():
        if re.search(search_str, line):
            start_idx = line.find(search_str)+len(search_str)
            end_idx = line.find('ms.')
            ms = float(line[start_idx:end_idx].strip())
            return ms
    return 0 


def execute(config_file, gpu_id='0', bin=original_caffebin):
    logfile = '%s.log'%config_file
    cmd = '%s time -model=%s/%s -gpu=%s -iterations=16>&%s'%(bin, config_file_home, config_file, gpu_id, logfile)
    #print cmd
    os.system(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    ms = get_average_time(logfile)
    return ms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark script')
    parser.add_argument('-d', '--gpu_id', help='GPU ID used', default='0')
    parser.add_argument('-b', '--original_caffe', help='Benchmark original (0) or optimized (1)', default='0')
    p = parser.parse_args()
    hiddens = [[4096, 4096]]
    batches = [256, 512, 1024, 2048, 4096]
    bin = original_caffebin 
    if p.original_caffe == '1':
        bin = caffebin 
    for hidden in hiddens:
        h1 = hidden[0]
        h2 = hidden[1]
        for batch in batches:
            # Create prototxt
            #cmd = 'batch_size=%d ./gen-fcn5.sh' % batch
            #process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            #process.wait()
            #config_file = 'fcn5-b%d.prototxt' % batch
            config_file = '%d-%d-b%d.prototxt' % (h1, h2, batch)
            ms = execute(config_file, p.gpu_id, bin)
            print ','.join([str(batch), str(h1), str(h2), str(ms/1000)])
