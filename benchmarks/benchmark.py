import argparse
import sys,os,time
import subprocess
import re


caffebin='/home/dl/caffe-hkbu-lr/build-8.0/tools/caffe'
#caffebin='/home/dl/caffe-openblas/build/tools/caffe'
#caffebin='/home/shshi/repos/caffe-optimized/build-8.0/tools/caffe'
#caffebin='/home/shshi/repos/caffe-tnn/build-8.0/tools/caffe'
#caffebin='/home/shshi/repos/caffe/build-8.0/tools/caffe'
#caffebin='/home/dl/caffe-hkbu/build-8.0/tools/caffe'

#config_file_home='/home/dl/caffe-hkbu-lr/benchmarks/2_layer'
#config_file_home='/home/shshi/repos/caffe-optimized/benchmarks/2_layer'
config_file_home='/home/dl/caffe-hkbu-lr/benchmarks'
default_gpu_id=0
#default_gpu_id=1

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


def execute(config_file):
    logfile = '%s.log'%config_file
    cmd = '%s time -model=%s/%s -gpu=%d -iterations=16>&%s'%(caffebin, config_file_home, config_file, default_gpu_id, logfile)
    #print cmd
    #os.system(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    ms = get_average_time(logfile)
    return ms

if __name__ == '__main__':
    #hiddens = [[4096, 4096], [128, 1024]]
    hiddens = [[4096, 4096]]
    batches = [128, 256, 512, 1024, 2048]
    #batches = [1024, 2048, 4096, 8192, 16384, 16384*2]
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
            ms = execute(config_file)
            print ','.join([str(batch), str(h1), str(h2), str(ms/1000)])
