python benchmark.py -d 0 -b 0 -s mn >caffe1080mn.txt
python benchmark.py -d 0 -b 1 -s mn >svmcaffe1080mn.txt
python benchmark.py -d 1 -b 0 -s mn >caffe980mn.txt
python benchmark.py -d 1 -b 1 -s mn >svmcaffe980mn.txt

python benchmark.py -d 0 -b 0 -s sy >caffe1080sy.txt
python benchmark.py -d 0 -b 1 -s sy >svmcaffe1080sy.txt
python benchmark.py -d 1 -b 0 -s sy >caffe980sy.txt
python benchmark.py -d 1 -b 1 -s sy >svmcaffe980sy.txt
