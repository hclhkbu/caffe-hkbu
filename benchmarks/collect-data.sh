#python benchmark.py -d 0 -b 0 -s mn >caffe1080mn.txt
#python benchmark.py -d 0 -b 1 -s mn >svmcaffe1080mn.txt
#python benchmark.py -d 1 -b 0 -s mn >caffe980mn.txt
#python benchmark.py -d 1 -b 1 -s mn >svmcaffe980mn.txt
#
#python benchmark.py -d 0 -b 0 -s sy >caffe1080sy.txt
#python benchmark.py -d 0 -b 1 -s sy >svmcaffe1080sy.txt
#python benchmark.py -d 1 -b 0 -s sy >caffe980sy.txt
#python benchmark.py -d 1 -b 1 -s sy >svmcaffe980sy.txt

SUBFIX=total2
LAYER=2l
#python benchmark.py -d 0 -b 0 -l 2 -s mn >caffetitanxmn$LAYER$SUBFIX.txt
#python benchmark.py -d 0 -b 1 -l 2 -s mn >xgcaffetitanxmn$LAYER$SUBFIX.txt
#python benchmark.py -d 0 -b 0 -l 2 -s sy >caffetitanxsy$LAYER$SUBFIX.txt
#python benchmark.py -d 0 -b 1 -l 2 -s sy >xgcaffetitanxsy$LAYER$SUBFIX.txt

python benchmark.py -d 1 -b 0 -l 2 -s mn >caffe1080mn$LAYER$SUBFIX.txt
python benchmark.py -d 1 -b 1 -l 2 -s mn >xgcaffe1080mn$LAYER$SUBFIX.txt
#python benchmark.py -d 1 -b 0 -l 2 -s sy >caffe1080sy$LAYER$SUBFIX.txt
#python benchmark.py -d 1 -b 1 -l 2 -s sy >xgcaffe1080sy$LAYER$SUBFIX.txt

SUBFIX=total
LAYER=3l
python benchmark.py -d 0 -b 0 -l 3 -s mn >caffetitanxmn$LAYER$SUBFIX.txt
python benchmark.py -d 0 -b 1 -l 3 -s mn >xgcaffetitanxmn$LAYER$SUBFIX.txt
#python benchmark.py -d 0 -b 0 -l 3 -s sy >caffetitanxsy$LAYER$SUBFIX.txt
#python benchmark.py -d 0 -b 1 -l 3 -s sy >xgcaffetitanxsy$LAYER$SUBFIX.txt

python benchmark.py -d 1 -b 0 -l 3 -s mn >caffe1080mn$LAYER$SUBFIX.txt
python benchmark.py -d 1 -b 1 -l 3 -s mn >xgcaffe1080mn$LAYER$SUBFIX.txt
#python benchmark.py -d 1 -b 0 -l 3 -s sy >caffe1080sy$LAYER$SUBFIX.txt
#python benchmark.py -d 1 -b 1 -l 3 -s sy >xgcaffe1080sy$LAYER$SUBFIX.txt

SUBFIX=total
LAYER=4l
python benchmark.py -d 0 -b 0 -l 4 -s mn >caffetitanxmn$LAYER$SUBFIX.txt
python benchmark.py -d 0 -b 1 -l 4 -s mn >xgcaffetitanxmn$LAYER$SUBFIX.txt
#python benchmark.py -d 0 -b 0 -l 4 -s sy >caffetitanxsy$LAYER$SUBFIX.txt
#python benchmark.py -d 0 -b 1 -l 4 -s sy >xgcaffetitanxsy$LAYER$SUBFIX.txt

python benchmark.py -d 1 -b 0 -l 4 -s mn >caffe1080mn$LAYER$SUBFIX.txt
python benchmark.py -d 1 -b 1 -l 4 -s mn >xgcaffe1080mn$LAYER$SUBFIX.txt
#python benchmark.py -d 1 -b 0 -l 4 -s sy >caffe1080sy$LAYER$SUBFIX.txt
#python benchmark.py -d 1 -b 1 -l 4 -s sy >xgcaffe1080sy$LAYER$SUBFIX.txt
