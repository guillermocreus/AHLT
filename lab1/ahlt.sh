#!/bin/bash

python3 extract-features.py data/train > data/features/train.feat
python3 extract-features.py data/devel > data/features/devel.feat
python3 extract-features.py data/test > data/features/test.feat

# cat data/features/train.feat | cut -f5- | grep -v ^$ > data/features/train.mem.feat
# cat data/features/devel.feat | cut -f5- | grep -v ^$ > data/features/devel.mem.feat
# cat data/features/test.feat | cut -f5- | grep -v ^$ > data/features/test.mem.feat


python3 train-crf.py model.crf < data/features/train.feat
# ./megam-64.opt -quiet -nc -nobias multiclass data/features/train.mem.feat > model.mem


python3 predict.py model.crf < data/features/train.feat > results/train.out
python3 predict.py model.crf < data/features/devel.feat > results/devel.out
python3 predict.py model.crf < data/features/test.feat > results/test.out


# python3 predict.py model.mem < data/features/train.mem.feat > results/train.mem.out
# python3 predict.py model.mem < data/features/devel.mem.feat > results/devel.mem.out
# python3 predict.py model.mem < data/features/test.mem.feat > results/test.mem.out


echo -e "----- TRAIN (CRF) ----- \n\n"
python3 evaluator.py NER data/train results/train.out
echo -e "\n\n\n"

echo -e "----- DEVEL (CRF) ----- \n\n" 
python3 evaluator.py NER data/devel results/devel.out
echo -e "\n\n\n"

echo -e "----- TEST (CRF) ----- \n\n" 
python3 evaluator.py NER data/test results/test.out
echo -e "\n\n\n"


# echo -e "----- TRAIN (MEM) ----- \n\n"
# python3 evaluator.py NER data/train results/train.mem.out
# echo -e "\n\n\n"

# echo -e "----- DEVEL (MEM) ----- \n\n" 
# python3 evaluator.py NER data/devel results/devel.mem.out
# echo -e "\n\n\n"

# echo -e "----- TEST (MEM) ----- \n\n" 
# python3 evaluator.py NER data/test results/test.mem.out
# echo -e "\n\n\n"