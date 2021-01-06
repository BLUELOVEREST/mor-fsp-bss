#!/bin/shell

INPUT_WAVSCP=$1

while read line
    do
        wav=`echo $line | awk '{print $2}'`
        # echo $wav
        python ./jd_shengwen.py $wav >> result_2mic_2point_noise.txt
done < $INPUT_WAVSCP