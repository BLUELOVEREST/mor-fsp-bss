#!/bin/shell

PROJECT_DIR=$(cd "$(dirname "$0")";pwd)

# python ./vad_test.py ./model_template/vad_dnn.json ./ model_vad_8k_v2 --epoch 5 --hipass 4000 --gpu_index 1 --result_dir ./vad_result_8k_v2 --start_threshold 0.7 --end_threshold 0.6

CSV_FLE=$1
WAV_SCP=$2
VAD_OUT_AUDIO_DIR=$3

if [ $# -ne 3 ];then
    echo "usage : [csv file] [wav.scp] [vad_outputdir]"
    exit 0
fi

if [ ! -e $CSV_FLE ];then
    echo "不存在csv文件"
    exit 0
fi

if [ ! -e $WAV_SCP ];then
    echo "不存在wav.scp文件"
    exit 0
fi

if [ ! -e $VAD_OUT_AUDIO_DIR ];then
    mkdir $VAD_OUT_AUDIO_DIR
fi


# mkdir vad_cut_audio
count=1

while read line_wavscp
    do
        WAV_FILE=`echo $line_wavscp | awk '{ print $2}'`
        while read line_csv
            do
                if [ $count -eq 1 ] 
                    then
                        let count+=1
                        continue
                fi
                wav_name=`echo $WAV_FILE | awk -F '.' '{ print $1}'`
                output_wav="$PROJECT_DIR/$VAD_OUT_AUDIO_DIR/"$count".wav"
                start_time=`echo $line_csv | awk '{ print $2}'`
                duration=`echo $line_csv | awk '{ print $3}'`
                sox $WAV_FILE $output_wav trim $start_time $duration
                result=`python $PROJECT_DIR/jd_shengwen.py $output_wav`
                echo $result
                # echo $result
                if [ $result -eq 1 ]
                    then
                        sed -i "${count}s/$/&master/g" $CSV_FLE
                else
                    sed -i "${count}s/$/&other/g" $CSV_FLE
                fi
                let count+=1
        done < $CSV_FLE
done < $WAV_SCP







