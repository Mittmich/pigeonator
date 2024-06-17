if [ $1 == "start" ];  
then
    cd ~/pigeonator

    # activate virtual env

    source env/bin/activate

    pgn deter --slack 200 --effector sound \
              --sound_path sounds/knocking_1.mp3 \
                --minimum_number_detections 2 \
                --motion_th_area 3_000 \
                --stream_type raspi  "" "./recordings" &> $2 &
fi

if [ $1 == "stop" ];  
then
    pkill -9 -f pigeonator
fi