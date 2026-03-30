if [ $1 == "start" ];  
then
    cd ~/pigeonator

    # activate virtual env

    source env/bin/activate

    pgn deter --slack 200 --effector sound \
              --sound_path sounds/knocking_1.mp3 \
                --minimum_number_detections 5 \
                --motion_th_area 3_000 \
                --bh_server https://192.168.0.150/api \
                --bh_user USER \
                --bh_password PASSWORD \
                --stream_type raspi  "" "./recordings"
fi

if [ $1 == "stop" ];  
then
    pkill -9 -f pigeonator
fi