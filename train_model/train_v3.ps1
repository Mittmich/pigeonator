$Env:KMP_DUPLICATE_LIB_OK = "TRUE"
python ../yolov5/train.py --data data/data.yaml --weights yolov5s.pt --epochs 150 --batch 32 --freeze 10