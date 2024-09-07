$Env:KMP_DUPLICATE_LIB_OK = "TRUE"
python ../yolov5/train.py --data data/data.yaml --weights yolov5n.pt --img 320 --epochs 150 --batch 4 --freeze 10