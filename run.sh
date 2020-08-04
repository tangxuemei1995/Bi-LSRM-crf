#!/bin/bash
# python3 main.py  --mode=train --model_path=model_2 --lr=0.0005 --filter_size=5 --dropout=0.8
# python3 main.py  --mode=train --model_path=model_3 --lr=0.0003 --cnn_filter_size=8 --dropout=0.9
# python3 main.py  --mode=train --model_path=model_4 --lr=0.001 --cnn_filter_size=5 --dropout=0.8
python3 main.py  --mode=test --model_path=model_4 --lr=0.001 --cnn_filter_size=5 --dropout=0.8
# python3 main.py  --mode=test --model_path=model_3 --lr=0.0003 --cnn_filter_size=8 --dropout=0.9
# python3 main.py  --mode=demo --demo_model=1595870544