import os
from q1_segmentation import train_and_save_q1_artifacts
from data_handling import load_english_data

base = os.path.dirname(os.path.abspath(__file__))
train_sents, _, _ = load_english_data(split_ratio=0.8)
train_and_save_q1_artifacts(train_sents, os.path.join(base, 'q1_artifacts.pkl'))
