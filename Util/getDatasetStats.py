#getDataasetStats.py
from sahi.utils.coco import Coco




# init Coco object
cocotrain = Coco.from_coco_dict_or_path("D:/Jeju/Thai/Dataset/Insect detection\AHMDPDatasetFullYOLO/runs/slice_coco/train_split.json")
cocoval = Coco.from_coco_dict_or_path("D:/Jeju/Thai/Dataset/Insect detection\AHMDPDatasetFullYOLO/runs/slice_coco/val_split.json")
coco = Coco.from_coco_dict_or_path("D:/Jeju/Thai/Dataset/Insect detection\AHMDPDatasetFullYOLO/runs/slice_coco/annotations_2560_02.json")
print("----------------------------cocotrain-----------------------------")
print(cocotrain.stats)
print("----------------------------cocoval-----------------------------")
print(cocoval.stats)
print("----------------------------coco-----------------------------")
print(coco.stats)