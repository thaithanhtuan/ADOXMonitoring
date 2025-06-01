#splitCOCOTrainVal.py
from sahi.utils.coco import Coco
from sahi.utils.file import save_json




# specify coco dataset path
coco_path = "D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/annotations_2560_02.json"
# D:/Jeju/Thai/Dataset/Forest Dataset For Tree Instance Segmentation/Crawl Dataset/TreeUAVCOCO/runs/slice_coco/annotations_images_640_02




# init Coco object
coco = Coco.from_coco_dict_or_path(coco_path)




# split COCO dataset with a 85% train/15% val split
result = coco.split_coco_as_train_val(
train_split_rate=0.85
)




# export train val split files
save_json(result["train_coco"].json, "D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/train_split.json")
save_json(result["val_coco"].json, "D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/val_split.json")
