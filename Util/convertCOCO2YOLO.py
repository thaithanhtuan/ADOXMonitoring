#convertCOCO2YOLO.py


from sahi.utils.coco import Coco




coco = Coco.from_coco_dict_or_path("D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/annotations_2560_02.json", image_dir="D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/annotations_images_2560_02/")
coco.export_as_yolov5(
output_dir="D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/YOLO_HOMONA_ADOX",
train_split_rate=0.85,
disable_symlink=True
)




coco_train = Coco.from_coco_dict_or_path("D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/train_split.json", image_dir="D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/annotations_images_2560_02/")
coco_train.export_as_yolov5(
output_dir="D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/YOLO_HOMONA_ADOX_Train",
train_split_rate = 1.0,
disable_symlink=True
)


coco_val = Coco.from_coco_dict_or_path("D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/val_split.json", image_dir="D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/annotations_images_2560_02/")
coco_val.export_as_yolov5(
output_dir="D:/Jeju/Thai/Dataset/Insect detection/AHMDPDatasetFullYOLO/runs/slice_coco/YOLO_HOMONA_ADOX_Val",
train_split_rate = 0.0,
disable_symlink=True
)
