from loader import InitialLoader

# TODO: change { to [S

query = "LABELID 1 IMAGEREF [n]{image} XMIN [n]{xmin} YMIN [n]{ymin} XMAX [n]{xmax} YMAX [n]{ymax}"


coco_loader = InitialLoader(
    img_dir_path="/home/lukas/datasets/car_dataset/training_images",
    annot_path="/home/lukas/datasets/car_dataset/train_annotations.csv",
    query=query,
    annot_mode="onefile",
    annot_file_type="csv",
    img_file_type="jpg",
)
coco_data = coco_loader.load()
print(len(coco_data), coco_data[15:20])


