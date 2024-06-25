from loader import Query, AnnotationLoader

coco_query = Query(
    mode="onefile",
    query_string="LABELID {annotations}[n]{category_id} IMAGEREF {annotations}[n]{image_id} "
                 "XMIN {annotations}[n]{bbox}[0] YMIN {annotations}[n]{bbox}[1] "
                 "WIDTH {annotations}[n]{bbox}[2] HEIGHT {annotations}[n]{bbox}[3] "
                 "CUSTOM {annotations}[n]{iscrowd},{annotations}[n]{area}"
)
coco_loader = AnnotationLoader(
    image_folder_path="",
    annotation_path="/home/lukas/datasets/coco_dataset/instances_val2017.json",
    query="LABELID {annotations}[n]{category_id} IMAGEREF {annotations}[n]{image_id} "
                 "XMIN {annotations}[n]{bbox}[0] YMIN {annotations}[n]{bbox}[1] "
                 "WIDTH {annotations}[n]{bbox}[2] HEIGHT {annotations}[n]{bbox}[3] "
                 "CUSTOM {annotations}[n]{iscrowd},{annotations}[n]{area}",
    annotation_mode="onefile",
    annotation_file_type="json",
    image_file_type="jpg"
)
coco_data = coco_loader.load()
print(len(coco_data), coco_data[1:5])


