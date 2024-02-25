from ultralytics import YOLO
model = YOLO()


dataset = run.input_datasets['input_data']

# The dataset is a mounted path provided by Azure ML
mounted_path = dataset.as_mount()

# Parse options, ensuring to pass the mounted dataset path for data_dir
opt = parse_opt()
opt.data_dir = mounted_path


data_dir = "training_dataset/data.yaml"  
output_dir = "/home/azureuser/cloudfiles/code/Users/djsdavis/training_dataset/output"

results = model.train(data=data_dir, 
                      epochs=1,
                      device="cpu",
                      pretrained="yolov8n.yaml",
                      workers=8)