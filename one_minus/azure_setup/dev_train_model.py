from ultralytics import YOLO

# Initialize YOLO model
model = YOLO()

# Access the input dataset provided by Azure ML
dataset = run.input_datasets['input_data']

# Mount the dataset path
mounted_path = dataset.as_mount()

# Parse options, using the mounted dataset path for data_dir
opt = parse_opt()
opt.data_dir = mounted_path

# Define dataset and output directories
data_dir = "training_dataset/data.yaml"
output_dir = "/home/azureuser/cloudfiles/code/Users/djsdavis/training_dataset/output"

# Train the YOLO model using specified options
results = model.train(data=data_dir, 
                      epochs=1,
                      device="cpu",
                      pretrained="yolov8n.yaml",
                      workers=8)
