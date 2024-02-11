import argparse
import os
from azureml.core import Run
from ultralytics import YOLO
import torch
# from yolov8.utils.general import increment_path
# from yolov8 import train  # Adjust this import based on your setup

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Mounted path to the dataset')
    parser.add_argument('--pretrained', type=str, default="yolov8n.pt", help='Path to pre-trained weights (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=640, help='Train, test image sizes')
    parser.add_argument('--output_dir', type=str, default='runs/train', help='Output folder')
    parser.add_argument('--exp_name', type=str, default='exp_default', help='Experiment name')
    parser.add_argument('--ml_workspace_dataset', type=str, help='Dataset name from ML Workspace')
    opt = parser.parse_args()
    return opt

def train_model(opt, mounted_path):
    print("printing opt 2")
    print(opt)

    # # Ensure the dataset directory exists
    # if not os.path.exists(opt.data_dir):
    #     raise ValueError("Dataset directory does not exist: " + opt.data_dir)
    
    # Adjust the data path in your YOLOv8 dataset configuration to use opt.data_dir
    # You may need to dynamically update your dataset.yaml to point to the mounted dataset location
    
    # Check if pre-trained weights are specified and exist
    # if opt.weights and not os.path.exists(opt.weights):
    #     raise ValueError(f"Pre-trained weights file does not exist: {opt.weights}")
    
# Force PyTorch to use CPU
    device = torch.device('cpu')
    # Load your YOLO model
    model = YOLO().to(device)

    #model = YOLO()

    # Example training call for local development — you might need to adjust arguments based on your specific training configuration
    # model.train(data=(opt.data_dir+"/data.yaml"),
    #               pretrained=opt.pretrained,
    #           batch=opt.batch_size,  # Batch size
    #           imgsz=opt.img_size,  # Image size
    #           epochs=opt.epochs,  # Number of epochs
    #           project=opt.output_dir,  # Output directory
    #           name=opt.exp_name,  # Name of the run
    #           exist_ok=True,
    #           device='cpu')
    
    # Example training call for local development — you might need to adjust arguments based on your specific training configuration
    model.train(data=(mounted_path+"/data.yaml"),
                pretrained=opt.pretrained,
                batch=opt.batch_size,  # Batch size
                imgsz=opt.img_size,  # Image size
                epochs=opt.epochs,  # Number of epochs
                project=mounted_path,  # Output directory
                name=opt.exp_name,  # Name of the run
                exist_ok=True,
                device='cpu')

    # After training completes, save the model
    # Ensure to modify this based on how YOLOv8 or your custom training script handles saving models
    # For example, the train.run function might already handle saving models based on the 'project' and 'name' parameters

if __name__ == '__main__':
    print("Hello world")
    #Get the Azure ML run context
    run = Run.get_context()
    
    # Assuming the dataset is passed as an argument named 'input_data'
    # The name 'input_data' should match with what you specify when attaching the dataset to the ScriptRunConfig
    #dataset = run.input_datasets['input_data']
    
    
    # # The dataset is a mounted path provided by Azure ML
    # mounted_path = dataset.as_mount()
    
    # Parse options, ensuring to pass the mounted dataset path for data_dir
    opt = parse_opt()
    print("printing opt1")
    print(opt)
    #print(run.experiment.workspace.datasets)
    print("printing datasets keys")
    print(run.experiment.workspace.datasets.keys())

    dataset = run.experiment.workspace.datasets[opt.ml_workspace_dataset]

    mounted_path = dataset.as_mount()
    print("printing mounted_path: ", mounted_path)
    # opt.data_dir = mounted_path  # Update data_dir to use the mounted dataset path
    
    # Start the training process
    train_model(opt, mounted_path)
