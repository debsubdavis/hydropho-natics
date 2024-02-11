import argparse
from ultralytics import YOLO
import torch
import sys

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--pretrained', type=str, default="yolov8n.pt", help='Path to pre-trained weights (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=640, help='Train, test image sizes')
    parser.add_argument('--output_dir', type=str, default='runs/train', help='Output folder')
    parser.add_argument('--exp_name', type=str, default='exp_default', help='Experiment name')
    return parser.parse_args()

def train_model(opt):
    print("Training options:", opt)

    device = torch.device('cpu')
    model = YOLO().to(device)

    # Here, assume 'data.yaml' is available in the --data_dir directory.
    # You might need to adjust the dataset configuration accordingly.
    model.train(
        data=f"{opt.data_dir}/data.yaml",
        pretrained=opt.pretrained,
        batch=opt.batch_size,
        imgsz=opt.img_size,
        epochs=opt.epochs,
        project=opt.output_dir,
        name=opt.exp_name,
        exist_ok=True,
        device='cpu'
    )

if __name__ == '__main__':
    print("Raw arguments: ", sys.argv)
    opt = parse_opt()
    train_model(opt)
