import argparse
from ultralytics import YOLO
import torch
import sys

# Parse command-line arguments
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--pretrained', type=str, default="yolov8m.pt", help='Path to pre-trained weights (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=640, help='Train, test image sizes')
    parser.add_argument('--output_dir', type=str, default='runs/train', help='Output folder')
    parser.add_argument('--exp_name', type=str, default='exp_default', help='Experiment name')
    parser.add_argument('--patience', type=int, default=20, help="Patience value")
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial Learning Rate')
    parser.add_argument('--iou', type=float, default=0.3, help='IoU')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    return parser.parse_args()

# Train the model based on the provided options
def train_model(opt):
    print("Training options:", opt)

    device = torch.device('cpu')
    model = YOLO().to(device)

    # Train the model
    model.train(
        data=f"{opt.data_dir}/data.yaml",
        pretrained=opt.pretrained,
        batch=opt.batch_size,
        imgsz=opt.img_size,
        epochs=opt.epochs,
        project=opt.output_dir,
        name=opt.exp_name,
        exist_ok=True,
        device='cpu',
        patience=opt.patience,
        optimizer=opt.optimizer,
        lr0=opt.lr0,
        iou=opt.iou,
        dropout=opt.dropout
    )

if __name__ == '__main__':
    print("Raw arguments: ", sys.argv)
    opt = parse_opt()
    train_model(opt)