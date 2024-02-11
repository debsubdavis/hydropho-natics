from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
import argparse

# New way to parse arguments
def parse_opt_v2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Azure URI to the dataset')
    parser.add_argument('--pretrained', type=str, default="yolov8n.pt", help='Path to pre-trained weights (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=640, help='Train, test image sizes')
    parser.add_argument('--output_dir', type=str, default='azureml://datastores/workspaceblobstore/paths/outputs/', help='Output folder')
    parser.add_argument('--exp_name', type=str, default='exp_default', help='Experiment name')
    parser.add_argument('--ml_workspace_dataset', type=str, help='Dataset name from ML Workspace')
    return parser.parse_args()

def main():
    print("Using Azure ML SDK v2")

    # Load the environment variables and create a MLClient
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    # Parse the new options
    opt = parse_opt_v2()
    print("Options:", opt)

    # Accessing dataset
    dataset = ml_client.data.get(opt.ml_workspace_dataset)
    dataset_input = Input(type=AssetTypes.URI_FOLDER, path=dataset.id)

    # Assuming you have a command function or script for training
    # Adjust paths and names as necessary
    train_command = command(
        code="./path_to_your_training_script_or_folder",
        command="python train.py --data_dir ${{inputs.dataset}} --pretrained {} --batch_size {} --epochs {} --img_size {} --output_dir {} --exp_name {}".format(
            opt.pretrained, opt.batch_size, opt.epochs, opt.img_size, opt.output_dir, opt.exp_name),
        inputs={"dataset": dataset_input},
        environment="azureml:YOLOv8_Environment:1",  # Specify your environment
        compute="your-compute-target",  # Specify your compute target
        experiment_name=opt.exp_name,
        display_name=opt.exp_name
    )

    # Submit the command
    run = ml_client.jobs.create_or_update(train_command)
    print(f"Submitted job: {run.name}")

if __name__ == '__main__':
    main()
