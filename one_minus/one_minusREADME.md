# Guide to Using Project Files
This guide provides an overview of how to use the model template file, hyperparameter tuning file, and Intersection over Union (IoU) framework for your project.

## 1. Model Template File - scr/Yolov8_ModelTemplate.ipynb

The model template file serves as a foundational framework for implementing object detection models, specifically YOLOv8, for sound detection in spectrograms.

### Usage Instructions:

1. **Download the Model Template File:** Obtain the model template file from the scr/ directory.

2. **Customization:** Customize the model template file according to your project requirements. This may include modifying model architecture, input/output configurations, loss functions, and evaluation metrics.

3. **Data Preparation:** Roboflow was used to prepare the data. The following steps were followed in Roboflow to create the annotated data:
   - Annotation Import: Upload annotations in formats like COCO JSON, Pascal VOC XML, or LabelMe JSON
   - Image Upload: Upload images for annotation individually or in bulk
   - Annotation Process: Annotate objects by drawing bounding boxes and assigning class labels.
   - Annotation Adjustment: Fine-tune annotations for accuracy and consistency.
   - Exporting Annotations: Export annotated data in formats compatible with YOLOv8, TensorFlow Object Detection API, or PyTorch.
   - Dataset Versioning: The team created seperate data versions for the two models. For the Minus iterations, the fish and mooring noises were removed from the pre-existing annotations. For the "One" iteration we used all labelled noises (except fish and mooring) and labelled them as "sound"

Data Format: Annotations typically include bounding box coordinates and class labels for objects within the images. These annotations are stored in formats compatible with YOLOv8, facilitating model training and deployment.

Test and Train-Validation Split: Roboflow allows users to define test and train-validation splits during the export process. This ensures that the dataset is divided into subsets for training and testing the model.

Folder Structure: The folder structure often includes directories for training, validation, and testing datasets, each containing images and their associated annotation files. This organized structure simplifies the integration of data into machine learning pipelines and frameworks. Roboflow also provides a data.yaml (\hydropho-natics\one_minus\data.yaml) file that can be used to train the model.

4. **Training:** Utilize the model template file to train your object detection models on annotated spectrograms. Follow the instructions provided within the file to initiate training and monitor model performance.

5. **Evaluation:** After training, evaluate the performance of your models using relevant evaluation metrics. We have used the mAP50 metric to judge the performance of the models.

6. **Fine-tuning and Iteration:** Iterate on model training and fine-tuning based on evaluation results and feedback to improve model accuracy and performance. 

You can use this model template to train a stand-alone model. The .ipynb file can be run either locally or on Google Colaboratory. *To run the model on Google Colaboratory, the data paths must be specified explicitly based on where the data is stored. The data storage format should be similar as in the data.yaml file (\hydropho-natics\one_minus\data.yaml)*

## 2. Hyperparameter Tuning File

The hyperparameter tuning file is designed to streamline the process of optimizing model hyperparameters for improved performance and efficiency.

### Usage Instructions:

1. **Download the Hyperparameter Tuning File:** Obtain the hyperparameter tuning file from the src/ directory.

2. **Configuration:** Configure the hyperparameter tuning file to specify the range of hyperparameters to be explored during the tuning process. This may include parameters such as epochs, batch size, learning rate, optimizer type, and dropout rates.

3. **Execution:** Execute the hyperparameter tuning file, leveraging the computing resources in the system of your choice (e.g., Azure, AWS, Google Colab, etc.). Monitor the progress of hyperparameter optimization and adjust configurations as necessary.

4. **Evaluation:** Evaluate the performance of the trained models generated through hyperparameter tuning using evaluation metrics of your choice. Compare the performance of tuned models against baseline models to assess improvements.

The scr/Yolov8-HyperparameterTuning.ipynb notebook can be used to run standalone iterations of hyperparameters locally or on Google colaboratory.
*To run the model on Google Colaboratory, data paths must be specified explicitly based on where the data is stored. The data storage format should be similar as in the data.yaml file (\hydropho-natics\one_minus\data.yaml)*
The team utilized resources from Azure ML to deploy multiple experiments in parallel using clusters. While the Azure setup will likely be unique based on each team's resources and perferred infrastructure, but codes used to run parallel hyperparameter tuning experiments have been included in the 'azure_setup' folder as a starting point if desired.

## 3. Intersection over Union (IoU) Framework

The Intersection over Union (IoU) framework facilitates the analysis of model predictions by calculating the overlap between predicted bounding boxes.

### Description:

#### Model Training and Prediction:
- Train the "ONE" and "MINUS" models using the best-trained weights obtained during the training phase.
- Utilize the models to predict the "ONE" and "MINUS" labels on a set of test images containing spectrograms depicting diverse sound sources.

#### Intersection over Union Calculation:
- Conduct an IoU calculation by overlapping the "MINUS" labels on the "ONE" image.
- Identify areas of agreement or disagreement between the models in identifying sound sources.

#### Identification of Areas of Interest:
- Calculate IoU scores for each bounding box pair to identify areas where the agreement between the "ONE" and "MINUS" models is below a predefined threshold.
- Mark areas of interest where models diverge in their predictions or face challenges in identifying sound sources accurately.

#### User Evaluation and Determination:
- Provide users with bounding boxes delineating areas of interest for visual inspection.
- Enable users to determine whether regions contain noises emitted by WECs or other unidentified sources.

#### Feedback and Analysis:
- User feedback and analysis play a crucial role in refining the models and enhancing their accuracy in identifying sound sources within spectrograms.

### Usage Instructions for the Script:

1. **Download the Script:** Obtain the script file from the designated repository or source provided by the project team.

2. **Input Data:** Prepare input data including images and corresponding annotations in compatible formats.

3. **Run the Script:** Execute the script to perform the following tasks:
    - Train the "ONE" and "MINUS" models using provided data.
    - Predict labels on a set of test images containing spectrograms.
    - Conduct IoU calculations to identify areas of interest and areas of agreement/disagreement between models.
    - Provide visual outputs for user evaluation and determination.

4. **User Interaction:** Interact with the script outputs to analyze areas of interest and provide feedback for model refinement.

5. **Iterative Process:** Iterate through model training, evaluation, and refinement based on user feedback and analysis to enhance model accuracy and performance.

In summary, the IoU framework, coupled with the provided script, offers a comprehensive toolset for evaluating and refining object detection models for sound source detection within spectrograms.

### File Usage

To use the object detection pipeline, follow these steps:

1. **Specify Input Image**: Provide the path to the test image you want to perform object detection on.

2. **Define Output Directory**: Specify the directory path where you want to save the output files and results.

3. **Set Pretrained Weights**: Define pretrained weights for each model type you want to use in the pipeline.

4. **Execute the Pipeline**: Run the `main` function to execute the object detection pipeline.

Here's an example of how you can execute the pipeline:

```python
if __name__ == "__main__":
    main()
   ```
    

### Expected Input

The object detection pipeline expects the following input:

- **Test Image**: Path to the test image for inference.
- **Output Directory**: Directory path for saving output files.
- **Pretrained Weight Dictionary**: Dictionary containing model types as keys and pretrained weights as values.
- **Image Name**: Name of the image file.

## Expected Output

The object detection pipeline generates the following output:

- **Predictions**: Predictions generated from the one and the minus object detection models.
- **Overlapping Bounding Boxes**: Identification of overlapping bounding boxes of the one and the minus models to assess sound detection.
- **Areas of Interest**: Identification of areas of interest based on the specified Intersection over Union (IoU) threshold. The regions in the spectrogram that have little to no overlap can be further explored as areas of interest to identify WEC and other ambient noises.
- **Plots of Areas of Interest**: Visual representation of areas of interest overlaid on the input image.
- **CSV Files**: Files containing areas of interest data in CSV format. This contains the bounding boxes of individual model predictions as well as the bounding boxes of areas of interest with their coordinates.

## How to Interpret Data

Here's how to interpret the data generated by the object detection pipeline:

1. **Combined Predictions**:
   - This is the predictions of the "One" and the "Minus" models generated individually

2. **Overlapping Bounding Boxes**:
   - Identify overlapping bounding boxes between different models (One and Minus).
   - Assess the consistency and alignment of detections across different models.

3. **Areas of Interest**:
   - Analyze areas of interest based on the specified IoU threshold.
   - Focus on areas where multiple models agree on the presence of objects.

4. **Plots of Areas of Interest**:
   - Visualize areas of interest overlaid on the input image.
   - Use the plots to gain insights into the distribution and location of detected objects.

5. **CSV Files**:
   - Use CSV files containing areas of interest data for further analysis and reporting.
   - Explore the data to understand patterns and correlations in object detections.

By following these guidelines, users can effectively interpret the data generated by the object detection pipeline and derive meaningful insights from the object detection process.

## Input and Output Structure


### Model and Hyperparameter tuning Framework

#### Inputs

1. Folder Structure for YOLOv8 Data Storage

To ensure compatibility with YOLOv8, it's essential to adhere to a specific folder structure for storing data and labels. Here's how the data should be organized:

- **Training Data:**
  - Images: Store training images in the following directory: `../train/images`
  - Labels: Corresponding labels for the training images should be stored in the `../train/labels` directory. The label files should have the same name as the corresponding image files.

- **Validation Data:**
  - Images: Store validation images in the directory: `../valid/images`
  - Labels: Similar to training data, labels for validation images should be stored in the `../valid/labels` directory. Ensure that the label files have the same name as their corresponding image files.

- **Testing Data:**
  - Images: For testing purposes, store test images in the directory: `../test/images`
  - No separate label directory is necessary for testing data as testing is typically done to evaluate model performance without ground truth labels.

By organizing your data and labels according to this folder structure, you ensure seamless integration with the YOLOv8 model. This structure facilitates efficient training, validation, and testing processes, allowing for accurate object detection within your dataset.

2. data.yaml file

Additionally, it's recommended to include a `data.yaml` file in the root directory to provide necessary information about the dataset. The `data.yaml` file typically includes details such as:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
nc: 6  # Number of classes
names: ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']  # List of class names
```
**Roboflow Export structures the data in the similar format and creates it's own data.yaml file to run the model. The data must be structured in that way manually if not using Roboflow.**

3. Input - Hyperparameters for Training

When setting hyperparameters for training a machine learning model, it's crucial to understand their significance and the impact they have on the model's performance. Below are the hyperparameters commonly used in training, along with their explanations and probable values:

#### Batch Size (`batch`)

- **Explanation:** Batch size refers to the number of training examples utilized in one iteration of gradient descent. A larger batch size can lead to faster convergence but may require more memory.
- **Probable Values:** Common values include 8, 16, 32, 64, and 128.

#### Optimizer (`optimizer`)

- **Explanation:** The optimizer determines the algorithm used to update the model's parameters during training. Different optimizers have distinct update rules and convergence behaviors.
- **Probable Values:** Popular optimizers include "Adam," "SGD" (Stochastic Gradient Descent), and "RMSProp."

#### Initial Learning Rate (`lr0`)

- **Explanation:** The initial learning rate defines the step size at which the model parameters are updated during training. It influences the rate of convergence and model stability.
- **Probable Values:** Typically set between 0.001 and 0.1, depending on the problem complexity and dataset characteristics.

#### Intersection over Union Threshold (`iou`)

- **Explanation:** The Intersection over Union (IoU) threshold is used for evaluating the accuracy of object detection models. It determines the level of overlap required for a predicted bounding box to be considered correct.
- **Probable Values:** Common values range from 0.1 to 0.5, where higher values indicate stricter criteria for correct detections.

#### Dropout Probability (`dropout`)

- **Explanation:** Dropout is a regularization technique used to prevent overfitting by randomly dropping a fraction of neurons during training. It helps improve the model's generalization performance.
- **Probable Values:** Typically set between 0.1 and 0.5, where higher values indicate more aggressive dropout.

#### Number of Epochs (`epochs`)

- **Explanation:** An epoch represents one complete pass through the entire training dataset. The number of epochs determines how many times the model sees the entire dataset during training.
- **Probable Values:** Generally set based on the convergence behavior of the model, ranging from tens to hundreds of epochs.

#### Patience for Early Stopping (`patience`)

- **Explanation:** Patience is a parameter used in early stopping, a technique to prevent overfitting by stopping training when the model's performance on a validation set stops improving.
- **Probable Values:** Typically set between 5 and 100, representing the number of epochs to wait for improvement before stopping training.

Understanding these hyperparameters and selecting appropriate values based on the specific characteristics of the dataset and model architecture is essential for training effective machine learning models. Adjustments to these parameters may be necessary through iterative experimentation to achieve optimal performance.

#### Outputs

1. Best Model Weights - .pt file saved in the output directory provided.

### Intersection over Union Framework

#### Inputs
- **Test Image**: Path to the test image for inference.
- **Output Directory**: Directory path for saving output files.
- **Pretrained Weight Dictionary**: Dictionary containing model types as keys and pretrained weights as values.
- **Image Name**: Name of the image file.

#### Outputs

Output Folder structure is in the following format


Output Directory
│
├── one
│   ├── predict
│   │   ├── images
│   │   └── labels
│   └── predicted_info.csv
│
├── minus
│   ├── predict
│   │   ├── images
│   │   └── labels
│   └── predicted_info.csv
│
├── overlap
│   └── (Overlap images or visual representations)
│
└── areas_of_interest
    └── (Areas of interest images and CSV files)

where each folder is described as below

## Folder Structure

The output of the IoU framework follows a structured organization, facilitating the analysis and interpretation of object detection results:

- **Output Directory**: This is the main folder containing all the output generated by the IoU framework.

  - **Model-specific Subfolders (one, minus)**: These subfolders contain results specific to each object detection model used in the pipeline.

    - **Predicted Images and Labels**: Within each model-specific subfolder, there's a "predict" folder containing predicted images and labels. These represent the detected objects overlaid on the original test images.

    - **Combined Predictions (predicted_info.csv)**: CSV files containing combined predictions from multiple object detection models. Each row in the CSV file represents a detection with bounding box coordinates, model types, class labels, and confidence scores.

  - **Overlap Folder**: This folder stores overlapping predictions between different models. It contains images of overlapping predictions.

  - **Areas of Interest Folder**: This folder contains areas of interest identified based on a specified IoU threshold. It includes images and CSV files highlighting regions where model agreement is below the predefined threshold.

This structured organization helps users navigate through the output of the IoU framework, facilitating analysis and interpretation of object detection results.

1. Area of interest csv: path - Output Directory/area_of_interest_image_name.csv

| Field                | Explanation                                                      | Sample Value                                         |
|----------------------|------------------------------------------------------------------|------------------------------------------------------|
| box_index            | Index of the bounding box                                        | 2                                                    |
| bounding_box_xywh    | Bounding box coordinates in (x, y, width, height) format        | (750.5858, 1043.4794, 75.79303, 31.340088)          |
| bounding_box_xywhn   | Bounding box coordinates in normalized (x_center, y_center, width_normalized, height_normalized) format | (0.31274408, 0.86956614, 0.03158043, 0.02611674) |
| bounding_box_xyxy    | Bounding box coordinates in (x_min, y_min, x_max, y_max) format | (712.6893, 1027.8093, 788.4823, 1059.1494)         |
| bounding_box_xyxyn   | Bounding box coordinates in normalized (x_center, y_center, width_normalized, height_normalized) format | (0.29695386, 0.8565078, 0.3285343, 0.8826245)    |
| label                | Class label assigned to the detection                           | box_index                                           |
| confidence           | Confidence score associated with the detection                   | 2                                                    |
| box_coordinates      | Combined bounding box coordinates                                | (0.31274408, 0.86956614, 0.03158043, 0.02611674)    |
| model                | Type of model used for detections                               | one                                                  |
| IoU                  | Intersection over Union (IoU) between two bounding boxes        | 0.0                                                  |
| OverlapPercentage    | Percentage of overlap between two bounding boxes                |                                                      |


# Conclusion

In conclusion, the object detection pipeline presented in this guide offers a comprehensive framework for performing object detection tasks using multiple models and analyzing the results effectively. By leveraging the YOLOv8 model and Intersection over Union (IoU) framework, users can achieve accurate and consistent object detection across various scenarios.
