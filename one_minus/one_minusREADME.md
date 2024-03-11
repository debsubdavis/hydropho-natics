## About
This README contains information on the Yolov8 model, how to annotate the data, run the  individual "One" and "Minus" YOLO models,tune them, and identify the areas of interest using the intersection over Union framework.

## Steps to Execute Analysis

### I. Data Preparations

1. **Create a folder with all the unannotated spectrograms:** 
   Begin by organizing all unannotated spectrograms within a dedicated folder. If you have audio files, process them to convert audio files into spectrograms. The spectrograms used in this project have the following specifications:
   - Linear y-axes (frequency)
   - X-axis is Standard  one minute
   - Varying spectral resolution. 20 Hz (0.05 s) 



2. **Setting up Roboflow:** 
   - Start by referring to the user-friendly guide "[Getting Started with Roboflow](https://blog.roboflow.com/getting-started-with-roboflow/)".
   - Create a free Roboflow account and select the plan appropriate for your needs.
   - Invite collaborators to help you annotate images in your workspace. Once you have invited people to your workspace, you will be able to create a project. Leave the project type as "Object Detection".
   - Upload the unannotated spectrograms into the upload area. Once uploaded, you can start annotating the files.
   - Usage guide can be found in the repository - `Roboflow - Usage Guide.pdf`

3. **Annotation Verification:** 
   Once you have annotated all of your spectrograms, you can verify the correctness of the annotations. Any unannotated spectrogram will be marked as "Not Annotated" on the dashboard you can annotate them later. Save and continue to upload your data.

4. **Dataset Splitting:** 
   You will be asked to choose a dataset split where you can choose the percentage split between Train, test, and validation. You can edit the classes and data and create multiple versions to fit your needs and rename them appropriately.

5. **Model Configuration:** 
- For the "Minus" model, retain all the classes except WEC and store the data. 
- For the "One" model include all the sounds you are interested in, including WEC noises, and alter the labels to "sound" from their original labels.

6. **Note:** 
   If you have preexisting annotations in a different format, you can upload the images with labels onto Roboflow, and the platform will convert the annotations to bounding boxes. You can further edit the bounding boxes and add more labels if necessary.

7. **Downloading Data:** 
   Once satisfied with the annotations and the labels, download the data to your local system. Specify the format that you want the labels to be (Select "YOLOv8" option in this case). The data is downloaded in the .zip format and is structured as explained below. This structure is the only structure accepted by the Yolov8 model:

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

   **data.yaml file:**
   The data.yaml file can be directly used with the YOLO model. It typically includes details such as:

   ```yaml
   train: ../train/images
   val: ../valid/images
   test: ../test/images
   nc: 6  # Number of classes
   names: ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']  # List of class names

8. Save Data:
   Save the data in the location of your preference. The team saved and referenced data from Google Drive and Azure Blob Storage.

### II. Model Training

### I. Model Template and Data Preparation

1. **Download the Model Template File:** 
   Acquire the model template file `Yolov8_ModelTemplate.ipynb` from the `src/` directory. This file serves as a standalone YOLOv8 model. You have the option to run it locally or on Google Drive, with the team having tested the model on Google Drive.

2. **Data Preparation for the Model:** 
   If you're using your own data, organize the labels and images into the standard YOLO format as mentioned earlier. Alternatively, if you're using Roboflow data export, the folder structure can remain as is. Transfer the data to a storage container of your choice; the team utilized Google Drive for storing training data.

3. **Customization of the Model Template:** 
   Customize the model template file to align with your project requirements. This may involve modifications to the model architecture, input/output configurations, loss functions, and evaluation metrics. The template comes with standard configurations.

4. **One and Minus Models Training:** 
   Train the "One" and "Minus" models separately. Given that the data for both models differs, the corresponding results and performance will vary accordingly.

5. **Model Results Storage:** 
   The model stores results in an arbitrary intermediate folder, encompassing labels, samples, model metrics, confusion matrix, weights, and other outputs essential for assessing model performance. Save the results in a designated folder to facilitate model performance evaluation, with the team relying on mAP50 metric to gauge model efficacy.

### II. Hyperparameter Tuning the Model

1. **Download the Hyperparameter Tuning File:** 
   Obtain the hyperparameter tuning file `Yolov8-HyperparameterTuning.ipynb` from the `src/` directory. This facilitates running a series of hyperparameter tuning experiments on a YOLOv8 model. Similar to the model template, you can run this file locally or on Google Drive.

2. **Configuration of Hyperparameters:** 
   Configure the hyperparameter tuning file to define the range of hyperparameters to be explored during the tuning process. Parameters include epochs, batch size, learning rate, optimizer type, dropout rates, and more.

3. **Execution and Monitoring:** 
   Execute the hyperparameter tuning file, leveraging computing resources from your chosen system (e.g., Azure, AWS, Google Colab). Monitor the progress of hyperparameter optimization and adjust configurations as needed. Tune the model separately for "One" and "Minus" models based on their unique objectives.

4. **Performance Evaluation:** 
   Evaluate the performance of trained models resulting from hyperparameter tuning using predefined evaluation metrics. Compare tuned models against baseline models to gauge improvements, with the team assessing experiments based on mAP50 score.

5. **Saving the Best Iterations:** 
   Preserve the weights of the best models for subsequent use in the Intersection over Union framework. Note: It's imperative to save the weights of "One" and "Minus" models separately for running the prediction framework.

### III. Intersection over Union (IoU) Framework

1. **Objective of the Framework:** 
   The framework aims to facilitate model training and prediction, calculation of IoU by overlapping "Minus" labels on "One" images, identification of areas of interest, and user evaluation to determine the accuracy of sound source identification.

2. **Download the IoU Framework File:** 
   Access the IoU framework file `Yolov8_IntersectionOverUnion_Framework.ipynb` from the `src/` directory. Similar to previous files, this can be run locally or on Google Drive.

3. **Input Data Preparation:** 
   Prepare input data, including images and annotations, in compatible formats for testing.

4. **Running the Main Function:** 
   Execute the `main` function to initiate the object detection pipeline.

5. **Examination of Outputs:** 
   Examine the output folder structure to assess model performance and identify areas of interest for further investigation.


Output Directory:
   - one
      - predict
         - images
         - labels
      - predicted_info.csv
   - minus
      - predict
         - image
         - labels
      - predicted_info.csv
   - overlap
      - (Overlap images or visual representations)
   - areas_of_interest
      - (Areas of interest images and CSV files)

 
The output of the IoU framework follows a structured organization, facilitating the analysis and interpretation of object detection results:

- **Output Directory**: This is the main folder containing all the output generated by the IoU framework.

  - **Model-specific Subfolders (one, minus)**: These subfolders contain results specific to each object detection model used in the pipeline.

    - **Predicted Images and Labels**: Within each model-specific subfolder, there's a "predict" folder containing predicted images and labels. These represent the detected objects overlaid on the original test images.

    - **Combined Predictions (predicted_info.csv)**: CSV files containing combined predictions from multiple object detection models. Each row in the CSV file represents a detection with bounding box coordinates, model types, class labels, and confidence scores.

  - **Overlap Folder**: This folder stores overlapping predictions between different models. It contains images of overlapping predictions.

  - **Areas of Interest Folder**: This folder contains areas of interest identified based on a specified IoU threshold. It includes images and CSV files highlighting regions where model agreement is below the predefined threshold.


## Data Files Created

1. **Area of interest csv** : path - Output Directory/area_of_interest_image_name.csv

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


