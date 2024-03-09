# Hyrdopho-natics

This repository was built to contain the work developed by the Hydropho-natics 2024 MSDS Capstone Team (Emily Creeden, Debbie Davis, Shweta Manjunath, and Saumya Nauni). Our goal was to develop machine learning methods capable of identifying WEC sounds in hydrophone recording and spectrogram data. We explored 2 methodologies and image recognition algorithms, the One Minus Approach (YOLOv8 model) and Audio Embedding Approach (VGGish model). Each modeling apparoach has its own directory containing code to rerun the model and generate shared results, a README file, and a more detailed report on the development of the method.
 
## Repo Structure and Contents

one_minus/: 
#    -scr/
#    -dev/
#    -utils/
    -results_data/: contains output and results images from 2024 Capstone project
        - One Model Predictions/: 2 images, 1 from Fred Olsen and another from Azura, with the One Model predictions
            - 20160222T090004-File-8_20Hz_one_prediction.png
            - 20190221T143004-File-1_20Hz_one_prediction.png
        - Minus Model Predictions/: same as above, but Minus Model predictions
            - 20160222T090004-File-8_20Hz_minus_prediction.png 
            - 20190221T143004-File-1_20Hz_minus_prediction.png
        - One Minus Overlap Predictions/: same as above, but One and Minus Model predictions overlaid            
            - 20160222T090004-File-8_20Hz_overlap_prediction.png
            - 20190221T143004-File-1_20Hz_overlap_prediction.png
        - Mondel Confusion Matricies/: best model classification confusion matricies
            - best_one_model_confusion_matrix_normalized.png
            - best_minus_model_confusion_matrix_normalized.png
vggish/: Directory continaing the code for the VGGish audio embedding approach
    -vggishREADME.md: instructions for how to run the code in the VGGish directory and details on the data outputs
    -vggishReport.md: report on the approach's development, thought process, and learnings 
    -environment.yml: necessary libraries to run the code, usage instructions in vggishREADME.md
    -scr/: contains code for approach
        - requirements.txt: contains libraries for installation (backup to environment.yml)
        - vggish_params.py: sets parameters for VGGish model - see vggish/vggishREADME.md for more info
        - 01_vggish_smoke_test.py: verifies everything is setup correctly to run VGGish
        - 02_audio_resampling.py: resamples input audio files into preferred VGGish format
        - 03_vggish_audio_embeddings.py: runs resampled audio files through VGGish
        - 04_combine_csv_files.py: combines VGGish audio embedding CSVs into a single CSV
        - 05_preprocess_embeddings.py: preprocesses the combined CSV of embeddings
        - 06_tsne_process.py: performs t-SNE on the combined, cleaned embeddings
        - 07_metrics_calculation.py: OPTIONAL calculates silhouette score and density of known annotations per cluster for 2024 Capstone data
        - 08_final_tsne_plot.ipynb: OPTIONAL generate final t-SNE plot for 2024 Capstone data
        - mel_features.py: computes log mel spectrogram features from audio waveforms
        - umap_process.py: reduces combined embedding csv from 128-D to 2-D
        - vggish_input.py: computes input log mel spectrograms & examples for VGGish from audio waveform
        - vggish_postprocess.py: post processes embededings. Not used in 2024 Capstone, necessary for 01_vggish_smoke_test.py
        - vggish_slim.py: defines the VGGish model used to generate AudioSet embedding features
    -results_data/: contains output and results images from 2024 Capstone project
        - Data/: contains the data files for the best VGGish results set (5-second t-SNE reduced embeddings)
            - combined_data_5secs.csv: combined 128-dimensional embeddings for 5 second embeddings
            - tsne_data_5secs.csv: combined 2-dimensional t-SNE reduced results for 5 second embeddings
            - clust_output.csv: matches 5 second embeddings back to the clusters
        - 0.96_sec_heatmap.png: heatmap of known sound cluster density for 0.96 second embeddings
        - 2_sec_heatmap.png: heatmap of known sound cluster density for 2 second embeddings
        - 5_sec_heatmap.png: heatmap of known sound cluster density for 5 second embeddings
#        - cluster of t-SNE reduced 0.96 second embeddings
#        - cluster of t-SNE reduced 2 second embeddings
        - tsne_plot_5secs.png: cluster of t-SNE reduced 5 second embeddings
