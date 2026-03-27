# AutonomousDriving-Code
Scripts developed for CNN-vs-ViT-in-Autonomous-Driving-Analysis.
This is a pipeline designed to merge different sessions into a single large dataset to train CNN and ViT architectures for autonomous driving tasks.
For full documentation and results obtained check: [CNN-vs-ViT-in-Autonomous-Driving-Analysis](https://github.com/Oso1718/CNN-vs-ViT-in-Autonomous-Driving-Analysis).



## Quick Start (Summary)

1) Ingestion: Merge raw sessions from data_logs via dataset.py.
2) Data Integrity: Run deduplicate_dataset, clean_dataset, and validate_dataset.
3) Preprocessing: Apply rgb, sobel, or hsv filters in the preprocess/ folder.
4) Benchmarking: Train and compare models using train.py (CNN) or vis-transformer.py (ViT).

## Project Description (General)

1) Project root is the main directory from which the scripts are run. Make sure that when you download it, you place everything inside it to be able to follow this work.

2) data_logs contains all the directories with the information collected from the autonomous vehicle. (In case of creating the CSV for the first time, place all the folders from which the information will be extracted to create the global dataset in the "robot" folder).

data_logs folder: where you put all the files extracted from the robot.

3) The robot folder is where the processed information is stored (unified images, lidar, and image filters {processed folder}).

robot folder: where all the files will be stored and the CSVs are created.

4) Adding new info to the dataset once built. In the "tools" folder, look for the file append_session.py.

/project_root/robot/tools/append_session.py

It is the script that allows you to add more info to a previously built dataset. In this way, it does not have to be rebuilt from scratch every time it is necessary to increase data.

5) /project_root/training has the .py files to:
- train CNN with train.py
- train Visual Transformer with vis_transformer.py
- Compare models with models_comparison.py
- metrics.py and graphics_plots.py are libraries to obtain the training metrics.

# Execution Process

## 1) Creating the dataset

Place all folders in data_logs to create the working directories in the /robot directory.

|- /project_root
    |-data_logs
        |- images
        |- lidar_images
        |- csv

Run the script dataset.py to join all sessions into a single dataset.

```python -m dataset.py```

**1.5) If you want to add only a new dataset. Place the folder in the directory /agregar_datasets with the following structure:**

/agregar_datasets
    |- /carpeta_a_agregar
        |-data_logs
            |- images
            |- lidar_images
            |- csv

run from the main directory the command:
```python append_session.py agregar_datasets/carpeta_a_agregar```

Once executed, it will move all the images and lidar to the folders /robot/imagenes and /robot/lidar as well as add the records to the global CSV.

## 2) Deduplicate images

Once with the finished global dataset (global.csv), run the code deduplicate_dataset.py with the command: python -m deduplicate_dataset
This is to check for any duplicate records.

## 3) Cleaning

Next, run:

```clean_dataset.py.``` 

The function of this code is to be able to check that only records that have an image and a lidar file in the folders where the dataset was gathered are saved. For this, it takes the file global_dedup.csv and checks that these are complete. The goal is to have complete information that is useful for training.

## 4) Validation of the dataset to ensure that it is ready for preprocessing.

Finally, we run the code validate_dataset.py to ensure that our dataset is ready for preprocessing.
```
Graphic workflow:

dataset.py ← creates the general dataset
↓
append_session.py ← adds sessions
↓
deduplicate_dataset.py ← removal of repeated samples
↓
clean_dataset.py ← removal of broken references
↓
validate_dataset.py ← final verification
↓
run preprocess_rgb.py, preprocess_sobel.py, and preprocess_hsv.py
↓
training/
```
## 5) Preprocessing
Once the data cleaning is done, we proceed to apply the preprocessing to the images with the files in the preprocessing folder.
Run:
```
python -m preprocess.preprocess_rgb.py
python -m preprocess.preprocess_sobel.py
python -m preprocess.preprocess_hsv.py
```

With this, we generate the images with the indicated preprocessing for training.

## 6) Training

Finally, we can perform the training of the models; in this project, we have 2: a CNN and a ViT architecture.

- To run the CNN architecture, run the program -> train.py
- To run the ViT architecture, run the program vis-transformer.py

```
python -m train.py --rgb
python -m vis-transformer.py --rgb
```

Choose between the options of rgb, sobel, or hsv according to the folder you want to process...
You can also modify the epochs (20 by default) and the batch_size (32 by default).
