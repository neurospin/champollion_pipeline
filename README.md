# Champollion pipeline - step-by-step tutorial

This tutorial gives the steps to go from a list of T1 MRIs to their corresponding 56 Champollion embeddings.

# 1. Get Started

The first step is to have a dedicated work environment. We advise you to create a new folder before cloning this repository and clone it inside the newly created folder:

```bash
mkdir Champollion
cd Champollion
git clone https://github.com/neurospin/champollion_pipeline.git
```

if you already have a work environment setup you can directly install and initialize the pipeline. To do so, please tun the install script like so:
```bash
python3 install.py --install_dir ..
```

This will install everything it the previously created project folder. You can, of course, use an absolute path.
It will create an architecture like so:
```
champollion_pipeline/ champollion_V1/ deep_folding/ data/
```

The data/ folder is used to store the raw data and the derivatives outputs. You can, of course, use any other folder if your environment is already setup.

In order to run the pipeline, enter the pixi environment:

```bash
source ~/.barshrc # sourcing your newly installed environment
pixi shell
```

# 2. Generate the Morphologist graphs

To generate the Morphologist graphs from the T1 MRIs, you will use morphologist-cli.

## If you want to run each subject serially:

```bash
cd ../data/ 

mkdir TEST_your_last_name # creating your dataset folder, you could just also let everything in data/
mv pipeline_loop_2mm.json TEST_your_last_name/ # reporting the config file in the new folder

# if you copied your data in data/ you can use it like so
LIST_MRI_FILES="/my/path/to/data/TEST08/rawdata/sub-0001.nii.gz /my/path/to/data/TEST08/rawdata/sub-0002.nii.gz"
OUTPUT_PATH="/my/path/to/data/TEST08/" # The program will put the output in $OUTPUT_PATH/derivatives/morphologist-5.2
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0
```

## If you want to run each subject in parallel using soma-workflow:

First, set the maximum number of processors (it will be set once and for all); for this:

- launch soma_work_flow_gui

```bash
soma_workflow_gui
```

Then, under the subwindow "Computing resources", put '24' as the number of CPUs (every user is limited to 24 CPUs on rosette; just do it once). By doing regularly refresh on the "submitted workflows" sub-window, you can follow the advancement of the pipeline.

Then launch the Morphologist command with the option --swf (for soma-workflow)

```bash
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0 --swf
```

This may last around 15-30 minutes.

# 3. Generate the sulcal regions

In $PATH_TO_DATA, you will create the folder deep_folding-2025 in the derivatives, make a symbolic link between the deep_folding datasets folder and this deep_folding-2025 folder (This is necessary as the deep_folding software is looking for a folder, $PATH_TO_DEEP_FOLDING_DATASETS, where all deep_folding datasets lie). :


* "graphs_dir" -> contains the path to the morphologist folder
* "path_to_graph": -> contains the sub-path that, for each subject, permits getting the sulcal graphs
* "path_to_skeleton_with_hull" -> contains the sub-path where to get the skeleton with hull
* "skel_qc_path" -> the path to the QC file if it exists (the format of the QC file is given below)
* "output_dir" -> the output directory where the deep_folding outputs will lie

For example, if your dataset is TESTXX, and you have no QC file, the corresponding parameters in the run_deep_folding script file will look like:

```bash
python3 ./src/run_deep_folding.py /my/path/to/data/TESTXX/ /my/path/to/data/TESTXX/derivatives/ --path_to_graph "t1mri/default_acquisition/default_analysis/folds/3.1" --path_to_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation" --sk_qc_path ""
```

If you have a QC file, it will be a tabular-separated file (for example,  qc.tsv). It will have a minimum of two columns: "participant_id" and "qc" (with an optional third column named "comments" to explain the reason for the rejection). qc will be set to 1 if the subject should be processed, and to 0 otherwise. Here is an example of a QC file:

```bash
participant_id	qc  comments
bvdb            0   Right graph does not exist
sub-1000021     1   
```

It will last 15-30 minutes. To check that everything went smoothly, you can print the subfolders of the crop folder:

```bash
ls /my/path/to/data/TESTXX/derivatives/deep_folding/crops/2mm
```

You should see 28 subfolders like this:

```
F.C.L.p.-subsc.-F.C.L.a.-INSULA.  S.C.-S.Pe.C. S.Or.-S.Olf.
F.C.M.post.-S.p.C.		          S.C.-S.Po.C. S.Pe.C.
F.Coll.-S.Rh.			          S.C.-sylv.   S.Po.C.
F.I.P.-F.I.P.Po.C.inf.		      S.F.inf.-BROCA-S.Pe.C.inf. S.s.P.-S.Pa.int.
F.P.O.-S.Cu.-Sc.Cal.		      S.F.inter.-S.F.sup.        S.T.i.-S.O.T.lat.
Lobule_parietal_sup.		      S.F.int.-F.C.M.ant.		  S.T.i.-S.T.s.-S.T.pol.
OCCIPITAL			              S.F.int.-S.R.			     S.T.s.
S.Call.				              S.F.marginal-S.F.inf.ant.	 S.T.s.br.
S.Call.-S.s.P.-S.intraCing.	      S.F.median-S.F.pol.tr.-S.F.sup.
Sc.Cal.-S.Li.			          S.Or.
```

Then exit the pixi environment:

```bash
exit
```

# 4. Generate the embeddings

## 4.1. Generate the dataset config files

We first need to generate the configuration files for each region of the new dataset $DATA (It can be anywhere in the dataset configuration folder: $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset). For this, we will first create a folder called $DATA in the datasets folder of the champollion_V1 configuration, and copy the file 'reference.yaml' (the one in this GitHub) into this folder:

```bash
python3 generate_champollion_config.py my/path/to/data/TESTXX/derivatives/deep_folding/crops/2mm --config_loc my/path/to/data/TESTXX
```

```bash
mkdir -p $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset/julien/$DATA
cp reference.yaml $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset/julien/$DATA/
```

You will now replace in the newly created file reference.yaml all occurrences of TESTXX with $DATA. For example, if DATA was for you equal to "TEST04", then the reference.yaml file will look like:

Example reference.yaml file after substitution of TESTXX by TEST04:

```bash
# @package dataset.REPLACE_DATASET
dataset_name: REPLACE_DATASET
pickle_normal: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEskeleton.pkl
numpy_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEskeleton.npy
subjects_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEskeleton_subject.csv
crop_dir: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEcrops
foldlabel_dir: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDElabels
foldlabel_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDElabel.npy
subjects_foldlabel_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDElabel_subject.csv
distbottom_dir: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEdistbottom
distbottom_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEdistbottom.npy
extremity_dir: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEextremities
extremity_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEextremities.npy
subjects_extremity_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEextremities_subject.csv
subjects_distbottom_all: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEdistbottom_subject.csv
crop_file_suffix: _cropped_skeleton.nii.gz
pickle_benchmark: 
train_val_csv_file: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEskeleton_subject.csv
subject_labels_file: 
subject_column_name:
cutout_mask_path:
cutin_mask_path: ${dataset_folder}/TEST04/crops/2mm/REPLACE_CROP_NAME/mask/REPLACE_SIDEmask.npy
flip_dataset: False
input_size: (1, REPLACE_SIZEX, REPLACE_SIZEY, REPLACE_SIZEZ)
```

Once you have changed the reference.yaml file, you will generate the dataset config files for all sulcal regions by using create_dataset_config_files.py, which lies in $PATH_TO_PROGRAM/champollion_V1/contrastive/utils. You first need to change inside the file TEXTXX by $DATA (for example TEST04) at the top of the file:

Change TESTXX to $DATA at the top of create_dataset_config_files.py:

```
path = f"{os.getcwd()}/../configs/dataset/julien/TESTXX"
ref_file = f"{path}/reference.yaml"
crop_path = "/neurospin/dico/data/deep_folding/current/datasets/TESTXX/crops/2mm"
```

Then generate the config files:

```bash
cd $PATH_TO_PROGRAM/champollion_V1/contrastive/utils
python3 create_dataset_config_files.py
```

To check that it works, you verify that you get 56 yaml files (like FCLp-subsc-FCLa-INSULA_left.yaml) corresponding to the 56 sulcal regions + the file reference.yaml inside $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset/julien/$DATA

```bash
ls $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset | wc -l
```
It should output 57

## 4.2. Generate the embeddings

Inside the file embeddings_pipeline.py ($PATH_TO_PROGRAM/champollion_V1/contrastive/evaluation/embeddings_pipeline.py):

```
    embeddings_pipeline("/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation",
        dataset_localization="neurospin",
        datasets_root="julien/TESTXX",
        short_name='testxx',
        overwrite=True,
        datasets=["toto"],
        idx_region_evaluation=None,
        labels=["Sex"],
        classifier_name='logistic',
        embeddings=True, embeddings_only=True, use_best_model=False,
        subsets=['full'], epochs=[None], split='random', cv=1,
        splits_basedir='',
        verbose=False) 
```


## 4.2. Putting together the embeddings

By using the code put_together_embeddings_files, which lies in $PATH_TO_PROGRAM/champollion_V1/contrastive/utils, you will put together the embeddings.

At the top of the file, you will change:

```
embeddings_subpath = "testxx_random_embeddings/full_embeddings.csv"
output_path = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/embeddings/TESTXX_embeddings"
```
by substituting textxx with the short_name defined in generate_embeddings (for example testxx), and by giving to output_path the content of $PATH_TO_OUTPUT/TEST04_embeddings (you can also just substitute TESTXX with your data folder, for example TEST04; in this case, it will copy the output to "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/embeddings/TEST04_embeddings")

Check that you have 56 csv files in the output directory.

That's it! You have now the champollion_V1 embedddings....
