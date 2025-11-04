# Champollion pipeline - step-by-step tutorial

This tutorial gives the steps to go from a list of T1 MRIs to their corresponding 56 Champollion embeddings.

# 1. Get Started


In Neurospin, you should first connect to a computer with some computing power, for example rosette:

```
ssh rosette
```
Note that, specifically for Neurospin, you need to connect to rosette in ssh, not with TurboVNC.

You should first get pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

Note that, if you enter again on rosette, you will need to read .bashrc first (it is not done automatically with ssh):

```bash
. ~/.basrhc
```

## 1.1 Define convenience environment variables

To facilitate the description of this README, we define several environment variables (these correspond to the path to data and your program folders,... that are not used directly by the underneath software. They are just here for the convenience of this tutorial):

For convenience, you will define your_user_name, which is the one used as subfolder in /neurospin/dico, and the one used to determine where to put the pixi environmenent You will also choose a test directory TESTXX (change XX to a number that has not been used, that is such that $PATH_TO_TEST_DATA/TESTXX doesn't exist beforehand) where you will put two T1 MRIs. In my case, YOUR_PROGRAM=YY_ZZ/Program (where YY is the number of the experiment, like 01 if it is your first one, and ZZ is the name of the experiment, like "champollion_tutorial").

Please change your_user_name, TESTXX, and YOUR_PROGRAM (YY_ZZ) in the bash lines below, and execute them::

```
export YOUR_PROGRAM=YY_ZZ/Program
export USERNAME=your_user_name # jdupond for example (first letter of first name, followed by family name)
export PATH_TO_PIXI_AIMS=/neurospin/software/$USERNAME/pixi_aims # path to your pixi environment containing morphologist and deep_folding
export PATH_TO_PIXI_CHAMPOLLION=/neurospin/software/$USERNAME/pixi_champollion # path to your pixi environment containing champollion
export PATH_TO_TEST_DATA=/neurospin/dico/data/test # path the directory where lie some T1 MRIs
export DATA=TESTXX # change XX with numbers, you will copy your test data here
export PATH_TO_DATA=$PATH_TO_TEST_DATA/$DATA
export PATH_TO_PROGRAM=/neurospin/dico/$USERNAME/Runs/$YOUR_PROGRAM # where you will put your programs downloaded below
export PATH_TO_DEEP_FOLDING_DATASETS=/neurospin/dico/data/deep_folding/current/datasets
```

## 1.2 Create your environments

You then create two environments, one for morphologist and deep_folding, and another one for Champollion. Indeed, there is a mismatch between the PyTorch version of the two environments.

### Create the pyaims environment

```bash
mkdir -p $PATH_TO_PIXI_AIMS
cd $PATH_TO_PIXI_AIMS
pixi init -c conda-forge -c https://brainvisa.info/neuro-forge
pixi add anatomist morphologist soma-env=0.0 pip ipykernel
```

Enter the pixi environment:

```bash
pixi shell
```

Then, download the different software:

* deep_folding: to tile the cortex in 56 sulcal regions

```bash
cd $PATH_TO_PROGRAM
git clone https://github.com/neurospin/champollion_pipeline.git
git clone https://github.com/neurospin/deep_folding.git
```

Install the software:

```bash
cd deep_folding
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip3 install -e .
python3 -m pytest # To run the deep_folding test
```

Then, exit the environment:

```bash
exit
```

### Create the Champollion environment

```bash
mkdir -p $PATH_TO_PIXI_CHAMPOLLION
cd $PATH_TO_PIXI_CHAMPOLLION
pixi init -c conda-forge
pixi add pip ipykernel
```

Enter the pixi environment:

```bash
pixi shell
```

Then, download the different software:

* champollion_V1: the software used to generate the Deep learning embeddings

```bash
cd $PATH_TO_PROGRAM
git clone https://github.com/neurospin/champollion_V1.git
```

Install the software:

```bash
cd champollion_V1
pip3 install -e .
```

Then, exit the environment:

```bash
exit
```

# 2. Generate the Morphologist graphs

To generate the Morphologist graphs from the T1 MRIs, you will use morphologist-cli.

First, copy the source example TEST_TEMPLATE, present in $PATH_TO_TEST:

```bash
cd $PATH_TO_PIXI_AIMS
pixi shell
cd $PATH_TO_TEST_DATA
rsync -a TEST_TEMPLATE/* $PATH_TO_DATA
```

The folder now $PATH_TO_DATA contains two T1 MRI files in the subfolder rawdata. The following bash command will generate the Morphologist graph from the two T1 MRIs and put them in the subfolder "derivatives/morphologist-5.2". You provide a list of MRIs (LIST_MRI_FILES) separated by spaces. We will now generate the Morphologist outputs. Note that the steps described here generate the "classical" Morphologist output, NOT with the BIDS organization. You can generate them serially or in parallel (choose only one of the two options):

## If you want to run each subject serially:

```bash
cd $PATH_TO_DATA
LIST_MRI_FILES="rawdata/sub-0001.nii.gz rawdata/sub-0002.nii.gz"
OUTPUT_PATH="." # The program will put the output in $OUTPUT_PATH/derivatives/morphologist-5.2
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

In $PATH_TO_DATA, you will create the folder deep_folding-2025 in the derivatives, make a symbolic link between the deep_folding datasets folder and this deep_folding-2025 folder (This is necessary as the deep_folding software is looking for a folder, $PATH_TO_DEEP_FOLDING_DATASETS, where all deep_folding datasets lie). You will copy there the file pipeline_loop_2mm.json (the one in this GitHub):

```bash
cd $PATH_TO_DATA/derivatives
mkdir -p derivatives/deep_folding-2025
cd derivatives/deep_folding-2025
ln -s $PATH_TO_DATA/derivatives/deep_folding-2025 $PATH_TO_DEEP_FOLDING_DATASETS/$DATA 
cp $PATH_TO_PROGRAM/champollion_pipeline/pipeline_loop_2mm.json .
```

We will now adapt the file pipeline_loop_2mm.json (the one in $PATH_TO_DEEP_FOLDING_DATASETS/$DATA) to our dataset. For this, we only need to change 5 lines of this file:

* "graphs_dir" -> contains the path to the morphologist folder
* "path_to_graph": -> contains the sub-path that, for each subject, permits getting the sulcal graphs
* "path_to_skeleton_with_hull" -> contains the sub-path where to get the skeleton with hull
* "skel_qc_path" -> the path to the QC file if it exists (the format of the QC file is given below)
* "output_dir" -> the output directory where the deep_folding outputs will lie

For example, if your dataset is TESTXX, and you have no QC file, the corresponding parameters in the JSON file will look like:

```bash
   "graphs_dir": "/neurospin/dico/data/test/TESTXX/derivatives/morphologist-5.2",
   "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
   "path_to_skeleton_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
   "skel_qc_path": "",
   "output_dir": "/neurospin/dico/data/deep_folding/current/datasets/TESTXX",
```

If you have a QC file, it will be a tabular-separated file (for example,  qc.tsv). It will have a minimum of two columns: "participant_id" and "qc" (with an optional third column named "comments" to explain the reason for the rejection). qc will be set to 1 if the subject should be processed, and to 0 otherwise. Here is an example of a QC file:

```bash
participant_id	qc  comments
bvdb            0   Right graph does not exist
sub-1000021     1   
```

Now, you go to the deep_folding program folder (the one in which yu made the git clone of the deep_folding library) and generate the sulcal regions:

```bash
cd $PATH_TO_PROGRAM/deep_folding/deep_folding/brainvisa
python3 multi_pipelines.py -d $DATA
```

It will last 15-30 minutes. To check that everything went smoothly, you can print the subfolders of the crop folder:

```bash
ls $PATH_TO_DEEP_FOLDING_DATASETS/$DATA/crops/2mm
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

Enter the pixi environment containing the Champollion program:

```bash
cd $PATH_TO_PIXI_CHAMPOLLION
pixi shell
```


We first need to generate the configuration files for each region of the new dataset $DATA (It can be anywhere in the dataset configuration folder: $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset). For this, we will first create a folder called $DATA in the datasets folder of the champollion_V1 configuration, and copy the file 'reference.yaml' (the one in this GitHub) into this folder:

```bash
mkdir -p $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset/julien/$DATA
cp $PATH_TO_PROGRAM/champollion_pipeline/reference.yaml $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset/julien/$DATA/
```

You will now replace in the newly created file reference.yaml ($PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset/julien/$DATA/reference.yaml) all occurrences of TESTXX with $DATA. For example, if DATA was for you equal to "TEST04", then the reference.yaml file will look like:

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
ls $PATH_TO_PROGRAM/champollion_V1/contrastive/configs/dataset/julien/$DATA | wc -l
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
Then generate the embeddings:

```
cd $PATH_TO_PROGRAM/champollion_V1/contrastive
python3 evaluation/generate_embeddings.py
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




