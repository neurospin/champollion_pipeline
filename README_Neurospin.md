# Neurospin

In Neurospin, you should first connect to a computer with some computing power, for example rosette:

```
ssh -X rosette
```
Note that, specifically for Neurospin, you need to connect to rosette in ssh, not with TurboVNC.

# Champollion pipeline - step-by-step tutorial

This tutorial gives the steps to go from a list of T1 MRIs to their corresponding 56 Champollion embeddings.

# 1. Get Started

The first step is to have a dedicated work environment. We advise you to create a new folder before cloning this repository and clone it inside the newly created folder. As a first step, we advise creating it in /home_local/$USER (USER is your personal folder, for example, your ID abXXXXXX):

```bash
cd /home_local/$USER
```

You then create your new folder 'Champollion':

```bash
mkdir -p Champollion
cd Champollion
git clone https://github.com/neurospin/champollion_pipeline.git
```

If you have not installed anything, please run the install script as follows (it will install all needed libraries in a pixi environment):
```bash
cd champollion_pipeline
python3 install.py --installation_dir ..
```

This will install everything in the previously created project folder. You can, of course, use an absolute path for the installation_dir.

It will create an architecture like so:
```
champollion_pipeline/ champollion_V1/ deep_folding/ data/
```

The data/ folder is used to store the raw data and the derivatives outputs. You can, of course, use any other folder if your environment is already setup.

To run the pipeline, enter the pixi environment:

```bash
source ~/.bashrc # sourcing your newly installed environment
pixi shell
```

# 2. Generate the Morphologist graphs

To generate the Morphologist graphs from the T1 MRIs, you will use morphologist-cli.

First, copy the source example TEST_TEMPLATE, present in $PATH_TO_TEST:

```bash
cd ../data/
rsync -a /neurospin/dico/data/test/TEST_TEMPLATE/ TEST_your_last_name
```

The data/TEST_your_last_name folder now contains two T1 MRI files in the rawdata subfolder. The following bash command will generate the Morphologist graph from the two T1 MRIs and put them in the subfolder "derivatives/morphologist-5.2". You provide a list of MRIs (LIST_MRI_FILES) separated by spaces. We will now generate the Morphologist outputs. Note that the steps described here generate the "classical" Morphologist output, NOT with the BIDS organization. You can generate them serially or in parallel (choose only one of the two options):

## If you want to run each subject serially:

First go in your data folder:
```bash
cd TEST_your_last_name
```

If you want to run Morphologist serially:

```bash
LIST_MRI_FILES="./rawdata/sub-0001.nii.gz ./rawdata/sub-0002.nii.gz"
OUTPUT_PATH="." # The program will put the output in $OUTPUT_PATH/derivatives/morphologist-5.2
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0
```

## If you want to run each subject in parallel using soma-workflow:

Alternatively, if you want to run Morphologist in parallel, set the maximum number of processors (it will be set once and for all); for this:

- launch soma_work_flow_gui

```bash
soma_workflow_gui
```

Then, under the subwindow "Computing resources", set '24' as the number of CPUs (every user is limited to 24 CPUs on rosette; do this only once). By regularly refreshing the "submitted workflows" sub-window, you can track the pipeline's progress.

Then launch the Morphologist command with the option --swf (for soma-workflow)

```bash
morphologist-cli $LIST_MRI_FILES $OUTPUT_PATH -- --of morphologist-auto-nonoverlap-1.0 --swf
```

This may last around 15-30 minutes.

# 3. Generate the sulcal regions

In TEST_your_last_name, you will create the folder deep_folding-2025 in the derivatives, make a symbolic link between the deep_folding datasets folder and this deep_folding-2025 folder (This is necessary as the deep_folding software is looking for a folder, $PATH_TO_DEEP_FOLDING_DATASETS, where all deep_folding datasets lie). :


* "graphs_dir" -> contains the path to the morphologist folder
* "path_to_graph": -> contains the sub-path that, for each subject, permits getting the sulcal graphs
* "path_to_skeleton_with_hull" -> contains the sub-path where to get the skeleton with hull
* "skel_qc_path" -> the path to the QC file if it exists (the format of the QC file is given below)
* "output_dir" -> the output directory where the deep_folding outputs will lie

For example, if your dataset is TESTXX, and you have no QC file, the corresponding parameters in the run_deep_folding script file will look like (you are supposed to be here in the champollion_pipeline folder):

```bash
python3 ./src/run_deep_folding.py /my/path/to/data/TESTXX/ /my/path/to/data/TESTXX/derivatives/ --path_to_graph "t1mri/default_acquisition/default_analysis/folds/3.1" --path_sk_with_hull "t1mri/default_acquisition/default_analysis/segmentation" --sk_qc_path ""
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
