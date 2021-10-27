# Scripts 

## Automatic variant of running test examples.
If you have data prepared according to ScanNet standard, then you can run **[run.sh](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/run.sh)** script that does all the sequence of automatic processing like downloading and preparing, running training and testing datasets automatically. However, we did not get access to the required ScanNat dataset. Therefore, not all requirements are fulfilled yet.


## Semi-automatic variant of running test examples.
If during the work of **[run.sh](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/run.sh)** script an error occurs, other scripts are not able to launch. If you face this problem, for instance, you do not have ScanNet data or some other; you should run the sequence of the following scripts one by one until you do not face a problem. After fixing it, one should run the remaining scripts.

**[download.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/download.py)** - downloads training and testing datasets and pre-trained models 

**[prepare.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/prepare.py)** - download ScanNet programs, modify them according to python3 format and prepare data for further use 

**[train.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/train.py)** - an example of training a model, where all arguments and keys are used as local variables

**[test.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/test.py)** - an example of testing a model, where all arguments and keys are used as local variables
