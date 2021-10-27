# Scripts 

## Automatic variant of running test examples.
If you have data prepared according to ScanNet standard, then you can run **[run.sh](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/run.sh)** script that does all the sequence of automatic processing like downloading and preparing, running training and testing datasets automatically. However, we did not get access to the required ScanNat dataset. Therefore, not all requirements are fulfilled yet.

**[download.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/download.py)** - downloads training and testing datasets and pre-trained models 

**[prepare.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/prepare.py)** - download ScanNet programs, modify them according to python3 format and prepare data for further use 

**[train.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/train.py)** - an example of training a model, where all arguments and keys are used as local variables

**[test.py](https://github.com/Nik212/FSE_project_team_6/blob/main/scripts/test.py)** - an example of testing a model, where all arguments and keys are used as local variables
