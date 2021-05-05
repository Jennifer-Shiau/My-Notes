# Commands

## Tar command
- Compress
```
tar zcvf <filename.tgz> <filename>
```
- Extract
```
tar zxvf <filename.tgz> [--directory <folder>]
```

## Git command
```
git add <filename>
git add <folder/*>
git status
git commit -m "commit message"
git push
```
```
git rm <filename>
git status
git commit -m "commit message"
git push
```
```
git branch
git checkout -b <branch_name>
git branch
git add .
git status
git commit -m "commit_message"
git push --set-upstream origin <branch_name>
```

## Conda environment
- Create environment
```
conda create -n <env_name> python=3.6
```
- Remove environment
```
conda env remove -n <env_name>
```
- Activate/Deactivate environment
```
conda activate <env_name>
conda deactivate
```
- View a list of all environments
```
conda env list
```
- Requirements
```
pip freeze > requirements.txt
pip install -r requirements.txt
```
