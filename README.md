## Installation

#### 1. Prepare Environment
- Install this repo and create conda environment:
    ```
    git clone https://github.com/alter-c/goose.git
    conda create -n goose python=3.10
    conda activate goose
    ```

#### 2. Install Dependencies
- Install requirements.
    ```
    pip install -r requirements.txt
    ```

- Build planners.
    ```
    python3 setup.py
    ```

#### 3. Other Settings
- Set environment variable, add the following to your ~/.bashrc file
    ```
    export GOOSE_ROOT=</path/to/goose>
    export OPENAI_API_KEY=<your secret key>
    ```
    Remember to source your bashrc `source ~/.bashrc`

## Train and Eval
Our main train and eval code are in `learner` directory, so first `cd learner`

Train
```
python3 train_gnn.py gripper --save-file gripper.dt
```
Eval
```
python3 run_gnn.py ../dataset/goose/gripper/domain.pddl ../dataset/goose/gripper/test/gripper-n15.pddl gripper.dt
```

## ICML-2026 experiment
```
python3 experiment.py
```