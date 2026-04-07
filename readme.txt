WELCOME TO THE FINAL PROJECT
Frailty Syndrome in Older Adults: Modeling as a Complex Adaptive System
Course: GACS-7401 — Complex Adaptive Systems | The University of Winnipeg

OVERVIEW:
This project models frailty syndrome as a complex adaptive system using a 4-node network (weakness, slowness, low activity, exhaustion). 
The model demonstrates two key properties: (1) emergence — where frailty arises from node interactions through cascade propagation and 
nonlinear threshold effects; and (2) self-organization — where the system converges to stable attractors (robust, pre-frail, frail) 
driven solely by internal dynamics.

============
INSTRUCTIONS
============
An anaconda virtual environment is recommended.

STEP 1: ENVIRONMENT SETUP

    conda create -n demo_test python=3.11.11
    conda activate demo_test

STEP 2: DEPENDENCIES INSTALLATION

    pip install -r requirements.txt

STEP 3: PYTHON VERSION VERIFICATION   # Expected output: Python 3.11.11

    python --version 

STEP 4: DIRECTORY CHANGE               #All commands should be run from the src/ directory
    
    cd src/

STEP 5: EXECUTION
    ### There are several options to choose from

    (1) All experiments (recommended for fully observation)

        python main.py 

    (2) Only system singular demo
        
        python main.py --only demo

    (3) Only emergence experiments
        
        python main.py --only emergence

    (4) Only self-organization experiments
        
        python main.py --only so   


DATASET:
Dataset is stored in dataset/frailty_dataset.csv


OUTPUTS:
Outputs are stored in src/outputs/

    *Network model*
    network structure:      network_structure.png

    *singular state demo*
    frail state:            demo_frail.png
    pre-frail state:        demo_pre_frail.png
    robust state:           demo_robust.png

    *Emergence*
    Emergence #1:           E1_cascade_propagation.png
    Emergence #2:           E2_nonlinear_threshold.png
    Emergence #3:           E3_network_vs_isolated.png

    *Self-organization*
    Self-organization #1:   SO1_attractor_convergence.png
    Self-organization #2:   SO2_resilience_states.png
    Self-organization #3:   SO3_complexity_metrics.png
    


