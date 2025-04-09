# Racetrack Programming Expansion For Exploration Bonus Implementation
This repository explores the implementation of an exploration bonus across various algorithms applied to a simple grid world racetrack environment.

## Environment Files:
There are several modules important to constructing the racetrack and implementing the environment.
1. **track_builder.py**
Builds progressively more complicated tracks. These can be adjusted to be more or less complex and additional tracks can be added so long as they are appropriately labeled and called later on.
2. **environment.py** 
Sets up the environment and defines the RaceTrack class
3. **visualizations.py**
Additional visualizations to be used with the .pkl files

## The Dyna-Q Algorithms
In the first part of the experiment, we implement an exploration bonus on a Dyna-Q algorithm. The following files contain the Dyna-Q algorithm and its variations. Once each script is run, the output is saved as a pickle file, which can be loaded into **visualizations.py**
1. **dyna_base.py**
Implements a basic dyna-q algorithm with no exploration bonus.
2. **dynaq_plus.py**
Implements a dyna-q algorithm with an exploration bonus added to the rewards in the planning phase
3. **dynaq_plus_action.py
Implements a dyna-q algorithm with an exploration bonus added in at action selection.

## Monte Carlo Control
In the second part of the experiment, we attempt to implement the exploration bonus in a Monte Carlo algorithm. Similar to the Dyna-Q, all results are saved in a pickle file that can be loaded into **visualizations.py** for further visualization.
1. **montecarlo_op.py**
Implements a basic Monte Carlo off policy algorithm
2. **montecarlo.py**
Implements a basic on policy Monte Carlo algorithm
3. **montecarlo_updates.py**
Implements a Monte Carlo on policy with an exploration bonus applied in Q-value updates.
4. **montecarlo_actionselection.py**
Implements a Monte Carlo on policy with an exploration bonus at action selection.
5. **mc_op_updates.py**
Implements an off policy Monte Carlo algorithm with an exploration bonus in Q-value updates.
6. **mc_op_action.py**
Implements an off policy Monte Carlo algorithm with exploration bonus at action selection

## Report
All findings from this experiment are summarized in the attached pdf.

## How To Run
### Install Dependencies
Make sure you have Python 3 installed. Then you can install the required packages by running:
'''bash
pip install numpy matplotlib seaborn pickle os random


