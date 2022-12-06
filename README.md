# True Online Sarsa (Lambda) Agent with Linear Function Approximation

-Implemented By
1. Abhishek Jindal (ajinda10@asu.edu)
2. Megha Sudhakaran Nair (mnair5@asu.edu)
3. Akshara Trichambaram (atricham@asu.edu)

This project is developed using python 3.6 and requires the following packages:
These packages can be installed using the pip manager command:
pip install package-name
- numpy
- matplotlib
- pandas
- seaborn

This code is built on top of Project 4 (Reinforcement Learning Agent - CSE 571)
for implementing True Online Lambda Sarsa Agent. Furthermore, 
the converging speed is measured for this agent and compared against
the Approximate Q Learning Agent (implemented in Project 4)

Following is the list of files modified/created 
for implementation of this project.

### Modified Files
- qlearningAgents.py
  - This file is modified to write the score of 
  every episode at the end of every training batch 
  to the output directory.
- featureExtractors.py
  - This file is modified to add a complex feature extractor
  on top of Simple Extractor. This is done by incorporating 
  capsule and non-scared ghost position features.

### Created Files
- q_convergence_test.py
  - This test is used to compute the data files to calculate the converging 
  speed of Approximate Q Learning Agent (implemented in Project 4). It runs the
  agent for 1000 episodes for 50 runs.
- sarsa_convergence_test.py
  - This test is used to compute the data files to calculate the converging 
  speed of True Online Sarsa Lambda Learning Agent. It runs the agent 
  for 1000 episodes for 50 runs.
- multi_env_generator.py
  - This file is used to compute the data files for different pacman environments
  for different sizes and complexities such as:
  Layouts: 'trickyClassic', 'powerClassic', 'capsuleClassic', 'originalClassic', 'mediumGrid', 'smallGrid'
  Feature Extractors: IdentityExtractor', 'SimpleExtractor', 'ComplexExtractor', 'IdentityExtractor', 'ComplexExtractor', 'ComplexExtractor'
  Number of Ghosts: 2, 3, 4, 4, 3, 3
- plot_experiments.py
  - This file is used to plot the experiments based on data files computed in the previous step.
  - Following Experiments have been plotted:
    - True Online Lambda Sarsa Agent Average Reward Value against 
    number of Training Episodes for Simple Extractor (Medium Classic)
    - True Online Lambda Sarsa Agent Average Reward Value against 
    number of Training Episodes for different lambda values (Medium Classic)
    - Converging Behaviour of True Online Lambda Sarsa Agent 
    vs Approximate Q Learning Agent (Medium Classic)
    - True Online Lambda Sarsa Agent Average Reward Value against 
    number of Training Episodes for different environments
    - Box Plot for True Online Lambda Sarsa Agent Average Reward Value 
    for different environments

### Data Files Folder
- ApproximateQAgentSimpleExtractor
- ApproximateSarsaAgentSimpleExtractor
- ApproximateSarsaAgent_lambda5 (for lambda = 0.5)
- ApproximateSarsaAgent_lambda75 (for lambda = 0.75)
- ApproximateSarsaAgent_lambda9 (for lambda = 0.9)
- ApproximateQAgent (for Converging Behaviour)
- ApproximateSarsaAgent (for Converging Behaviour, lambda = 0)
- ApproximateSarsaAgent_lambda9_convergence (lambda = 0.9)
- run_layouts (for different environments)

## Run Instructions
1. Run python q_convergence_test.py
2. Run python sarsa_convergence_test.py
3. Run python multi_env_generator.py
4. Run python plot_experiments.py