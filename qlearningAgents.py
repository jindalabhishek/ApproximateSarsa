# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import csv
import uuid

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math

scores = [['Score']]
run_number = 0
folder = ''


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        else:
            return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        q_values = []
        for action in self.getLegalActions(state):
            val = self.getQValue(state, action)
            q_values.append(val)
        return max(q_values, default=0)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if state == 'TERMINAL_STATE':
            return None
        max_action_val = []
        max_action = []
        for action in self.getLegalActions(state):
            val = self.getQValue(state, action)
            max_action_val.append(val)
            max_action.append(action)
        max_val = max(max_action_val, default=float('inf'))
        if max_val == float('inf'):
            return None
        return max_action[max_action_val.index(max_val)]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        flag = util.flipCoin(self.epsilon)
        if flag:
            legalActions = self.getLegalActions(state)
            if legalActions:
                return random.choice(legalActions)
            else:
                return None
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        maxQAction = self.getPolicy(nextState)
        self.q_values[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
                reward + self.discount * self.getQValue(nextState, maxQAction))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        print(alpha)
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        q_value = 0
        if action is None:
            return q_value
        feature_vector = self.featExtractor.getFeatures(state, action)
        # print(feature_vector)
        for key in feature_vector:
            if key in self.getWeights():
                q_value += self.getWeights()[key] * feature_vector[key]
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        maxQAction = self.getPolicy(nextState)
        feature_vector = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.getQValue(nextState, maxQAction)) - self.getQValue(state, action)
        for key in feature_vector:
            self.weights[key] += self.alpha * difference * feature_vector[key]
        # print(self.weights[(state, action)])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        scores.append([state.getScore()])
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            file_name = folder + '/test-' + str(run_number) + '.csv'
            print('folder:' + folder)
            print(file_name)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(scores)


class ApproximateSarsaAgent(ApproximateQAgent):
    """
       ApproximateSarsaAgent
    """

    def __init__(self, lambda_value=0.9, extractor='IdentityExtractor', **args):
        ApproximateQAgent.__init__(self, **args)
        print(self.alpha)
        print(self.epsilon)
        print(self.discount)
        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()
        self.lambda_value = lambda_value
        self.z_weights = util.Counter()
        self.q_old_value = 0

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        q_value = 0
        if action is None:
            return q_value
        feature_vector = self.featExtractor.getFeatures(state, action)
        # print(feature_vector)
        for key in feature_vector:
            if key in self.getWeights():
                q_value += self.getWeights()[key] * feature_vector[key]
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        feature_vector = self.featExtractor.getFeatures(state, action)
        next_action = self.getAction(nextState)
        q_value = self.getQValue(state, action)
        next_q_value = self.getQValue(nextState, next_action)
        delta = reward + self.discount * next_q_value - q_value
        z_weights_sum = 0

        for key in feature_vector:
            for key_itr in feature_vector:
                z_weights_sum += self.z_weights[key_itr] * feature_vector[key_itr]

            self.z_weights[key] = self.discount * self.lambda_value * self.z_weights[key] + (
                    1 - self.alpha * self.discount * self.lambda_value * z_weights_sum) * feature_vector[key]
            self.weights[key] += self.alpha * self.z_weights[key] * (
                    delta + q_value - self.q_old_value) - self.alpha * feature_vector[key] * (
                                         q_value - self.q_old_value)
        self.q_old_value = next_q_value

    def final(self, state):
        self.q_old_value = 0
        self.z_weights = util.Counter()
        ApproximateQAgent.final(self, state)
        # scores.append([state.getScore()])
        # if self.episodesSoFar == self.numTraining:
        #     file_name = folder + '/test-' + str(run_number) + '.csv'
        #     print('folder:' + folder)
        #     print(file_name)
        #     os.makedirs(os.path.dirname(file_name), exist_ok=True)
        #     with open(file_name, 'w+', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerows(scores)
