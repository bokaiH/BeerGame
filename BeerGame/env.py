import gym
from gym import spaces
import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, agentNum, IL, AO, AS, c_h, c_p):
        self.agentNum = agentNum
        self.IL = IL  # Inventory level
        self.OO = 0  # Open order
        self.ASInitial = AS  # Initial arriving shipment
        self.ILInitial = IL  # Initial inventory level
        self.AOInitial = AO  # Initial arriving order
        self.curState = []  # Current state of the game
        self.nextState = []
        self.curReward = 0  # Reward observed at the current step
        self.cumReward = 0  # Cumulative reward
        self.c_h = c_h  # Holding cost
        self.c_p = c_p  # Backorder cost
        self.AS = np.zeros(102)  # Arrived shipment
        self.AO = np.zeros(102)  # Arrived order
        self.action = 0  # Action at time t

    def resetPlayer(self, T):
        self.IL = self.ILInitial
        self.OO = 0
        self.AS = np.zeros(T + 10)
        self.AO = np.zeros(T + 10)
        self.curReward = 0
        self.cumReward = 0
        self.action = []
        self.curObservation = self.getCurState(1)
        self.nextObservation = []

    def recieveItems(self, time):
        self.IL += self.AS[time]
        self.OO -= self.AS[time]

    def actionValue(self, actionList, curTime):
        return max(0, actionList[np.argmax(self.action)] + self.AO[curTime].item())

    def getReward(self):
        self.curReward = (self.c_p * max(0, -self.IL) + self.c_h * max(0, self.IL)) / 200.0
        self.curReward = -self.curReward
        self.cumReward = 0.99 * self.cumReward + self.curReward

    def getCurState(self, t):
        curState = np.array([1 * (self.IL < 0) * self.IL, 1 * (self.IL > 0) * self.IL, self.OO])
        return curState


class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents=4, n_turns_per_game=100):
        super().__init__()
        self.n_agents = n_agents
        self.n_turns = n_turns_per_game

        self.curGame = 1
        self.curTime = 0
        self.m = 10
        self.totIterPlayed = 0
        self.players = self.createAgent()
        self.T = 0
        self.demand = []
        self.orders = []
        self.shipments = []
        self.rewards = []
        self.cur_demand = 0

        self.action_space = spaces.Tuple(tuple([spaces.Discrete(5)] * self.n_agents))

        ob_spaces = {}
        for i in range(self.m):
            ob_spaces[f'current_stock_minus{i}'] = spaces.Discrete(5)
            ob_spaces[f'current_stock_plus{i}'] = spaces.Discrete(5)
            ob_spaces[f'OO{i}'] = spaces.Discrete(5)
            ob_spaces[f'AS{i}'] = spaces.Discrete(5)
            ob_spaces[f'AO{i}'] = spaces.Discrete(5)

        x = [750, 750, 170, 45, 45]
        oob = []
        for _ in range(self.m):
            for ii in range(len(x)):
                oob.append(x[ii])
        self.observation_space = spaces.Tuple(tuple([spaces.MultiDiscrete(oob)] * self.n_agents))

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def createAgent(self):
        return [Agent(i, 0, 0, 0, 2.0, 0.0) for i in range(self.n_agents)]

    def resetGame(self, demand):
        self.demand = demand
        self.curTime = 0
        self.curGame += 1
        self.totIterPlayed += self.T
        self.T = self.n_turns
        self.totalReward = 0

        self.deques = []
        for i in range(self.n_agents):
            deques = {}
            deques[f'current_stock_minus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'current_stock_plus'] = deque([0.0] * self.m, maxlen=self.m)
            deques[f'OO'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AS'] = deque([0] * self.m, maxlen=self.m)
            deques[f'AO'] = deque([0] * self.m, maxlen=self.m)
            self.deques.append(deques)

        for k in range(self.n_agents):
            self.players[k].resetPlayer(self.T)

    def reset(self):
        demand = [random.randint(0, 2) for _ in range(102)]
        self.resetGame(demand)
        obs = [[], [], [], []]

        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))

        for i in range(self.n_agents):
            for j in range(self.m):
                obs[i].append(self.deques[i]['current_stock_minus'][j])
                obs[i].append(self.deques[i]['current_stock_plus'][j])
                obs[i].append(self.deques[i]['OO'][j])

        obs_array = np.array([np.array(row) for row in obs])
        return obs_array

    def step(self, action):
        if len(action) != self.n_agents:
            raise ValueError(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise ValueError(f"You can't order negative amount. Your agents actions are: {action}")

        self.handleAction(action)
        self.next()

        self.orders = action

        for i in range(self.n_agents):
            self.players[i].getReward()
        self.rewards = [1 * self.players[i].curReward for i in range(self.n_agents)]

        if self.curTime == self.T + 1:
            self.done = [True] * self.n_agents
        else:
            self.done = [False] * self.n_agents

        obs = [[], [], [], []]
        for i in range(self.n_agents):
            curState = self.players[i].getCurState(self.curTime)
            self.deques[i]['current_stock_minus'].appendleft(int(curState[0]))
            self.deques[i]['current_stock_plus'].appendleft(int(curState[1]))
            self.deques[i]['OO'].appendleft(int(curState[2]))

        for i in range(self.n_agents):
            for j in range(self.m):
                obs[i].append(self.deques[i]['current_stock_minus'][j])
                obs[i].append(self.deques[i]['current_stock_plus'][j])
                obs[i].append(self.deques[i]['OO'][j])

        obs_array = np.array([np.array(row) for row in obs])
        state = obs_array
        return state, self.rewards, self.done, {}

    def handleAction(self, action):
        leadTime = random.randint(2, 4)
        self.cur_demand = self.demand[self.curTime]
        self.players[0].AO[self.curTime] += self.demand[self.curTime]

        for k in range(self.n_agents):
            self.players[k].action = np.zeros(5)
            self.players[k].action[action[k]] = 1
            self.players[k].OO += self.players[k].actionValue([0, 1, 2, 3, 4], self.curTime)
            leadTime = random.randint(2, 4)
            if k < self.n_agents - 1:
                self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue([0, 1, 2, 3, 4], self.curTime)

    def next(self):
        leadTimeIn = random.randint(2, 4)
        self.players[self.n_agents - 1].AS[self.curTime + leadTimeIn] += self.players[self.n_agents - 1].actionValue([0, 1, 2, 3, 4], self.curTime)

        self.shipments = []
        for k in range(self.n_agents - 1, -1, -1):
            current_IL = max(0, self.players[k].IL)
            current_backorder = max(0, -self.players[k].IL)
            self.players[k].recieveItems(self.curTime)
            possible_shipment = min(current_IL + self.players[k].AS[self.curTime], current_backorder + self.players[k].AO[self.curTime])
            self.shipments.append(possible_shipment)

            if self.players[k].agentNum > 0:
                leadTimeIn = random.randint(2, 4)
                self.players[k - 1].AS[self.curTime + leadTimeIn] += possible_shipment

            self.players[k].IL -= self.players[k].AO[self.curTime]
            self.players[k].getReward()
            self.players[k].nextObservation = self.players[k].getCurState(self.curTime + 1)

        self.curTime += 1

    def render(self, mode='human', log_file_path='render_log.txt', episode=None):
        with open(log_file_path, 'a') as f:
            f.write('\n' + '=' * 20 + '\n')
            if episode is not None:
                f.write(f"Episode:  {episode}\n")
            f.write(f'Turn:     {self.curTime}\n')
            stocks = [p.IL for p in self.players]
            f.write('Stocks:   ' + ", ".join([str(x) for x in stocks]) + '\n')
            f.write(f'Orders:   {self.orders}\n')
            f.write(f'Shipments: {self.shipments}\n')
            f.write(f'Rewards: {self.rewards}\n')
            f.write(f'Customer demand: {self.cur_demand}\n')

            AO = [p.AO[self.curTime] for p in self.players]
            AS = [p.AS[self.curTime] for p in self.players]

            f.write(f'Arrived Order: {AO}\n')
            f.write(f'Arrived Shipment: {AS}\n')

            OO = [p.OO for p in self.players]
            f.write(f'Working Order: {OO}\n')

