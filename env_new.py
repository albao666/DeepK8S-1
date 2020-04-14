import numpy as np
from node import Node
from task import TaskCollection

from baseline.sjf import SJF
from baseline.packer import Packer

RANDOM_SEED = 42
MAX_NODE_NUM = 1000
NODE_NUM = 5
TASK_NUM = 10

class Env:
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.node_list = []
        self.task_collector = None
        self.task_buffer = []
        # 16ms, limit cpu tickcount
        self.base_time = 16 * 1e-3
        self.last_state = None
        self.act_time = 0.
        self.buffer_time = 0

    def _now(self):
        return self.act_time

    def reset(self):
        self.act_time = 0.
        self.node_list = []
        self.task_collector = None
        self.task_buffer = []
        self.buffer_time = 0
        # node_num = np.random.randint(10, MAX_NODE_NUM)
        node_num = NODE_NUM
        for _ in range(node_num):
            self.node_list.append(Node())
        self.task_collector = TaskCollection()
        while len(self.task_buffer) == 0:
            self.tasks, _ = self.task_collector.get_tasks(self.act_time)
            for task in self.tasks:
                self.task_buffer.append(task)
            self.act_time += self.base_time * np.random.rand()
        # print(len(self.task_buffer))
        state, node_idx = [], []
        self.info = {}
        for t in range(min(TASK_NUM, len(self.task_buffer))):
            for node in self.node_list:
                state.append(node.get_status() + self.task_buffer[t].get_status())
            #     if node.available(task):
            #         state.append(node.get_status() + t.get_status())
            # state.append([-1., -1.] + t.get_status())

        # padding
        for i in range(len(state), NODE_NUM * TASK_NUM):
            state.append([0] * 7)

        for idx, node in enumerate(self.node_list):
            if node.available(self.task_buffer[0]):
                node_idx.append(idx)
        self.info['idx'] = node_idx

        return state, self.info

    def step(self, action, schedule_task = -1):
        if schedule_task == -1:
            self.buffer_time += 1
            selected_task = self.task_buffer.pop(0)
            selected_node = action
            if selected_node not in self.info['idx']:
                selected_node = -1
            if selected_node < 0:
                self.task_buffer.append(selected_task)
                reward = 0.
                # reward = -self.buffer_time * len(self.task_buffer)
            else:
                self.node_list[selected_node].append(selected_task)
                reward =  - (self.act_time - selected_task.submit_time) / selected_task.task_duration # + selected_task.task_duration
                # reward /= 1000.
        else:
            selected_task = self.task_buffer.pop(schedule_task)
            selected_node = action
            try:
                self.node_list[selected_node].append(selected_task)
            except:
                print(len(self.node_list), selected_node)
            reward = -(self.act_time - selected_task.submit_time) / selected_task.task_duration
        done = False
        while len(self.task_buffer) == 0:
            self.buffer_time = 0
            self.tasks, done = self.task_collector.get_tasks(self.act_time)
            if done and not self.tasks:
                break
            for task in self.tasks:
                self.task_buffer.append(task)
            forward_time = self.base_time * np.random.rand()
            for p in self.node_list:
                p.run(forward_time)
            self.act_time += forward_time
            done = False
            # if len(self.task_buffer):
            #     print(len(self.task_buffer))
        state, node_idx = [], []
        self.info = {}
        if not done:
            while True:
                avail = False
                for task in self.task_buffer:
                    if not avail:
                        for node in self.node_list:
                            if node.available(task):
                                avail = True
                                break
                if avail:
                    break
                forward_time = self.base_time * np.random.rand()
                for p in self.node_list:
                    p.run(forward_time)
                self.act_time += forward_time
                
            for t in range(min(TASK_NUM, len(self.task_buffer))):
                for node in self.node_list:
                    state.append(node.get_status() + self.task_buffer[t].get_status())
                #     if node.available(task):
                #         state.append(node.get_status() + t.get_status())
                # state.append([-1., -1.] + t.get_status())

            # padding
            for i in range(len(state), TASK_NUM * NODE_NUM):
                state.append([0] * 7)
        
            for idx, node in enumerate(self.node_list):
                if node.available(self.task_buffer[0]):
                    node_idx.append(idx)
            node_idx.append(-1)
            self.info['idx'] = node_idx
        # print(np.array(state).shape)
        # if not state:
        #     print(done)

        return state, reward, done, self.info

if __name__ == "__main__":
    env = Env()
    obs, info = env.reset()
    # for i in range(10):
    #     obs, info = env.reset()
    #     print(np.array(obs).shape)
    # i = 0
    R = []
    while True:
        # node_idx = info['idx']
        # selected_idx = np.random.randint(len(node_idx))
        # obs, rew, done, info = env.step(node_idx[selected_idx])
        # scheduler = SJF()
        # action, task = scheduler.schedule(obs)
        # obs, rew, done, info = env.step(action, task)
        scheduler = Packer()
        action, task = scheduler.schedule(obs)
        obs, rew, done, info = env.step(action, task)
        # if rew != 0.:
        #     print(rew, i)
        #     i += 1
        # print(obs[0])
        # print(env.act_time)
        if done:
            break
        R.append(rew)
    print(env.task_buffer)
    print(env.act_time)
    print(np.sum(R))