import numpy as np
from node import Node
from task import TaskCollection
RANDOM_SEED = 42
MAX_NODE_NUM = 1000

class Env:
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.node_list = []
        self.task_collector = None
        self.task_buffer = []
        # 16ms, limit cpu tickcount
        self.base_time = 16
        self.last_state = None
        self.act_time = 0.

    def _now(self):
        return self.act_time

    def reset(self):
        self.act_time = 0.
        self.node_list = []
        self.task_collector = None
        self.task_buffer = []
        # node_num = np.random.randint(10, MAX_NODE_NUM)
        node_num = 10
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
        for t in range(min(5, len(self.task_buffer))):
            for node in self.node_list:
                state.append(node.get_status() + self.task_buffer[t].get_status())
            #     if node.available(task):
            #         state.append(node.get_status() + t.get_status())
            # state.append([-1., -1.] + t.get_status())

        # padding
        for i in range(len(state), 50):
            state.append([0] * 7)

        for idx, node in enumerate(self.node_list):
            if node.available(self.task_buffer[0]):
                node_idx.append(idx)
        self.info['idx'] = node_idx

        return state, self.info

    def step(self, action):
        selected_task = self.task_buffer[0]
        self.task_buffer.pop(0)
        selected_node = action
        if selected_node not in self.info['idx']:
            selected_node = -1
        if selected_node < 0:
            self.task_buffer.append(selected_task)
            reward = 0.
        else:
            self.node_list[selected_node].append(selected_task)
            reward =  - (self.act_time - selected_task.submit_time) / selected_task.task_duration # + selected_task.task_duration
            # reward /= 1000.
        done = False
        while len(self.task_buffer) == 0:
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
            for t in range(min(5, len(self.task_buffer))):
                for node in self.node_list:
                    state.append(node.get_status() + self.task_buffer[t].get_status())
                #     if node.available(task):
                #         state.append(node.get_status() + t.get_status())
                # state.append([-1., -1.] + t.get_status())

            # padding
            for i in range(len(state), 50):
                state.append([0] * 7)
        
            for idx, node in enumerate(self.node_list):
                if node.available(self.task_buffer[0]):
                    node_idx.append(idx)
            node_idx.append(-1)
            self.info['idx'] = node_idx
        # print(np.array(state).shape)
        if not state:
            print(done)

        return state, reward, done, self.info

if __name__ == "__main__":
    env = Env()
    for i in range(10):
        obs, info = env.reset()
        print(np.array(obs).shape)
    # i = 0
    # while True:
    #     node_idx = info['idx']
    #     selected_idx = np.random.randint(len(node_idx))
    #     obs, rew, done, info = env.step(node_idx[selected_idx])
    #     if rew != 0.:
    #         print(rew, i)
    #         i += 1
    #     # print(obs[0])
    #     if done:
    #         break