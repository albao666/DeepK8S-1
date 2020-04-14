CPU_MAX = 5
MEM_MAX = 0.05
NODE_NUM = 5
TASK_NUM = 10
class Packer:
    def __init__(self, ):
        self.name = 'Packer'

    def schedule(self, state):
        assert len(state) == NODE_NUM * TASK_NUM
        ans, ma = -1, float('-inf')
        for idx, s in enumerate(state):
            if s[3] > s[0] or s[4] > s[1]:
                continue
            cost = (s[3] / CPU_MAX) * (s[0] / CPU_MAX) + (s[4] / MEM_MAX) * (s[1] / MEM_MAX)
            if cost > ma:
                ans = idx
                ma = cost
        return ans % NODE_NUM, ans // TASK_NUM