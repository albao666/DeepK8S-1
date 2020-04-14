CPU_MAX = 5
MEM_MAX = 0.05
NODE_NUM = 5
TASK_NUM = 10
class SJF:
    def __init__(self, ):
        self.name = 'SJF'

    def schedule(self, state):
        assert len(state) == NODE_NUM * TASK_NUM
        ans, mi = -1, float('inf')
        for idx, s in enumerate(state):
            if s[3] > s[0] or s[4] > s[1]:
                continue
            cost = (s[3] / CPU_MAX + s[4] / MEM_MAX) * s[-1]
            if cost < mi:
                ans = idx
                mi = cost
        return ans % NODE_NUM, ans // TASK_NUM