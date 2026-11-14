class Node:
    def __init__(self, node_id, proc_delay, reliability):
        self.id = node_id
        self.proc_delay = proc_delay
        self.reliability = reliability

    def __str__(self):
        return f"Node{{id={self.id}, delay={self.proc_delay} ms, rel={self.reliability}}}"