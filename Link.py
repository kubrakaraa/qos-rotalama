class Link:
    def __init__(self, source, destination, delay, reliability, bandwidth):
        self.source = source
        self.destination = destination
        self.link_delay = delay
        self.link_reliability = reliability
        self.bandwidth = bandwidth

    def __str__(self):
        return f"Link{{{self.source}->{self.destination}, delay={self.link_delay}ms, bw={self.bandwidth}Mbps}}"