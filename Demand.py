class Demand:
    def __init__(self, source, destination, require_bandwidth):
        self.source = source
        self.destination = destination
        self.require_bandwidth = require_bandwidth

    def __str__(self):
        return f"({self.source}→{self.destination}, {self.require_bandwidth} Mbps)"