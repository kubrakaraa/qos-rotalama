class Chromosome:
    def __init__(self, gene_size=None, other=None):
        if other is not None:
            # Copy constructor
            self.genes = list(other.genes)
            self.path = list(other.path)
            self.fitness = other.fitness
            self.total_delay = other.total_delay
            self.total_reliability = other.total_reliability
            self.resource_cost = other.resource_cost
        else:
            # Normal constructor
            self.genes = [False] * gene_size
            self.path = []
            self.fitness = float('inf')
            self.total_delay = 0
            self.total_reliability = 0
            self.resource_cost = 0

    def __str__(self):
        gene_str = ''.join('1' if g else '0' for g in self.genes)
        return f"{gene_str} | fit={self.fitness}"