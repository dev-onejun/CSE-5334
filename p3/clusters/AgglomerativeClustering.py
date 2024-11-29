class AgglomerativeClustering:
    def __init__(self, n_clusters=8, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
