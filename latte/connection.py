from latte.mapping import Mapping

class Connection:
    def __init__(self, source_ens, sink_ens, mapping, reshape, clamp=False):
        self.source = source_ens
        self.sink = sink_ens
        self.mapping = Mapping(mapping, clamp)
        self.reshape = reshape
