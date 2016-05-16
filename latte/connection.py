from latte.mapping import Mapping

class Connection:
    def __init__(self, source_ens, sink_ens, mapping, reshape):
        self.source = source_ens
        self.sink = sink_ens
        self.mapping = Mapping(mapping)
        self.mapping_inserted = False
        self.reshape = reshape
