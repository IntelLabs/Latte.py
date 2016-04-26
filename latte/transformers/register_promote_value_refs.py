import ast
import ctree.c.nodes as C
import ctypes

class RegisterPromoteValueRefs(ast.NodeTransformer):
    def __init__(self, ensemble, direction):
        self.ensemble = ensemble
        self.direction = direction

    def visit_Subscript(self, node):
        if self.direction == "forward":
            target = "value"
        elif self.direction == "backward":
            target = "grad"
        else:
            raise NotImplementedError
        if node.value.id.endswith(target):
            return ast.Name(target, node.ctx)
        return node

    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if isinstance(node.target, ast.Name) and node.target.id == "_neuron_index_{}".format(self.ensemble.ndim):
            if self.direction == "forward":
                to_load = "value"
            elif self.direction == "backward":
                to_load = "grad"
            else:
                raise NotImplementedError
            # Initialize value
            node.body.insert(0, 
                C.Assign(
                    C.SymbolRef(to_load, ctypes.c_float()), 
                    ast.Subscript(ast.Name(self.ensemble.name+to_load, ast.Load()), ast.Index(ast.Tuple([ast.Name("_neuron_index_{}".format(i), ast.Load()) for i in range(self.ensemble.ndim + 1)], ast.Load())), ast.Load()),
                ))
            if to_load == "value":
                # Store value
                node.body.append( 
                    C.Assign(
                        ast.Subscript(ast.Name(self.ensemble.name+to_load, ast.Load()), ast.Index(ast.Tuple([ast.Name("_neuron_index_{}".format(i), ast.Load()) for i in range(self.ensemble.ndim + 1)], ast.Load())), ast.Load()),
                        C.SymbolRef(to_load)
                    ))
        return node

def register_promote_value_refs(ast, ensemble, direction):
    return RegisterPromoteValueRefs(ensemble, direction).visit(ast)
