import ast
import inspect
import latte.util as util
import latte
import ctree.c.nodes as C
import ctypes

class RangeDim(ast.AST):
    _fields = ['child_for']

    def __init__(self, child_for, mapping, ensemble):
        super().__init__()
        self.child_for = child_for
        self.mapping = mapping
        self.ensemble = ensemble


class NeuronTransformer(ast.NodeTransformer):
    def __init__(self, ensemble, connections, buffer_dim_info):
        super().__init__()
        self.ensemble = ensemble
        self.seen_vars = set()
        self.seen_vars2 = set()
        self.tiled_vars = dict()
        self.index_vars = set()
        self.connections = connections
        self.buffer_dim_info = buffer_dim_info
        #self.seen_by_enumerate_dims = already_seen
    def visit(self, node):
        """
        Support replacing nodes with a list of nodes by flattening `body`
        fields.
        """
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_Attribute(self, node):
        """
        A reference `self.field[...]` will be replaced with an array reference
        ensemble_namefield[...] to reflect SOA (struct of array) layout.
        """
        if node.value.id == "self":
            # name is ensemble name + attribute
            name = self.ensemble.name + node.attr

            ndim = self.ensemble.ndim
            offset = 0

            if node.attr.endswith("input"):
                assert isinstance(self.ensemble, latte.ensemble.ActivationEnsemble)
                # ActivationEnsembles support the self.input construct that is
                # equivalent to self.inputs[neuron_index...][0]
                name += "s"
                self.seen_vars.add(name)
                args = [ast.Name("_neuron_index_{}".format(i), ast.Load()) for i in range(ndim + 1)]
                field = name.replace(self.ensemble.name, "")
                if field in self.ensemble.tiling_info:
                    for dim, _ in self.ensemble.tiling_info[field]:
                        dim += 1  # offset for batch dimension
                        args[dim].id += "_outer" 
                        args.append(ast.Name("_neuron_index_{}_inner".format(dim), ast.Load()))
                for i, p in enumerate(self.ensemble.pad):
                    if p[0] > 0:
                        #ANAND: if both tiling and padding on pad has to be divided by tile factor
                        if field in self.ensemble.tiling_info:
                            found = False
                            for dim, factor in self.ensemble.tiling_info[field]:
                                if dim == i:
                                    found = True
                                    pad = p[0]//factor
                                    args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(pad))
                                    pad2 = p[0]%factor
                                    args[i + ndim] = ast.BinOp(args[i + ndim], ast.Add(), ast.Num(pad2))
                            if found == False:
                                args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(p[0]))
                        else:
                             args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(p[0]))
                return ast.Subscript(ast.Name(name, ast.Load()), 
                                     ast.Index(ast.Tuple(args, ast.Load())), node.ctx)

            # mark as seen
            self.seen_vars.add(name)

            if node.attr in ["inputs", "grad_inputs"]:
                # only generate batch index for inputs/grad_inputs because
                # the user will provide rest of indices in expression
                ndim = 1
            elif node.attr in self.ensemble.batch_fields or node.attr in self.ensemble.private_info:
                # increment ndim for fields that have a batch dimension
                ndim += 1
            else:
                # fields that don't have a batch dimension start at an offset 1
                # as 0 is the batch dimension
                offset = 1
            if isinstance(self.ensemble, latte.ensemble.ConcatEnsemble):
                if "inputs" in node.attr or "grad_inputs" in node.attr:
                    ndim = 1
                    offset = 0
                else:
                   ndim = self.ensemble.ndim + 1
                   offset = 0       
            args = []

            # if "grad_" in node.attr and not node.attr.endswith("inputs"):
            # if node.attr in self.ensemble.private_info:
            #     # We privatize these buffers and reduce across threads at the
            #     # end, removing need for synchronization.  This is done by
            #     # adding an outer dimension of size num_threads to the buffer
            #     # args.append(ast.Call(ast.Name("omp_get_thread_num", ast.Load()), [], []))
            #     args.append(ast.Name("_neuron_index_0", ast.Load()))

            # only append dimensions if it is not fixed in self.buffer_dim_info
            # (used for values shared across a dimension)
            #if not isinstance(self.ensemble, latte.ensemble.ConcatEnsemble):



            #if node.attr not in ["value", "grad"]:
            if isinstance(self.ensemble, latte.ensemble.ConcatEnsemble):
                for i in range(ndim):
                    if name not in self.buffer_dim_info or not self.buffer_dim_info[name][i]:
                        if node.attr in ["value", "grad"] and i==1:
                            name2 =  "_output_offset_{}".format(i)
                            while name2 in self.seen_vars2:
                                name2 += str(i) 
                            name3 = "_neuron_index_{}".format(i + offset)
                            if node.attr in self.ensemble.tiling_info:
                                for dim, _ in self.ensemble.tiling_info[node.attr]:
                                    if dim == 0:
                                        name3 += "_outer" 
                            args.append(ast.BinOp(ast.Name(name3, ast.Load()), ast.Add(), ast.Name(name2, ast.Load())))
                            self.seen_vars2.add(name2)     
                        else:
                            args.append(ast.Name("_neuron_index_{}".format(i + offset), ast.Load()))
            else:
                 for i in range(ndim): 
                     if name not in self.buffer_dim_info or not self.buffer_dim_info[name][i]: 
                         args.append(ast.Name("_neuron_index_{}".format(i + offset), ast.Load())) 
        
    


            if node.attr in self.ensemble.scalar_fields and \
                    node.attr in self.ensemble.tiling_info:
                for dim, _ in self.ensemble.tiling_info[node.attr]:
                    if node.attr in self.ensemble.batch_fields or node.attr in self.ensemble.private_info:
                        dim += 1  # offset for batch dimension
                    if not dim == 1 or not isinstance(self.ensemble, latte.ensemble.ConcatEnsemble):
                        idx = args[dim].id
                        args[dim].id = idx + "_outer" 
                        args.append(ast.Name(idx + "_inner", ast.Load()))
                    else:
                        args.append(ast.Name("_neuron_index_1_inner", ast.Load()))
            

            if node.attr in ["value", "grad"]:
                for i, p in enumerate(self.ensemble.pad):
                    if p[0] > 0:
                        #ANAND 10/11/2016: Adding pading update by tiling factor
                        if node.attr in self.ensemble.tiling_info:
                            found = False
                            for dim, factor in self.ensemble.tiling_info[node.attr]: 
                                if dim == i:
                                    found = True    
                                    #factor = self.ensemble.tiling_info[node.attr]
                                    pad = p[0]//factor    
                                    args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(pad))
                                    pad2 = p[0]%factor
                                    args[i + ndim] = ast.BinOp(args[i + ndim], ast.Add(), ast.Num(pad2))
                            if found == False:
                                args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(p[0]))
                        else:            
                             args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(p[0]))

            # return updated indedxing expression
            return ast.Subscript(ast.Name(name, ast.Load()), 
                                 ast.Index(ast.Tuple(args, ast.Load())), node.ctx)
        else:
            raise Exception("Unsupported Attribute node")

    def visit_Subscript(self, node):
        """
        If self.visit(node.value) returns a Subscript, flatten that expression
        into the current subscript node.
        """
        node.value = self.visit(node.value)
        if isinstance(node.value, ast.Subscript):
            value = node.value
            # append or extend the indexing expressions for *current* node to child node
            if isinstance(node.slice.value, (ast.Name, ast.Num)):
                value.slice.value.elts.append(node.slice.value)
            elif isinstance(node.slice.value, ast.Tuple):
                value.slice.value.elts.extend(node.slice.value.elts)
            else:
                raise NotImplementedError(node.slice.value)

            field = value.value.id.replace(self.ensemble.name, '')
            if field in self.ensemble.tiling_info:
                for dim, _ in self.ensemble.tiling_info[field]:
                    # dim += 1  # offset for batch dimension
                    if field in self.ensemble.private_info:
                        dim += 1  # offset for omp_get_thread_num()
                    elif field in self.ensemble.batch_fields:
                        dim += 1
                    index = value.slice.value.elts[dim]
                    if isinstance(index, ast.Name):
                        orig_var = index.id
                        #Anand: modifying below, tiled variable names reflected only if
                        #they are  mapping dims   
                        #if "_neuron_index_" in orig_var:
                        value.slice.value.elts[dim] = ast.Name(orig_var + "_outer", ast.Load())
                        value.slice.value.elts.append(ast.Name(orig_var + "_inner", ast.Load()))
                            
                        self.tiled_vars[orig_var] = dim

                        #else:
                        #    value.slice.value.elts.append(ast.Name(orig_var, ast.Load()))

                    elif isinstance(value.slice.value.elts[dim], ast.Num) and \
                            index.n == 0:
                        value.slice.value.elts.append(ast.Num(0))
                    else:
                        raise NotImplementedError(type(value.slice.value.elts[dim]))
            if "inputs" in value.value.id or "grad_inputs" in value.value.id:
                # Add the input offsets defined by user's mapping for the
                # connection
                ndim = self.ensemble.ndim
                # if isinstance(value.slice.value.elts[1], ast.Num) and value.slice.value.elts[1].n == 0:
                #     value.slice.value.elts.append(ast.Name("_input_offset_1_inner", ast.Load()))
                #if not isinstance(self.ensemble, latte.ensemble.ConcatEnsemble):  
                for i in range(1, ndim + 1):
                    elem = value.slice.value.elts[i]
                    tile = False
                    if field in self.ensemble.tiling_info:
                        for dim, _ in self.ensemble.tiling_info[field]:
                            if dim + 1 == i:
                                tile = True
                    if tile:
                        length = 0
                        if len(self.connections[0].mapping.shape) > i:
                            if len(self.connections[0].mapping.shape[i-1]) == 1:
                                length = 1
                        if length == 0:          
                            value.slice.value.elts[i] = ast.BinOp(elem, ast.Add(), 
                                ast.Name("_input_offset_{}_outer".format(i), ast.Load())) 
                            value.slice.value.elts[i + ndim] = ast.BinOp(value.slice.value.elts[i + ndim], ast.Add(), 
                                ast.Name("_input_offset_{}_inner".format(i), ast.Load())) 
                        else:
                           value.slice.value.elts[i] = ast.Name("_neuron_index_{}_outer".format(i), ast.Load())
                           value.slice.value.elts[i + ndim] =  ast.Name("_neuron_index_{}_inner".format(i), ast.Load())

                    else:
                        value.slice.value.elts[i] = ast.BinOp(elem, ast.Add(), 
                                ast.Name("_input_offset_{}".format(i), ast.Load()))
            return value
        else:
            raise NotImplementedError()
        return node

    counter = -1
    def _gen_unique_variable(self):
        self.counter += 1
        return "__unique_loopvar{}".format(self.counter)

    def visit_For(self, node):
        """
        Converts iteration expressionsinto RangeDim semantic nodes
        """
        index = node.target
        if isinstance(index, ast.Name):
            self.index_vars.add(index.id) 
        _range = node.iter
        if isinstance(_range, ast.Call) and _range.func.id == "eachindex":
            loopvars = []
            for dim in self.connections[0].mapping.shape:
                loopvars.append(self._gen_unique_variable())
            nodes = []
            for index, var in enumerate(loopvars):
                nodes.append(ast.For(
                    ast.Name(var, ast.Store()),
                    ast.Call(ast.Name("range_dim", ast.Load()), [_range.args[0], ast.Num(index)], []),
                    [], []
                ))
            index_expr = ast.Tuple([ast.Name(var, ast.Load()) for var in loopvars], ast.Load())
            nodes[-1].body = [util.replace_name(node.target, index_expr, s) for s in node.body]
            for i in reversed(range(1, len(nodes))):
                nodes[i - 1].body.append(nodes[i])
            return self.visit(nodes[0])
        elif isinstance(_range, ast.Call) and _range.func.id in ["enumerate_dim", "range_dim"]:
            node.body = [self.visit(s) for s in node.body]
            node.body = util.flatten(node.body)
            return RangeDim(node, self.connections[0].mapping, self.connections[0].source)
        else:
            raise NotImplementedError(ast.dump(node))
