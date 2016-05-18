import ast
import ctree.c.nodes as C
import astor
from copy import deepcopy

class SimpleFusion(ast.NodeTransformer):
    """
    Performs simple fusion of loops when to_source(node.iter) and to_source(node.target) are identical

    Does not perform dependence analysis
    """
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body') and len(node.body) > 1:
            new_body = [node.body[0]]
            for statement in node.body[1:]:
                if isinstance(new_body[-1], ast.For) and \
                        isinstance(statement, ast.For) and \
                        astor.to_source(statement.iter) == astor.to_source(new_body[-1].iter) and \
                        astor.to_source(statement.target) == astor.to_source(new_body[-1].target):
                    new_body[-1].body.extend(statement.body)
                elif isinstance(new_body[-1], C.For) and \
                        isinstance(statement, C.For) and \
                        statement.init.codegen() == new_body[-1].init.codegen() and \
                        statement.incr.codegen() == new_body[-1].incr.codegen() and \
                        statement.test.codegen() == new_body[-1].test.codegen():
                    # new_body[-1].body.extend(statement.body)
                    for stmt in statement.body:
                        add = True
                        for seen in new_body[-1].body:
                            if stmt.codegen() == seen.codegen():
                                add = False
                                break
                        if add:
                            new_body[-1].body.append(stmt)
                else:
                    new_body.append(statement)
            node.body = [self.visit(s) for s in new_body]

        return node

    def visit_FunctionDecl(self, node):
        new_body = [node.defn[0]]
        for statement in node.defn[1:]:
            if isinstance(new_body[-1], ast.For) and \
                    isinstance(statement, ast.For) and \
                    astor.to_source(statement.iter) == astor.to_source(new_body[-1].iter) and \
                    astor.to_source(statement.target) == astor.to_source(new_body[-1].target):
                new_body[-1].body.extend(statement.body)
            elif isinstance(new_body[-1], C.For) and \
                    isinstance(statement, C.For) and \
                    statement.init.codegen() == new_body[-1].init.codegen() and \
                    statement.incr.codegen() == new_body[-1].incr.codegen() and \
                    statement.test.codegen() == new_body[-1].test.codegen():
                if "collapse" in new_body[-1].pragma:
                    candidate_node = deepcopy(new_body[-1])
                    candidate_node.body.extend(statement.body)
                    candidate_node = self.visit(candidate_node)
                    if len(candidate_node.body) == 1:
                        new_body[-1].body.extend(statement.body)
                    else:
                        new_body.append(statement)
                else:
                    new_body[-1].body.extend(statement.body)
            else:
                new_body.append(statement)
        node.defn = [self.visit(s) for s in new_body]
        return node

def simple_fusion(ast):
    return SimpleFusion().visit(ast)
