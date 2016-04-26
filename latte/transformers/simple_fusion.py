import ast
import astor

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
                else:
                    new_body.append(statement)
            node.body = [self.visit(s) for s in new_body]

        return node

def simple_fusion(ast):
    return SimpleFusion().visit(ast)
