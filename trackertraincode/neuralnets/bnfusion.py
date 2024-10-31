from typing import Tuple, Dict, Any
import copy

import torch
import torch.nn as nn
import torch.fx as fx


def _split_name(target : str) -> Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _split_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse_convbn(net : fx.GraphModule):
    '''From https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html'''
    net = copy.deepcopy(net)
    modules = dict(net.named_modules())
    for node in net.graph.nodes:
        # The FX IR contains several types of nodes, which generally represent
        # call sites to modules, functions, or methods. The type of node is
        # determined by `Node.op`.
        if node.op != 'call_module': # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        if type(modules[node.target]) is nn.BatchNorm2d and type(modules[node.args[0].target]) is nn.Conv2d:
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue
            print ("FUSING: ", node.target)
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            # As we've folded the batch nor into the conv, we need to replace all uses
            # of the batch norm with the conv.
            node.replace_all_uses_with(node.args[0])
            # Now that all uses of the batch norm have been replaced, we can
            # safely remove the batch norm.
            net.graph.erase_node(node)
    net.graph.lint()
    # After we've modified our graph, we need to recompile our graph in order
    # to keep the generated code in sync.
    net.recompile()
    #print ("FUSION RESULT: ")
    #net.graph.print_tabular()
    return net