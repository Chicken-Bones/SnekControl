import numpy as np
import itertools
import onnx
from onnx import helper, NodeProto, numpy_helper
import caffe2.python.onnx.backend as caffe_backend


class repeatedlist:
    def __init__(self, repeated):
        self.repeated = repeated

    def __enter__(self):
        self.list = list(self.repeated)
        return self.list

    def __exit__(self ,type, value, traceback):
        del self.repeated[:]
        self.repeated.extend(self.list)


def graph_pass(model, func):
    while True:
        applied = False
        for i, n in enumerate(model.graph.node):
            if func(model.graph, i, n):
                applied = True
                break

        if not applied:
            break

        remove_redundant_nodes(model.graph)
        del model.graph.value_info[:]
        model = onnx.utils.polish_model(model)

    return model


def node_with_output(graph, output_name):
    if output_name == "":
        return None

    return next(itertools.chain(
        (n for n in graph.node if n.output[0] == output_name),
        (i for i in graph.input if i.name == output_name)))


def nodes_with_input(graph, input_name):
    return [n for n in graph.node if any(i == input_name for i in n.input)]


def is_output_used(graph, name):
    return any(name in n.input for n in graph.node) or any(o.name == name for o in graph.output)


def replace_input_name(graph, name1, name2):
    for n in graph.node:
        with repeatedlist(n.input) as l:
            for i in range(len(l)):
                if l[i] == name1:
                    l[i] = name2


def value_info_by_name(graph, name):
    return next(x for x in itertools.chain(graph.input, graph.value_info) if x.name == name)


def attr_by_name(node, name):
    return next(a for a in node.attribute if a.name == name)


def get_constant_arr(node):
    assert node.op_type == "Constant"
    return numpy_helper.to_array(attr_by_name(node, "value").t)


def get_info_tensor_shape(value_info):
    return np.array([d.dim_value for d in value_info.type.tensor_type.shape.dim])


def is_constant_node(nodeorvalueinfo):
    return isinstance(nodeorvalueinfo, NodeProto) and nodeorvalueinfo.op_type == "Constant"


def make_constant_node(arr, output=[]):
    return helper.make_node(
        "Constant",
        [],
        output,
        value=helper.make_tensor(
            name="value",
            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
            dims=arr.shape,
            vals=arr.tobytes(),
            raw=True
        ))


def shape_to_constant(graph, i, n):
    if n.op_type != "Shape":
        return False

    input_info = value_info_by_name(graph, n.input[0])
    shape = get_info_tensor_shape(input_info)
    with repeatedlist(graph.node) as l:
        l[i] = make_constant_node(shape, n.output)

    print("%s -> Constant %s" % (n.op_type, n.output))
    return True


def constant_ops(graph, i, n):
    if len(n.input) == 0:
        return False

    input_nodes = [node_with_output(graph, i) for i in n.input]
    if not all(is_constant_node(i) for i in input_nodes):
        return False

    if n.op_type == "Gather":
        data = get_constant_arr(input_nodes[0])
        indices = get_constant_arr(input_nodes[1])
        axis = attr_by_name(n, "axis").i
        result = np.take(data, indices, axis=axis)
        with repeatedlist(graph.node) as l:
            l[i] = make_constant_node(result, n.output)

    elif n.op_type == "Unsqueeze":
        data = get_constant_arr(input_nodes[0])
        axes = attr_by_name(n, "axes").i
        result = np.expand_dims(data, axis=axes)
        with repeatedlist(graph.node) as l:
            l[i] = make_constant_node(result, n.output)

    elif n.op_type == "Concat":
        result = np.concatenate([get_constant_arr(i) for i in input_nodes])
        with repeatedlist(graph.node) as l:
            l[i] = make_constant_node(result, n.output)

    elif n.op_type == "ConstantFill" or n.op_type == "ConstantOfShape":
        shape = get_constant_arr(input_nodes[0])
        del n.input[:]

        if n.op_type == "ConstantFill":
            assert attr_by_name(n, "input_as_shape").i == 1
            #with repeatedlist(n.attribute) as l:
            #    attr_index = next(i for i, a in enumerate(n.attribute) if a.name == "input_as_shape")
            #    l[attr_index] = helper.make_attribute("shape", shape)

        # past here is assumption heavy
        value = attr_by_name(n, "value").f
        result = np.full(shape, value, dtype=np.float32)
        with repeatedlist(graph.node) as l:
            l[i] = make_constant_node(result, n.output)

    elif n.op_type == "Slice":
        input = get_constant_arr(input_nodes[0])
        axes = attr_by_name(n, "axes").ints
        starts = attr_by_name(n, "starts").ints
        ends = attr_by_name(n, "ends").ints
        for a, start, end in zip(axes, starts, ends):
            result = np.take(input, range(start, end), axis=a)

        with repeatedlist(graph.node) as l:
            l[i] = make_constant_node(result, n.output)

    else:
        return False

    print("%s -> Constant %s" % (n.op_type, n.output))
    return True


def initializers_to_constants(graph, *_):
    if len(graph.initializer) == 0:
        return False

    nodes = []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        nodes.append(make_constant_node(arr, [init.name]))
        with repeatedlist(graph.input) as l:
            index = next(i for i, e in enumerate(l) if e.name == init.name)
            del l[index]

        print("Init -> Constant " + init.name)

    del graph.initializer[:]
    with repeatedlist(graph.node) as l:
        l[0:0] = nodes

    return True


def lstm_final_hidden(graph, i, n):
    if n.op_type != "LSTM":
        return False

    def is_gather_last(n):
        if n.op_type != "Gather" or attr_by_name(n, "axis").i != 0:
            return False

        indices_input = node_with_output(graph, n.input[1])
        if not is_constant_node(indices_input):
            return False
        indices = get_constant_arr(indices_input)
        return indices == -1


    def is_squeeze(n, axes):
        return n.op_type == "Squeeze" and attr_by_name(n, "axes").ints == axes

    changed = False
    for node1 in [n for n in nodes_with_input(graph, n.output[0]) if is_squeeze(n, [1])]:
        for node2 in [n for n in nodes_with_input(graph, node1.output[0]) if is_gather_last(n)]:
            replace_input_name(graph, node2.output[0], n.output[1])
            print("LSTM Squeeze %s Gather %s -> Hidden %s" % (node1.output, node2.output, n.output[1]))
            changed = True

    return changed


def remove_redundant_nodes(graph):

    while True:
        unused = [i for i, n in enumerate(graph.node) if not any(is_output_used(graph, o) for o in n.output)]
        if len(unused) == 0:
            break

        with repeatedlist(graph.node) as l:
            for i in sorted(unused, reverse=True):
                print("Unused %s" % l[i].output)
                del l[i]

        value_infos = [v for v in graph.value_info if is_output_used(graph, v.name)]
        del graph.value_info[:]
        graph.value_info.extend(value_infos)


def processs_model(model):
    onnx.checker.check_model(model)
    input_shape = get_info_tensor_shape(model.graph.input[0])
    dummy_input = np.linspace(0, 1, num=np.prod(input_shape)).reshape(*input_shape).astype(np.float32)

    result1 = caffe_backend.run_model(model, [dummy_input])[0]

    model = graph_pass(model, initializers_to_constants)
    model = graph_pass(model, shape_to_constant)
    model = graph_pass(model, constant_ops)
    model = graph_pass(model, lstm_final_hidden)

    result2 = caffe_backend.run_model(model, [dummy_input])[0]
    if not np.isclose(result1, result2).all():
        raise Exception("model is not consistent: {} {}".format(result1, result2))

    print('Export Completed Successfully')
    return model


if __name__ == '__main__':
    import sys
    model = onnx.load(sys.argv[1])
    model = processs_model(model)
    onnx.save(model, sys.argv[1].replace(".onnx", "_s.onnx"))
