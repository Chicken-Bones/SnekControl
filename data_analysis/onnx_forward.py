from onnx_simplifier import *
import numpy as np
import onnx
import caffe2.python.onnx.backend as caffe_backend


def create_constant_forward(graph, node):
    assert len(node.output) == 1
    arr = get_constant_arr(node)

    def forward(dict):
        dict[node.output[0]] = arr
    return forward


def create_lstm_forward(graph, node):
    if any(attr.name != "hidden_size" for attr in node.attribute):
        raise Exception("Only hidden_size is supported")

    if len(node.input) >= 8 and node.input[8] != "":
        raise Exception("peepholes not supported")

    if len(node.input) >= 5 and node.input[4] != "":
        raise Exception("sequence_lens not supported")

    def get_weight(i):
        return np.squeeze(get_constant_arr(node_with_output(graph, node.input[i])), 0)

    W, R, B, H0, C0 = [get_weight(i) for i in [1, 2, 3, 5, 6]]
    B = np.add(*np.split(B, 2))

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def forward(dict):
        X = dict[node.input[0]]
        H = H0
        C = C0
        for x in np.split(X, X.shape[0], axis=0):
            x = np.squeeze(x, 0)
            gates = np.dot(x, np.transpose(W)) + np.dot(H, np.transpose(R)) + B
            i, o, f, c = np.split(gates, 4, -1)
            i = sigmoid(i)
            f = sigmoid(f)
            c = np.tanh(c)
            o = sigmoid(o)
            C = f * C + i * c
            H = o * np.tanh(C)

        dict[node.output[1]] = H

    return forward


def create_gemm_forward(graph, node):
    alpha = 1.0
    beta = 1.0
    transA = False
    transB = False
    for attr in node.attribute:
        if attr.name == "alpha": alpha = attr.f
        elif attr.name == "beta": beta = attr.f
        elif attr.name == "transA": transA = attr.i == 1
        elif attr.name == "transB": transB = attr.i == 1

    def forward(dict):
        a, b, c = [dict[i] for i in node.input]
        if transA: a = a.T
        if transB: b = b.T
        dict[node.output[0]] = alpha * np.dot(a, b) + beta * c

    return forward


def create_unsqueeze_forward(graph, node):
    axes = attr_by_name(node, "axes").ints
    if len(axes) != 1:
        raise Exception("multi axis unsqueeze not supported")
    axes = axes[0]

    def forward(dict):
        dict[node.output[0]] = np.expand_dims(dict[node.input[0]], axis=axes)
    return forward


op_type_to_creator = {
    "Constant": create_constant_forward,
    "Gemm": create_gemm_forward,
    "LSTM": create_lstm_forward,
    "Unsqueeze": create_unsqueeze_forward
}


def create_graph_forward(graph):
    assert len(graph.output) == 1
    assert len(graph.input) == 1

    node_ops = [op_type_to_creator[node.op_type](graph, node) for node in graph.node if node.op_type]

    def forward(input):
        dict = {graph.input[0].name: input, "": None}
        for op in node_ops:
            op(dict)

        return dict[graph.output[0].name]

    return forward


if __name__ == '__main__':
    import sys
    model = onnx.load(sys.argv[1])
    input_shape = get_info_tensor_shape(model.graph.input[0])
    dummy_input = np.linspace(0, 1, num=np.prod(input_shape)).reshape(*input_shape).astype(np.float32)
    f = create_graph_forward(model.graph)

    result1 = caffe_backend.run_model(model, [dummy_input])[0]
    result2 = f(dummy_input)
    if not np.isclose(result1, result2).all():
        raise Exception("model is not consistent: {} {}".format(result1, result2))

    print('Forward Test Passed')