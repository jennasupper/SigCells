import torch
from onnx import numpy_helper
from .layers import *


class NN(torch.nn.Module):
    def __init__(self, model):
        super(NN, self).__init__()
        self.model = model

        # Available layers
        self.layers = {
            "Relu": Relu,
            "Sigmoid": Sigmoid,
            "Conv": Conv,
            "Gemm": Gemm,
            "MaxPool": MaxPool,
            "AveragePool": AveragePool,
            "ConvTranspose": ConvTranspose,
            "Transpose": Transpose,
            "Shape": Shape,
            "Slice": Slice,
            "Exp": Exp,
            "RandomNormalLike": RandomNormalLike,
            "Flatten": Flatten,
            "ReduceSum": ReduceSum,
            "Unsqueeze": Unsqueeze,
        }
        self.multi_input_layers = {
            "Reshape": Reshape,
            "Resize": Resize,
            "Concat": Concat,
            "Add": Add,
            "Split": Split,
            "BatchNormalization": BatchNormalization,
            "Mul": Mul,
            "Pow": Pow,
            "Div": Div,
        }
        self.non_input_layers = {
            "Constant": Constant,
        }

    def forward(self, input):
        """
        Args:
            input (torch.Tensor or List[torch.Tensor]): input tensor or tensor list
        Returns:
            output (torch.Tensor or List[torch.Tensor]): output tensor or tensor list
        """

        node_output = {}
        for i, input_node in enumerate([self.model.graph.input[0]]):
            if len([self.model.graph.input[0]]) == 1:
                node_output[input_node.name] = input.detach().to(torch.float64)
            else:
                node_output[input_node.name] = input[i].detach().to(torch.float64)

        for tensor in self.model.graph.initializer:
            arr = numpy_helper.to_array(tensor)
            if tensor.data_type == 7:
                arr = torch.tensor(arr, dtype=torch.int64)
            else:
                arr = torch.tensor(arr, dtype=torch.float64)
            node_output[tensor.name] = arr

        with torch.no_grad():
            for node in self.model.graph.node:
                inputs = [
                    node_output[input_name]
                    for input_name in node.input
                    if input_name != ""
                ]
                op_type = node.op_type

                if op_type in self.layers:
                    layer = self.layers[op_type](inputs, node)
                    x = node_output[node.input[0]]
                    outputs = layer.forward(x)
                elif op_type in self.multi_input_layers:
                    layer = self.multi_input_layers[op_type](inputs, node, node_output)
                    outputs = layer.forward()
                elif op_type in self.non_input_layers:
                    layer = self.non_input_layers[op_type](inputs, node)
                    outputs = layer.forward()
                else:
                    raise NotImplementedError(f"Layer {op_type} is not supported.")

                if isinstance(outputs, torch.Tensor) or op_type == "Constant":
                    node_output[node.output[0]] = outputs
                else:
                    for i, output_name in enumerate(node.output):
                        node_output[output_name] = outputs[i]

        self.output_obs = node_output

        outputs = [node_output[output.name] for output in self.model.graph.output]
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def forward_si(self, input, bias, a, b, l, u):
        """
        Args:
            input (torch.Tensor or List[torch.Tensor]): input tensor or tensor list
            bias (torch.Tensor or List[torch.Tensor]): bias tensor or tensor list
            a (torch.Tensor or List[torch.Tensor]): a tensor or tensor list
            b (torch.Tensor or List[torch.Tensor]): b tensor or tensor list
            l (torch.Tensor or List[torch.Tensor]): l tensor or tensor list
            u (torch.Tensor or List[torch.Tensor]): u tensor or tensor list
        Returns:
            x (torch.Tensor or List[torch.Tensor]): output tensor or tensor list
            bias (torch.Tensor or List[torch.Tensor]): output bias tensor or tensor list
            a (torch.Tensor or List[torch.Tensor]): output a tensor or tensor list
            b (torch.Tensor or List[torch.Tensor]): output b tensor or tensor list
            l (torch.Tensor or List[torch.Tensor]): output l tensor or tensor list
            u (torch.Tensor or List[torch.Tensor]): output u tensor or tensor list
        """

        node_output = {}
        for i, input_node in enumerate([self.model.graph.input[0]]):
            if len([self.model.graph.input[0]]) == 1:
                node_output[input_node.name] = input.detach().to(torch.float64)
            else:
                node_output[input_node.name] = input[i].detach().to(torch.float64)

        node_output_si = {}
        for i, input_node in enumerate([self.model.graph.input[0]]):
            if len([self.model.graph.input[0]]) == 1:
                node_output_si[input_node.name] = (
                    bias.detach().to(torch.float64),
                    a.detach().to(torch.float64),
                    b.detach().to(torch.float64),
                    l.detach().to(torch.float64),
                    u.detach().to(torch.float64),
                )
            else:
                if all(input_si is not None for input_si in (bias[i], a[i], b[i])):
                    node_output_si[input_node.name] = (
                        bias[i].detach().to(torch.float64),
                        a[i].detach().to(torch.float64),
                        b[i].detach().to(torch.float64),
                        l[i].detach().to(torch.float64),
                        u[i].detach().to(torch.float64),
                    )
                else:
                    node_output_si[input_node.name] = (None, None, None, None, None)

        for tensor in self.model.graph.initializer:
            arr = numpy_helper.to_array(tensor)
            if tensor.data_type == 7:
                arr = torch.tensor(arr, dtype=torch.int64)
            else:
                arr = torch.tensor(arr, dtype=torch.float64)
            node_output[tensor.name] = arr

        with torch.no_grad():
            for node in self.model.graph.node:
                inputs = [
                    node_output[input_name]
                    for input_name in node.input
                    if input_name != ""
                ]
                op_type = node.op_type

                if op_type in self.layers:
                    layer = self.layers[op_type](inputs, node)
                    x, bias, a, b, l, u = layer.forward_si(
                        node_output[node.input[0]], *node_output_si[node.input[0]]
                    )
                elif op_type in self.multi_input_layers:
                    layer = self.multi_input_layers[op_type](inputs, node, node_output)
                    x, bias, a, b, l, u = layer.forward_si(
                        node, node_output, node_output_si
                    )
                elif op_type in self.non_input_layers:
                    layer = self.non_input_layers[op_type](inputs, node)
                    x, bias, a, b, l, u = layer.forward_si()
                else:
                    raise NotImplementedError(f"Layer {op_type} is not supported.")

                if isinstance(x, torch.Tensor) or op_type == "Constant":
                    assert l < u
                    node_output[node.output[0]] = x
                    node_output_si[node.output[0]] = (bias, a, b, l, u)
                else:
                    for i, output_name in enumerate(node.output):
                        assert l[i] < u[i]
                        node_output[output_name] = x[i]
                        node_output_si[output_name] = (bias[i], a[i], b[i], l[i], u[i])

        self.output = node_output
        self.output_si = node_output_si
        x, output_bias, output_a, output_b, l, u = zip(
            *[
                [
                    node_output[output.name],
                    node_output_si[output.name][0],
                    node_output_si[output.name][1],
                    node_output_si[output.name][2],
                    node_output_si[output.name][3],
                    node_output_si[output.name][4],
                ]
                for output in self.model.graph.output
            ]
        )

        if len(x) == 1:
            return x[0], output_bias[0], output_a[0], output_b[0], l[0], u[0]
        else:
            return x, output_bias, output_a, output_b, l, u
