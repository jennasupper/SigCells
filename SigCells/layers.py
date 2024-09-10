import numpy as np
import torch
import torch.nn as nn


def truncated_interval(tTa, tTb, bias):
    nominator = torch.neg(torch.add(tTa, bias))
    denominator = tTb
    plus_index = torch.greater(denominator, 0)
    minus_index = torch.less(denominator, 0)

    if torch.any(minus_index):
        l = torch.max(torch.div(nominator[minus_index], denominator[minus_index]))
    else:
        l = torch.tensor(-float("inf"), dtype=torch.float64)
    if torch.any(plus_index):
        u = torch.min(torch.div(nominator[plus_index], denominator[plus_index]))
    else:
        u = torch.tensor(float("inf"), dtype=torch.float64)
    return l, u


class Layer:
    def __init__(self, inputs, node):
        """
        Args:
            inputs (List): input tensor list
        """
        self.attribute = {}
        for attr in node.attribute:
            self.attribute[attr.name] = attr

    def forward(self):
        pass

    def forward_si(self):
        pass


class Relu(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        pass

    def forward(self, x):
        output = torch.nn.functional.relu(x)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        if all(input_si is not None for input_si in (bias, a, b)):
            relu_index = torch.greater_equal(x, 0)
            tTa = torch.where(relu_index, -a, a)
            tTb = torch.where(relu_index, -b, b)
            tTbias = torch.where(relu_index, -bias, bias)

            temp_l, temp_u = truncated_interval(tTa, tTb, tTbias)

            output_l = torch.max(l, temp_l)
            output_u = torch.min(u, temp_u)

            output_x = torch.where(relu_index, x, torch.tensor(0, dtype=torch.float64))
            output_bias = torch.where(
                relu_index, bias, torch.tensor(0, dtype=torch.float64)
            )
            output_a = torch.where(relu_index, a, torch.tensor(0, dtype=torch.float64))
            output_b = torch.where(relu_index, b, torch.tensor(0, dtype=torch.float64))
        else:
            output_x = torch.nn.functional.relu(x)
            output_bias, output_a, output_b = None, None, None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)

        return output_x, output_bias, output_a, output_b, output_l, output_u


class Sigmoid(Layer):
    """
    Sigmoid activation function can be used only in the intermediate layers
    that are not subject to the test or the final layer of the input subject to the test.
    """

    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        pass

    def forward(self, x):
        output = torch.nn.functional.sigmoid(x)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        if all(input_si is not None for input_si in (bias, a, b)):
            output_x = torch.nn.functional.sigmoid(x)
            output_bias = bias
            output_a = a
            output_b = b
        else:
            output_x = torch.nn.functional.sigmoid(x)
            output_bias, output_a, output_b = None, None, None
        return output_x, output_bias, output_a, output_b, l, u


class Conv(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.weight = inputs[1].to(torch.float64)
        self.bias = inputs[2].to(torch.float64) if len(inputs) > 2 else None
        self.strides = (
            tuple(self.attribute["strides"].ints) if "strides" in self.attribute else 1
        )
        self.pads = (
            int(self.attribute["pads"].ints[0]) if "pads" in self.attribute else 0
        )
        self.pads = (
            "same"
            if "pads" not in self.attribute and "auto_pad" in self.attribute
            else self.pads
        )
        # Not necessary since kernel_shape can be obtained from weight
        # self.kernel_shape = self.attribute["kernel_shape"].ints if "kernel_shape" in self.attribute else 1
        self.dilations = (
            tuple(self.attribute["dilations"].ints)
            if "dilations" in self.attribute
            else 1
        )
        self.group = self.attribute["group"].i if "group" in self.attribute else 1

    def forward(self, x):
        output = torch.nn.functional.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            dilation=self.dilations,
            groups=self.group,
        )
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = torch.nn.functional.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            dilation=self.dilations,
            groups=self.group,
        )
        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = torch.nn.functional.conv2d(
                input=bias,
                weight=self.weight,
                bias=self.bias,
                stride=self.strides,
                padding=self.pads,
                dilation=self.dilations,
                groups=self.group,
            )
            output_a = torch.nn.functional.conv2d(
                input=a,
                weight=self.weight,
                bias=None,
                stride=self.strides,
                padding=self.pads,
                dilation=self.dilations,
                groups=self.group,
            )
            output_b = torch.nn.functional.conv2d(
                input=b,
                weight=self.weight,
                bias=None,
                stride=self.strides,
                padding=self.pads,
                dilation=self.dilations,
                groups=self.group,
            )
        else:
            output_bias, output_a, output_b = None, None, None
        return output_x, output_bias, output_a, output_b, l, u


class ConvTranspose(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.weight = inputs[1].to(torch.float64)
        self.bias = inputs[2].to(torch.float64) if len(inputs) > 2 else None
        self.strides = (
            self.attribute["strides"].ints[0] if "strides" in self.attribute else 1
        )
        self.pads = self.attribute["pads"].ints[0] if "pads" in self.attribute else 0
        self.output_padding = (
            self.attribute["output_padding"].ints[0]
            if "output_padding" in self.attribute
            else 0
        )
        self.dilations = (
            self.attribute["dilations"].ints[0] if "dilations" in self.attribute else 1
        )

    def forward(self, x):
        output = torch.nn.functional.conv_transpose2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            output_padding=self.output_padding,
            dilation=self.dilations,
        )
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = torch.nn.functional.conv_transpose2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            output_padding=self.output_padding,
            dilation=self.dilations,
        )
        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = torch.nn.functional.conv_transpose2d(
                input=bias,
                weight=self.weight,
                bias=self.bias,
                stride=self.strides,
                padding=self.pads,
                output_padding=self.output_padding,
                dilation=self.dilations,
            )
            output_a = torch.nn.functional.conv_transpose2d(
                input=a,
                weight=self.weight,
                bias=None,
                stride=self.strides,
                padding=self.pads,
                output_padding=self.output_padding,
                dilation=self.dilations,
            )
            output_b = torch.nn.functional.conv_transpose2d(
                input=b,
                weight=self.weight,
                bias=None,
                stride=self.strides,
                padding=self.pads,
                output_padding=self.output_padding,
                dilation=self.dilations,
            )
        else:
            output_bias, output_a, output_b = None, None, None
        return output_x, output_bias, output_a, output_b, l, u


class MaxPool(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.kernel_shape = tuple(self.attribute["kernel_shape"].ints)
        self.strides = tuple(self.attribute["strides"].ints)
        self.pads = (
            int(self.attribute["pads"].ints[0]) if "pads" in self.attribute else 0
        )

    def forward(self, x):
        if x.dim() == 3:
            output = nn.functional.max_pool1d(
                x, kernel_size=self.kernel_shape, stride=self.stride, padding=self.pads
            )
            return output
        elif x.dim() == 4:
            output = nn.functional.max_pool2d(
                x, kernel_size=self.kernel_shape, stride=self.strides, padding=self.pads
            )
            return output
        else:
            raise ValueError(
                "Input dimension must be 3 (for 1D) or 4 (for 2D) but got {}".format(
                    x.dim()
                )
            )

    def forward_si(self, x, bias, a, b, l, u):
        B, C, H, W = x.shape
        H_out = (H + 2 * self.pads - self.kernel_shape[0]) // self.strides[0] + 1
        W_out = (W + 2 * self.pads - self.kernel_shape[1]) // self.strides[1] + 1
        x_im2coled = nn.functional.unfold(
            x, kernel_size=self.kernel_shape, stride=self.strides, padding=self.pads
        )
        x_im2coled_reshaped = x_im2coled.view(
            B, C, self.kernel_shape[0] * self.kernel_shape[1], H_out * W_out
        )
        max_index = x_im2coled_reshaped.argmax(dim=2)
        output_x = x_im2coled_reshaped.gather(
            dim=2, index=max_index.unsqueeze(2)
        ).squeeze(2)
        output_x = output_x.view(B, C, H_out, W_out)

        if all(input_si is not None for input_si in (bias, a, b)):
            bias_im2coled = nn.functional.unfold(
                bias,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
            )
            a_im2coled = nn.functional.unfold(
                a, kernel_size=self.kernel_shape, stride=self.strides, padding=self.pads
            )
            b_im2coled = nn.functional.unfold(
                b, kernel_size=self.kernel_shape, stride=self.strides, padding=self.pads
            )

            bias_im2coled_reshaped = bias_im2coled.view(
                B, C, self.kernel_shape[0] * self.kernel_shape[1], H_out * W_out
            )
            a_im2coled_reshaped = a_im2coled.view(
                B, C, self.kernel_shape[0] * self.kernel_shape[1], H_out * W_out
            )
            b_im2coled_reshaped = b_im2coled.view(
                B, C, self.kernel_shape[0] * self.kernel_shape[1], H_out * W_out
            )

            output_bias = bias_im2coled_reshaped.gather(
                dim=2, index=max_index.unsqueeze(2)
            ).squeeze(2)
            output_a = a_im2coled_reshaped.gather(
                dim=2, index=max_index.unsqueeze(2)
            ).squeeze(2)
            output_b = b_im2coled_reshaped.gather(
                dim=2, index=max_index.unsqueeze(2)
            ).squeeze(2)

            tTa = a_im2coled_reshaped - output_a.unsqueeze(2)
            tTb = b_im2coled_reshaped - output_b.unsqueeze(2)
            bias = bias_im2coled_reshaped - output_bias.unsqueeze(2)

            temp_l, temp_u = truncated_interval(tTa, tTb, bias)

            l = torch.maximum(l, temp_l)
            u = torch.minimum(u, temp_u)

            output_bias = output_bias.view(B, C, H_out, W_out)
            output_a = output_a.view(B, C, H_out, W_out)
            output_b = output_b.view(B, C, H_out, W_out)
        else:
            output_bias, output_a, output_b = None, None, None

        return output_x, output_bias, output_a, output_b, l, u


class AveragePool(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.kernel_shape = tuple(self.attribute["kernel_shape"].ints)
        self.strides = tuple(self.attribute["strides"].ints)
        self.pads = (
            int(self.attribute["pads"].ints[0]) if "pads" in self.attribute else 0
        )
        self.ceil_mode = (
            bool(self.attribute["ceil_mode"].i) if "ceil_mode" in self.attribute else 0
        )
        self.count_include_pad = (
            bool(self.attribute["count_include_pad"].i)
            if "count_include_pad" in self.attribute
            else 1
        )

    def forward(self, x):
        if x.dim() == 3:
            output = nn.functional.avg_pool1d(
                x,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
        elif x.dim() == 4:
            output = nn.functional.avg_pool2d(
                x,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
        return output

    def forward_si(self, x, bias, a, b, l, u):
        if x.dim() == 3:  # not tested
            output_x = nn.functional.avg_pool1d(
                x, kernel_size=self.kernel_shape, stride=self.strides, padding=self.pads
            )
            if all(input_si is not None for input_si in (bias, a, b)):
                output_bias = nn.functional.avg_pool1d(
                    bias,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
                output_a = nn.functional.avg_pool1d(
                    a,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
                output_b = nn.functional.avg_pool1d(
                    b,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
            else:
                output_bias, output_a, output_b = None, None, None
        elif x.dim() == 4:
            output_x = nn.functional.avg_pool2d(
                x,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            if all(input_si is not None for input_si in (bias, a, b)):
                output_bias = nn.functional.avg_pool2d(
                    bias,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
                output_a = nn.functional.avg_pool2d(
                    a,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
                output_b = nn.functional.avg_pool2d(
                    b,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
            else:
                output_bias, output_a, output_b = None, None, None
        return output_x, output_bias, output_a, output_b, l, u


class Gemm(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.weight = inputs[1].detach().to(torch.float64)
        self.bias = inputs[2].detach().to(torch.float64) if len(inputs) > 2 else None
        self.alpha = self.attribute["alpha"].f if "alpha" in self.attribute else 1.0
        self.beta = self.attribute["beta"].f if "beta" in self.attribute else 1.0
        self.transA = self.attribute["transA"].i if "transA" in self.attribute else 0
        self.transB = self.attribute["transB"].i if "transB" in self.attribute else 0

    def forward(self, x):
        if self.transA:
            x = x.t()
        if self.transB:
            self.weight = self.weight.t()
        output = self.alpha * torch.mm(x, self.weight) + self.beta * self.bias
        return output

    def forward_si(self, x, bias, a, b, l, u):
        if self.transA:
            x = x.t()
            if all(input_si is not None for input_si in (bias, a, b)):
                bias = bias.t()
                a = a.t()
                b = b.t()
            else:
                bias, a, b = None, None, None
        if self.transB:
            self.weight = self.weight.t()

        output_x = self.alpha * torch.mm(x, self.weight) + self.beta * self.bias

        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = (
                self.alpha * torch.mm(bias, self.weight) + self.beta * self.bias
            )
            output_a = self.alpha * torch.mm(a, self.weight)
            output_b = self.alpha * torch.mm(b, self.weight)
            output_l = l
            output_u = u
        else:
            output_bias, output_a, output_b = None, None, None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_x, output_bias, output_a, output_b, output_l, output_u


class Transpose(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.perm = self.attribute["perm"].ints

    def forward(self, x):
        output = x.permute(tuple(self.perm))
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = x.permute(tuple(self.perm))
        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = bias.permute(tuple(self.perm))
            output_a = a.permute(tuple(self.perm))
            output_b = b.permute(tuple(self.perm))
            output_l = l
            output_u = u
        else:
            output_bias, output_a, output_b = None, None, None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_x, output_bias, output_a, output_b, output_l, output_u


class Shape(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.end = self.attribute["end"].i if "end" in self.attribute else None
        self.start = self.attribute["start"].i if "start" in self.attribute else 0

    def forward(self, x):
        shape = x.shape
        rank = len(shape)

        if self.start < 0:
            start = max(0, rank + self.start)
        else:
            start = min(rank, self.start)

        if self.end is None or self.end >= rank:
            end = rank
        elif self.end < 0:
            end = max(0, rank + self.end)
        else:
            end = min(rank, self.end)

        output = torch.tensor(shape[start:end], dtype=torch.int64)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = self.forward(x)
        output_bias = torch.zeros_like(output_x)
        output_a = torch.zeros_like(output_x)
        output_b = torch.zeros_like(output_x)

        return output_x, output_bias, output_a, output_b, l, u


# Now fixing
class Slice(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)

        self.starts = list(inputs[1].item()) if inputs[1] != [] else None
        self.ends = list(inputs[2].item()) if inputs[2] != [] else None
        self.axes = (
            list(inputs[3].item()) if len(inputs) > 3 and inputs[3] != [] else None
        )
        self.steps = (
            list(inputs[4].item()) if len(inputs) > 4 and inputs[4] != [] else None
        )

    def forward(self, x):
        slices = [slice(None)] * x.dim()

        if self.axes is None:
            axes = list(range(x.dim()))
        else:
            axes = self.axes

        for i, axis in enumerate(axes):
            start = (
                self.starts[i]
                if self.starts is not None and i < len(self.starts)
                else None
            )
            end = self.ends[i] if self.ends is not None and i < len(self.ends) else None
            step = (
                self.steps[i]
                if self.steps is not None and i < len(self.steps)
                else None
            )

            slices[axis] = slice(start, end, step)

        output = x[tuple(slices)]
        return output

    def forward_si(self, x, bias, a, b, l, u):
        slices = [slice(None)] * x.dim()

        if self.axes is None:
            axes = list(range(x.dim()))
        else:
            axes = self.axes

        for i, axis in enumerate(axes):
            start = (
                self.starts[i]
                if self.starts is not None and i < len(self.starts)
                else None
            )
            end = self.ends[i] if self.ends is not None and i < len(self.ends) else None
            step = (
                self.steps[i]
                if self.steps is not None and i < len(self.steps)
                else None
            )

            slices[axis] = slice(start, end, step)

        output_x = x[tuple(slices)]
        output_bias = bias[tuple(slices)]
        output_a = a[tuple(slices)]
        output_b = b[tuple(slices)]

        return output_x, output_bias, output_a, output_b, l, u


class Exp(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)

    def forward(self, x):
        output = torch.exp(x)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = torch.exp(x)
        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = torch.exp(bias)
            output_a = torch.exp(a)
            output_b = torch.exp(b)
        else:
            output_bias, output_a, output_b = None, None, None
        return output_x, output_bias, output_a, output_b, l, u


class RandomNormalLike(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.mean = self.attribute["mean"].f if "mean" in self.attribute else 0.0
        self.scale = self.attribute["scale"].f if "scale" in self.attribute else 1.0
        self.seed = self.attribute["seed"].i if "seed" in self.attribute else 0
        self.rng = np.random.default_rng(self.seed)
        self.normal_like = torch.tensor(
            self.rng.normal(self.mean, self.scale, size=inputs[0].shape),
            dtype=torch.float64,
        )

    def forward(self, x, mean=None, scale=None):
        output = self.normal_like
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = self.normal_like
        output_bias = None
        output_a = None
        output_b = None
        return output_x, output_bias, output_a, output_b, l, u


class Flatten(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.axis = self.attribute["axis"].i if "axis" in self.attribute else 1

    def forward(self, x):
        output = torch.flatten(x, start_dim=self.axis)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = torch.flatten(x, start_dim=self.axis)
        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = torch.flatten(bias, start_dim=self.axis)
            output_a = torch.flatten(a, start_dim=self.axis)
            output_b = torch.flatten(b, start_dim=self.axis)
        else:
            output_bias, output_a, output_b = None, None, None
        return output_x, output_bias, output_a, output_b, l, u


class Reshape(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        self.shape = node_output[node.input[1]]

    def forward(self):
        output = torch.reshape(self.input, self.shape)
        return output

    def forward_si(self, node, node_output, node_output_si):
        bias = node_output_si[node.input[0]][0]
        a = node_output_si[node.input[0]][1]
        b = node_output_si[node.input[0]][2]
        l = node_output_si[node.input[0]][3]
        u = node_output_si[node.input[0]][4]

        output_x = torch.reshape(self.input, self.shape)
        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = torch.reshape(bias, self.shape)
            output_a = torch.reshape(a, self.shape)
            output_b = torch.reshape(b, self.shape)
            output_l = l
            output_u = u
        else:
            output_bias, output_a, output_b = None, None, None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_x, output_bias, output_a, output_b, output_l, output_u


class Resize(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        self.roi = (
            node_output[node.input[1]]
            if len(node.input) > 1 and node.input[1] != ""
            else None
        )
        self.scales = (
            node_output[node.input[2]]
            if len(node.input) > 2 and node.input[2] != ""
            else None
        )
        self.sizes = (
            node_output[node.input[3]]
            if len(node.input) > 3 and node.input[3] != ""
            else None
        )

        self.mode = (
            self.attribute["mode"].s.decode() if "mode" in self.attribute else "nearest"
        )
        self.coordinate_transformation_mode = (
            self.attribute["coordinate_transformation_mode"].s.decode()
            if "coordinate_transformation_mode" in self.attribute
            else "half_pixel"
        )
        self.antialias = (
            self.attribute["antialias"].i if "antialias" in self.attribute else 0
        )
        # self.nearest_mode = self.attribute["nearest_mode"].s.decode() if "nearest_mode" in self.attribute else "round_prefer_floor"
        # self.cubic_coeff_a = self.attribute["cubic_coeff_a"].f if "cubic_coeff_a" in self.attribute else -0.75
        # self.exclude_outside = self.attribute["exclude_outside"].i if "exclude_outside" in self.attribute else 0
        # self.extrapolation_value = self.attribute["extrapolation_value"].f if "extrapolation_value" in self.attribute else 0.0
        # self.keep_aspect_ratio_policy = self.attribute["keep_aspect_ratio_policy"].s.decode() if "keep_aspect_ratio_policy" in self.attribute else "stretch"
        # self.axes = self.attribute["axes"].ints if "axes" in self.attribute else None

        if self.input.dim() == 4:
            if self.scales is not None:
                self.scales = (float(self.scales[2]), float(self.scales[3]))
            if self.mode == "linear":
                self.mode = "bilinear"
            elif self.mode == "cubic":
                self.mode = "bicubic"
        elif self.input.dim() == 3:
            if self.scales is not None:
                self.scales = (float(self.scales[1]),)  # not tested

        self.align_corners = self.coordinate_transformation_mode == "align_corners"
        if self.mode not in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            self.align_corners = None

    def forward(self):

        output = torch.nn.functional.interpolate(
            input=self.input,
            size=self.sizes,
            scale_factor=self.scales,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.scales is None,
            antialias=self.antialias,
        )
        return output

    def forward_si(self, node, node_output, node_output_si):
        bias = node_output_si[node.input[0]][0]
        a = node_output_si[node.input[0]][1]
        b = node_output_si[node.input[0]][2]
        l = node_output_si[node.input[0]][3]
        u = node_output_si[node.input[0]][4]

        output_x = torch.nn.functional.interpolate(
            input=self.input,
            size=self.sizes,
            scale_factor=self.scales,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.scales is None,
            antialias=self.antialias,
        )

        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = torch.nn.functional.interpolate(
                bias,
                size=self.sizes,
                scale_factor=self.scales,
                mode=self.mode,
                align_corners=self.align_corners,
                recompute_scale_factor=self.scales is None,
            )
            output_a = torch.nn.functional.interpolate(
                a,
                size=self.sizes,
                scale_factor=self.scales,
                mode=self.mode,
                align_corners=self.align_corners,
                recompute_scale_factor=self.scales is None,
            )
            output_b = torch.nn.functional.interpolate(
                b,
                size=self.sizes,
                scale_factor=self.scales,
                mode=self.mode,
                align_corners=self.align_corners,
                recompute_scale_factor=self.scales is None,
            )
            output_l = l
            output_u = u
        else:
            output_bias, output_a, output_b = None, None, None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)

        return output_x, output_bias, output_a, output_b, output_l, output_u


class Concat(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.axis = self.attribute["axis"].i
        self.inputs = [node_output[input_name] for input_name in node.input]

    def forward(self):
        output = torch.cat(self.inputs, dim=self.axis)
        return output

    def forward_si(self, node, node_output, node_output_si):
        bias = [
            (
                node_output_si[input_name][0]
                if node_output_si[input_name][0] is not None
                else node_output[input_name]
            )
            for input_name in node.input
        ]
        a = [
            (
                node_output_si[input_name][1]
                if node_output_si[input_name][1] is not None
                else torch.zeros_like(node_output[input_name], dtype=torch.float64)
            )
            for input_name in node.input
        ]
        b = [
            (
                node_output_si[input_name][2]
                if node_output_si[input_name][2] is not None
                else torch.zeros_like(node_output[input_name], dtype=torch.float64)
            )
            for input_name in node.input
        ]
        l = torch.tensor([node_output_si[input_name][3] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][4] for input_name in node.input])

        output_x = torch.cat(self.inputs, dim=self.axis)
        output_bias = torch.cat(bias, dim=self.axis)
        output_a = torch.cat(a, dim=self.axis)
        output_b = torch.cat(b, dim=self.axis)
        output_l, output_u = l.max(), u.min()
        return output_x, output_bias, output_a, output_b, output_l, output_u


class Add(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.inputs = [node_output[input_name] for input_name in node.input]

    def forward(self):
        output = torch.add(self.inputs[0], self.inputs[1])
        return output

    def forward_si(self, node, node_output, node_output_si):
        bias = [
            (
                node_output_si[input_name][0]
                if node_output_si[input_name][0] is not None
                else node_output[input_name]
            )
            for input_name in node.input
        ]
        a = [
            (
                node_output_si[input_name][1]
                if node_output_si[input_name][1] is not None
                else torch.zeros_like(node_output[input_name], dtype=torch.float64)
            )
            for input_name in node.input
        ]
        b = [
            (
                node_output_si[input_name][2]
                if node_output_si[input_name][2] is not None
                else torch.zeros_like(node_output[input_name], dtype=torch.float64)
            )
            for input_name in node.input
        ]
        l = torch.tensor([node_output_si[input_name][3] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][4] for input_name in node.input])

        output_x = torch.add(self.inputs[0], self.inputs[1])
        output_bias = torch.add(bias[0], bias[1])
        output_a = torch.add(a[0], a[1])
        output_b = torch.add(b[0], b[1])
        output_l, output_u = l.max(), u.min()
        return output_x, output_bias, output_a, output_b, output_l, output_u


class Split(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.inputs = [node_output[input_name] for input_name in node.input]
        self.axis = self.attribute["axis"].i
        self.num_outputs = (
            self.attribute["num_outputs"].i if "num_outputs" in self.attribute else None
        )  # Not refactored

    def forward(self) -> tuple[torch.Tensor]:
        if len(self.inputs) > 1:
            output = torch.split(
                tensor=self.inputs[0],
                split_size_or_sections=self.inputs[1],
                dim=self.axis,
            )
        return output

    def forward_si(self, node, node_output, node_output_si):
        bias = node_output_si[node.input[0]][0]
        a = node_output_si[node.input[0]][1]
        b = node_output_si[node.input[0]][2]
        l = torch.tensor([node_output_si[input_name][3] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][4] for input_name in node.input])

        output_x = torch.split(
            tensor=self.inputs[0],
            split_size_or_sections=self.inputs[1],
            dim=self.axis
        )
        output_bias = torch.split(
            tensor=bias,
            split_size_or_sections=self.inputs[1],
            dim=self.axis
        )
        output_a = torch.split(
            tensor=a,
            split_size_or_sections=self.inputs[1],
            dim=self.axis
        )
        output_b = torch.split(
            tensor=b,
            split_size_or_sections=self.inputs[1],
            dim=self.axis
        )
        l = torch.tensor([node_output_si[node.input[0]][3] for _ in node.output])
        u = torch.tensor([node_output_si[node.input[0]][4] for _ in node.output])

        return output_x, output_bias, output_a, output_b, l, u


class BatchNormalization(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        # if len(self.input.shape) != 4:
        #     self.input = self.input.unsqueeze(0)
        self.scale = node_output[node.input[1]]
        self.B = node_output[node.input[2]]
        self.input_mean = node_output[node.input[3]]
        self.input_var = node_output[node.input[4]]
        self.epsilon = (
            self.attribute["epsilon"].f if "epsilon" in self.attribute else 1e-5
        )
        self.momentum = (
            self.attribute["momentum"].f if "momentum" in self.attribute else 0.9
        )
        self.training_mode = (
            self.attribute["training_mode"].i
            if "training_mode" in self.attribute
            else 1
        )

    def forward(self):
        if self.training_mode:
            raise NotImplementedError(
                "Training mode is not supported. Please save the model in evaluation mode."
            )
        else:
            output = torch.nn.functional.batch_norm(
                input=self.input, running_mean=self.input_mean, running_var=self.input_var, weight=self.scale, bias=self.B, eps=self.epsilon)
            # output = (self.input - self.input_mean) / torch.sqrt(
            #     self.input_var + self.epsilon
            # ) * self.scale + self.B
            
        return output

    def forward_si(self, node, node_output, node_output_si):

        if self.training_mode:
            raise NotImplementedError(
                "Training mode is not supported. Please save the model in evaluation mode."
            )
        else:
            bias = node_output_si[node.input[0]][0]
            a = node_output_si[node.input[0]][1]
            b = node_output_si[node.input[0]][2]
            l = node_output_si[node.input[0]][3]
            u = node_output_si[node.input[0]][4]
            # output_x = (self.input - self.input_mean) / torch.sqrt(
            #     self.input_var + self.epsilon
            # ) * self.scale + self.B

            output_x = torch.nn.functional.batch_norm(
                input=self.input, running_mean=self.input_mean, running_var=self.input_var, weight=self.scale, bias=self.B, eps=self.epsilon)

            if all(input_si is not None for input_si in (bias, a, b)):
                output_bias = []
                output_a = []
                output_b = []
                for i in range(len(self.input_mean)):
                    out = (bias[:, i, :, :] - self.input_mean[i]) / torch.sqrt(
                        self.input_var[i] + self.epsilon
                    ) * self.scale[i] + self.B[i]
                    output_bias.append(out)

                    out_a = a[:, i, :, :] / torch.sqrt(self.input_var[i] + self.epsilon) * self.scale[i]
                    output_a.append(out_a)
                    out_b = b[:, i, :, :] / torch.sqrt(self.input_var[i] + self.epsilon) * self.scale[i]
                    output_b.append(out_b)

                output_bias = torch.stack(output_bias, dim=1)
                output_a = torch.stack(output_a, dim=1)
                output_b = torch.stack(output_b, dim=1)
                output_l = l
                output_u = u
            else:
                output_bias, output_a, output_b = None, None, None
                output_l = torch.tensor(-float("inf"), dtype=torch.float64)
                output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_x, output_bias, output_a, output_b, output_l, output_u
    

class Mul(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.A = node_output[node.input[0]]
        self.B = node_output[node.input[1]]

    def forward(self):
        output = torch.mul(self.A, self.B)
        return output

    def forward_si(self, node, node_output, node_output_si):
        A_bias = node_output_si[node.input[0]][0]
        A_a = node_output_si[node.input[0]][1]
        A_b = node_output_si[node.input[0]][2]
        A_l = node_output_si[node.input[0]][3]
        A_u = node_output_si[node.input[0]][4]
        B_bias = node_output_si[node.input[1]][0]
        B_a = node_output_si[node.input[1]][1]
        B_b = node_output_si[node.input[1]][2]
        B_l = node_output_si[node.input[1]][3]
        B_u = node_output_si[node.input[1]][4]
        output_x = torch.mul(self.A, self.B)
        if all(input_si is not None for input_si in (A_bias, A_a, A_b)) or all(
            input_si is not None for input_si in (B_bias, B_a, B_b)
        ):
            if all(input_si is not None for input_si in (A_bias, A_a, A_b)):
                B_bias = self.B
                B_a = self.B
                B_b = self.B
            else:
                A_bias = self.A
                A_a = self.A
                A_b = self.A
            output_bias = torch.mul(A_bias, B_bias)
            output_a = torch.mul(A_a, B_a)
            output_b = torch.mul(A_b, B_b)
            output_l = torch.max(A_l, B_l)
            output_u = torch.min(A_u, B_u)
        else:
            output_bias, output_a, output_b = None, None, None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_x, output_bias, output_a, output_b, output_l, output_u


class Constant(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.dims = self.attribute["value"].t.dims
        self.data_type = self.attribute["value"].t.data_type
        self.raw_data = self.attribute["value"].t.raw_data

    def forward(self):
        if self.data_type == 1:
            x = np.frombuffer(self.raw_data, dtype=np.float32).astype(np.float64)
            x = x.reshape(self.dims)
            output = torch.tensor(x)
        elif self.data_type == 7:
            x = np.frombuffer(self.raw_data, dtype=np.int64)
            x = x.reshape(self.dims)
            output = tuple(x)
        elif self.data_type == 9:
            x = np.frombuffer(self.raw_data, dtype=np.uint8)
            x = x.reshape(self.dims)
            output = ()
        else:
            raise NotImplementedError(
                "data_type {} is not supported".format(self.data_type)
            )
        return output

    def forward_si(self):
        if self.data_type == 1:
            output_x = np.frombuffer(self.raw_data, dtype=np.float32).astype(np.float64)
            output_x = output_x.reshape(self.dims)
            output_x = torch.tensor(output_x, dtype=torch.float64)
            output_bias = None
            output_b = None
            output_a = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        elif self.data_type == 7:
            output_x = np.frombuffer(self.raw_data, dtype=np.int64)
            output_x = output_x.reshape(self.dims)
            if self.dims == [1]:
                output_x = int(output_x[0])
            else:
                output_x = tuple(output_x)
            output_bias = None
            output_b = None
            output_a = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        else:
            raise NotImplementedError(
                "data_type {} is not supported".format(self.data_type)
            )
        return output_x, output_bias, output_b, output_a, output_l, output_u
    

class Unsqueeze(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        # self.axis = self.attribute["axis"].i if "axis" in self.attribute else 1

    def forward(self, x):
        output = torch.unsqueeze(x, dim=-1)
        return output

    def forward_si(self, x, bias, a, b, l, u):
        output_x = torch.unsqueeze(x, dim=-1)
        if all(input_si is not None for input_si in (bias, a, b)):
            output_bias = torch.unsqueeze(bias, dim=-1)
            output_a = torch.unsqueeze(a, dim=-1)
            output_b = torch.unsqueeze(b, dim=-1)
        else:
            output_bias, output_a, output_b = None, None, None
        return output_x, output_bias, output_a, output_b, l, u
    

class Pow(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.inputs = [node_output[input_name] for input_name in node.input]

    def forward(self):
        output = torch.pow(self.inputs[0], self.inputs[1])
        return output

    def forward_si(self, node, node_output, node_output_si):
        bias = [
            (
                node_output_si[input_name][0]
                if node_output_si[input_name][0] is not None
                else node_output[input_name]
            )
            for input_name in node.input
        ]
        a = [
            (
                node_output_si[input_name][1]
                if node_output_si[input_name][1] is not None
                else torch.zeros_like(node_output[input_name], dtype=torch.float64)
            )
            for input_name in node.input
        ]
        b = [
            (
                node_output_si[input_name][2]
                if node_output_si[input_name][2] is not None
                else torch.zeros_like(node_output[input_name], dtype=torch.float64)
            )
            for input_name in node.input
        ]
        l = torch.tensor([node_output_si[input_name][3] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][4] for input_name in node.input])

        output_x = torch.pow(self.inputs[0], self.inputs[1])
        output_bias = torch.pow(bias[0], bias[1])
        output_a = torch.pow(a[0], a[1])
        output_b = torch.pow(b[0], b[1])
        output_l, output_u = l.max(), u.min()
        return output_x, output_bias, output_a, output_b, output_l, output_u
    

class Div(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.A = node_output[node.input[0]]
        self.B = node_output[node.input[1]]

    def forward(self):
        output = torch.div(self.A, self.B)
        return output

    def forward_si(self, node, node_output, node_output_si):
        A_bias = node_output_si[node.input[0]][0]
        A_a = node_output_si[node.input[0]][1]
        A_b = node_output_si[node.input[0]][2]
        A_l = node_output_si[node.input[0]][3]
        A_u = node_output_si[node.input[0]][4]
        B_bias = node_output_si[node.input[1]][0]
        B_a = node_output_si[node.input[1]][1]
        B_b = node_output_si[node.input[1]][2]
        B_l = node_output_si[node.input[1]][3]
        B_u = node_output_si[node.input[1]][4]
        output_x = torch.div(self.A, self.B)
        if all(input_si is not None for input_si in (A_bias, A_a, A_b)) or all(
            input_si is not None for input_si in (B_bias, B_a, B_b)
        ):
            if all(input_si is not None for input_si in (A_bias, A_a, A_b)):
                B_bias = self.B
                B_a = self.B
                B_b = self.B
            else:
                A_bias = self.A
                A_a = self.A
                A_b = self.A
            output_bias = torch.div(A_bias, B_bias)
            output_a = torch.div(A_a, B_a)
            output_b = torch.div(A_b, B_b)
            output_l = torch.max(A_l, B_l)
            output_u = torch.min(A_u, B_u)
        else:
            output_bias, output_a, output_b = None, None, None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_x, output_bias, output_a, output_b, output_l, output_u
    

class ReduceSum(Layer):
    
    def forward(self, x):
        return torch.sum(x)
    
    def forward_si(self, x, bias, a, b, l, u):
        output_x = torch.sum(x)
        output_bias = torch.sum(bias)
        output_a = torch.sum(a)
        output_b = torch.sum(b)
        output_l = l
        output_u = u
        return output_x, output_bias, output_a, output_b, output_l, output_u