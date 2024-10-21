import cellpose.models as cp_model
import cellpose.resnet_torch as cp_net
# import resnet_torch_copy as cp_net
import torch
import os
import argparse

import sys
# sys.path.append(r"C:\Users\s4702415\OneDrive\Documents\uni\2024\cellpose\cellpose")

import cv2

# image = r"C:\Users\s4702415\OneDrive\Documents\uni\2024\Honours\models\cellpose\000_img.png"
# # image = "/scratch/user/s4702415/Honours/models/cellpose/000_img.png"
# image = cv2.imread(image)

# image1 = image[:, :, 1]
# image2 = image[:, :, 2]

# image1 = cv2.resize(image1, (224, 224))
# image2 = cv2.resize(image2, (224, 224))

# image1 = torch.tensor(image1, dtype=torch.float)
# image2 = torch.tensor(image2, dtype=torch.float)

# # image2 = cv2.resize(image2, (d, d))
# image = torch.tensor(image, dtype=torch.float)
# input_x = torch.stack([image1, image2], dim=0).unsqueeze(0)

def convert_to_ONNX(model_path: str, output_directory: str, diam_mean: float, batch_size: int = 1):
    residual_on, style_on, concatenation = True, True, False
    model = cp_net.CPnet(
        [2, 32, 64, 128, 256],
        3,
        sz=3,
        # residual_on=residual_on,
        # style_on=style_on,
        # concatenation=concatenation,
        mkldnn=None,
        diam_mean=diam_mean)
    # model.concatenation = True
    model.load_model(model_path)
    
    # that means that the inputs are unused/untrained?
    model.eval()

    # convert to onnx
    onnx_model_path = os.path.join(output_directory, os.path.split(model_path)[-1] + ".onnx")
    dummy = torch.randn(batch_size, 2, 28, 28, requires_grad=True)
    torch.onnx.export(
        model.eval(),
        dummy,
        onnx_model_path,
        verbose=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        # autograd_inlining=True,
        keep_initializers_as_inputs=True,
        input_names=['input'],
        output_names=['output', 'style'])


def convert_all_models(output_directory: str):
    # get built-in model names and custom model names
    all_models = cp_model.MODEL_NAMES.copy()
    model_strings = cp_model.get_user_models()
    all_models.extend(model_strings)

    for model_type in all_models:
        model_string = model_type if model_type is not None else 'cyto'
        if model_string == 'nuclei':
            diam_mean = 17.
        else:
            diam_mean = 30.

        if model_type == 'cyto' or model_type == 'cyto2' or model_type == 'nuclei':
            model_range = range(4)
        else:
            model_range = range(1)

        model_pathes = [cp_model.model_path(model_string, j, True) for j in model_range]

        for model_path in model_pathes:
            convert_to_ONNX(model_path, output_directory, diam_mean)


def main():

    # parser = argparse.ArgumentParser(description='Converter parameters')
    # parser.add_argument('--output_directory', required=False, default=f"{os.path.join(cp_model.MODEL_DIR, 'output')}",
    #                     type=str, help='Output directory for converted models. ')
    # parser.add_argument('--model_path', required=False, default=None, type=str,
    #                     help='full path to the individual cellpose model')
    # parser.add_argument('--mean_diameter', required=False, type=float,
    #                       help='Mean diameter used for training the given model. 17.0 for nuclei-based models, otherwise 30.0')

    # args = parser.parse_args()
    # cellpose_path = r"C:\Users\s4702415\.cellpose\models\cyto3"

    # genesegnet_path = r"C:\Users\s4702415\OneDrive\Documents\uni\2024\models\GeneSegNet\models\GeneSegNet_residual_on_style_off_concatenation_off_GeneSegNet_2024_04_12_02_59_17.091359_epoch_499"
    # genesegnet_path = r"C:\Users\s4702415\OneDrive\Documents\uni\2024\Honours\models\cellpose\GeneSegNet_residual_on_style_off_concatenation_off_GeneSegNet3Channel_2024_07_23_08_50_42.393554_epoch_1"

    args = argparse.Namespace(output_directory=r"cellpose_n28.onnx",
                              model_path=cellpose_path, mean_diameter=30.0)

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    if args.model_path:
        if 'mean_diameter' not in vars(args) or args.mean_diameter is None:
            raise Exception("The __model_path argument requires the __mean_diameter.")
        if args.mean_diameter != 17.0 and args.mean_diameter != 30.0:
            raise Exception("Mean_diameter must be either 17.0 (for nuclei-based models) or 30.0 for all other models.")
        convert_to_ONNX(model_path=args.model_path, output_directory=args.output_directory, diam_mean=args.mean_diameter)
    else:
        # get user models and built-in models
        convert_all_models(args.output_directory)

    print("Output models are saved here: ", args.output_directory)
    print("Conversion completed.")


if __name__ == "__main__":
    main()
