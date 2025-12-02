""" Module for dealing with model saving and loading. """
import os
import torch


def arg_to_params(args):
    return vars(args)


def print_arguments(args, ignore_params=set()):
    print("Parsed arguments:")
    for k, v in vars(args).items():
        if k in ignore_params.union(
            {"device", "optimal", "save_model", "save_file", "no_tqdm", "tqdm", "fast_train"}
        ):
            continue
        print("{0:20}  {1}".format(k, v))
    print("___")


def save_model_from_dict(model_dict, args):
    if not hasattr(args, "save_file") or args.save_file is None:
        return
    save_file = args.save_file
    save_dir = os.path.dirname(save_file)
    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving model at {save_file}...")
    torch.save((model_dict, args), save_file)
    print("Model saved!")
    print("Model parameter file:", save_file)
    return


def save_model(model, args):
    save_model_from_dict(model.model.state_dict(), args)
    return


def load_model(path, print_args=False):
    # returns (GNN, Args)
    import torch
    from models.model import Model

    if not os.path.exists(path):
        print(f"Model not found at {path}")
        exit(-1)
    if torch.cuda.is_available():
        model_state_dict, args = torch.load(path)
    else:
        model_state_dict, args = torch.load(path, map_location=torch.device("cpu"))
    model = Model(params=arg_to_params(args))
    model.load_state_dict_into_gnn(model_state_dict)
    if print_args:
        print_arguments(args)
    model.set_eval()
    return model, args

def load_and_setup_gnn_model(path, domain_file, problem_file):
    model, args = load_model(path)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.batch_search(True)
    model.update_representation(
        domain_pddl=domain_file, problem_pddl=problem_file, args=args, device=device
    )
    model.set_zero_grad()
    model.eval()
    return model
