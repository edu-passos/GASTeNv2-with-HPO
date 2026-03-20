import os
import json
import torch
import torch.optim as optim
import torchvision.utils as vutils
from collections import OrderedDict

from src.classifier import construct_classifier


def _sanitize_state_dict(state_dict, model):
    """
    Convert between DataParallel and non-DataParallel state_dict keys.
    """
    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())
    model_has_module = any(k.startswith("module.") for k in model_keys)
    state_has_module = any(k.startswith("module.") for k in state_keys)

    if state_has_module and not model_has_module:
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[len("module."):] if k.startswith("module.") else k
            new_state[new_key] = v
        return new_state

    if model_has_module and not state_has_module:
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state["module." + k] = v
        return new_state

    return state_dict


def checkpoint(model, model_name, model_params, train_stats, args,
               output_dir=None, optimizer=None):
    output_dir = os.path.curdir if output_dir is None else str(output_dir)
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    save_dict = {
        'name': model_name,
        'state': model.state_dict(),
        'stats': train_stats,
        'params': model_params,
        'args': args
    }

    json.dump({
        'train_stats': train_stats,
        'model_params': model_params,
        'args': vars(args)
    }, open(os.path.join(output_dir, 'stats.json'), 'w'), indent=2)

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()

    torch.save(save_dict, os.path.join(output_dir, 'classifier.pth'))
    return output_dir


def load_checkpoint(path, model, device=None, optimizer=None):
    cp = torch.load(path, map_location=device, weights_only=False)

    state = _sanitize_state_dict(cp['state'], model)
    model.load_state_dict(state)
    model.eval()

    if optimizer is not None and 'optimizer' in cp:
        optimizer.load_state_dict(cp['optimizer'])


def construct_classifier_from_checkpoint(path, device=None, optimizer=False):
    cp = torch.load(
        os.path.join(path, 'classifier.pth'),
        map_location=device,
        weights_only=False
    )

    print(f" > Loading model from {path} ...")
    print(f"\t. Model {cp['name']}")
    print(f"\t. Params: {cp['params']}")

    model = construct_classifier(cp['params'], device=device)
    state = _sanitize_state_dict(cp['state'], model)
    model.load_state_dict(state)
    model.eval()

    if optimizer:
        opt = optim.Adam(model.parameters())
        if 'optimizer' in cp:
            opt.load_state_dict(cp['optimizer'])
        return model, cp['params'], cp['stats'], cp['args'], opt
    else:
        return model, cp['params'], cp['stats'], cp['args']


def construct_gan_from_checkpoint(path, device=None):
    from src.gan import construct_gan

    print(f"Loading GAN from {path} ...")
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config = json.load(f)

    model_params = config['model']
    optim_params = config['optimizer']

    gen_cp = torch.load(
        os.path.join(path, 'generator.pth'),
        map_location=device,
        weights_only=False
    )
    dis_cp = torch.load(
        os.path.join(path, 'discriminator.pth'),
        map_location=device,
        weights_only=False
    )

    if 'image-size' in model_params:
        image_size = tuple(model_params['image-size'])
    elif 'image_size' in model_params:
        image_size = tuple(model_params['image_size'])
    else:
        raise KeyError("Could not find 'image-size' or 'image_size' in model configuration.")

    G, D = construct_gan(model_params, image_size, device=device)

    G.load_state_dict(_sanitize_state_dict(gen_cp['state'], G))
    D.load_state_dict(_sanitize_state_dict(dis_cp['state'], D))

    g_optim = optim.Adam(G.parameters(),
                         lr=optim_params["lr"],
                         betas=(optim_params["beta1"], optim_params["beta2"]))
    d_optim = optim.Adam(D.parameters(),
                         lr=optim_params["lr"],
                         betas=(optim_params["beta1"], optim_params["beta2"]))

    if 'optimizer' in gen_cp:
        g_optim.load_state_dict(gen_cp['optimizer'])
    if 'optimizer' in dis_cp:
        d_optim.load_state_dict(dis_cp['optimizer'])

    G.eval()
    D.eval()
    return G, D, g_optim, d_optim


def get_gan_path_at_epoch(output_dir, epoch=None):
    base = str(output_dir)
    if epoch is None:
        return base

    try:
        ei = int(epoch)
        es = f"{ei:02d}"
    except (TypeError, ValueError):
        es = str(epoch)

    return os.path.join(base, es)

def load_gan_train_state(output_dir: str):
    """
    Load the GAN training state saved by checkpoint_gan().

    Expected file: <output_dir>/train_state.json
    Returns: dict (may be empty if missing)
    """
    path = os.path.join(str(output_dir), "train_state.json")
    if not os.path.exists(path):
        # be defensive: some older runs might not have it
        return {}
    with open(path, "r") as f:
        return json.load(f)


def checkpoint_gan(G, D, g_opt, d_opt, state, stats,
                   config, output_dir=None, epoch=None):
    if isinstance(output_dir, int) and epoch is None:
        epoch, output_dir = output_dir, None

    rootdir = os.path.curdir if output_dir is None else str(output_dir)
    os.makedirs(rootdir, exist_ok=True)  # <-- IMPORTANT: ensure root exists

    path = get_gan_path_at_epoch(rootdir, epoch)
    os.makedirs(path, exist_ok=True)

    torch.save({
        'state': G.state_dict(),
        'optimizer': g_opt.state_dict()
    }, os.path.join(path, 'generator.pth'))

    torch.save({
        'state': D.state_dict(),
        'optimizer': d_opt.state_dict()
    }, os.path.join(path, 'discriminator.pth'))

    json.dump(state, open(os.path.join(rootdir, 'train_state.json'), 'w'), indent=2)
    json.dump(stats, open(os.path.join(rootdir, 'stats.json'), 'w'), indent=2)
    json.dump(config, open(os.path.join(path, 'config.json'), 'w'), indent=2)

    print(f'> Saved checkpoint to {path}')
    return path



def checkpoint_image(image, epoch, output_dir=None):
    dirpath = os.path.curdir if output_dir is None else str(output_dir)
    dirpath = os.path.join(dirpath, 'gen_images')
    os.makedirs(dirpath, exist_ok=True)
    vutils.save_image(image, os.path.join(dirpath, f"{int(epoch):02d}.png"))
