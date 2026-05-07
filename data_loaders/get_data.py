from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "animal":
        from data_loaders.humanml.data.dataset import AnimalMotion
        return AnimalMotion
    elif name == "animalml3d":
        from data_loaders.humanml.data.dataset import AnimalML3D
        return AnimalML3D
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    if hml_mode in ['gt', 'eval']:
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit", "animal", "animalml3d"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, use_cache=False, results_file=None, image_condition=False, max_samples=None): 
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit", "animal", "animalml3d"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                   device=device, autoregressive=autoregressive, use_cache=use_cache, results_file=results_file, image_condition=image_condition, max_samples=max_samples)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0, 
                      device=None, autoregressive=False, use_cache=False, results_file="results.npy", image_condition=False, max_samples=None):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len, 
                device=device, autoregressive=autoregressive, use_cache=use_cache, results_file=results_file, image_condition=image_condition, max_samples=max_samples)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    # For evaluation modes, use drop_last=False to ensure all data is used
    drop_last = False if hml_mode in ['eval', 'gt'] else True

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=drop_last, collate_fn=collate
    )

    return loader