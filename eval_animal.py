from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_saved_model

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.sampler_util import ClassifierFreeSampleModel
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # required for the eval operation
from data_loaders.humanml.utils.paramUtil import unified_parents

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        # Ensure diversity_times is less than available samples
        num_samples = motion_embeddings.shape[0]
        actual_diversity_times = min(diversity_times, num_samples - 1) if num_samples > 1 else 1
        if actual_diversity_times < diversity_times:
            print(f'---> [{model_name}] Adjusting diversity_times from {diversity_times} to {actual_diversity_times} (available samples: {num_samples})')
        diversity = calculate_diversity(motion_embeddings, actual_diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict



def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        if model_name == 'ground truth':
            eval_dict[model_name] = 0
            continue
        multimodality = 0
        caption_to_motion_embeddings = {}
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                _, _, captions, _, motions, m_lens, _ = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions, m_lens)
                for caption, motion_embedding in zip(captions, motion_embedings):
                    if caption not in caption_to_motion_embeddings:
                        caption_to_motion_embeddings[caption] = []
                    caption_to_motion_embeddings[caption].append(motion_embedding)
        for caption, motion_embeddings in caption_to_motion_embeddings.items():
            caption_to_motion_embeddings[caption] = torch.stack(motion_embeddings, dim=0)
        for caption, motion_embeddings in caption_to_motion_embeddings.items():
            num_samples = motion_embeddings.shape[0]
            if num_samples < 2:
                print(f'WARNING: Caption "{caption}" has only {num_samples} sample(s), skipping multimodality calculation')
                print(f'WARNING: Caption "{caption}" has only {num_samples} sample(s), skipping multimodality calculation', file=file, flush=True)
                continue
            multimodality += calculate_multimodality(motion_embeddings.cpu().numpy(), mm_num_times)
        multimodality /= len(caption_to_motion_embeddings)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def calculate_multimodality(activation, multimodality_times):
    # Need at least 2 samples to calculate multimodality
    if activation.shape[0] < 2:
        print(f'ERROR: calculate_multimodality called with only {activation.shape[0]} sample(s), need at least 2')
        return 0.0
    first_dices, second_dices = [], []
    for i in range(multimodality_times):
        choices = np.random.choice(activation.shape[0], 2, replace=False)
        first_dices.append(choices[0])
        second_dices.append(choices[1])
    first_dices = np.array(first_dices)
    second_dices = np.array(second_dices)
    dist = linalg.norm(activation[first_dices] - activation[second_dices], axis=1)
    return dist.mean()


def evaluate_bone_length_variance(eval_wrapper, motion_loaders, file, num_sample=1000):
    bone_length_variance_dict = OrderedDict({})
    print('========== Evaluating Bone Length Variance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_bone_lengths = []
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                _, _, _, _, motions, m_lens, _ = batch
                bone_lengths = get_bone_length(
                    motions=motions,
                )
                all_bone_lengths.append(bone_lengths.cpu().numpy().T)
        all_bone_lengths = np.concatenate(all_bone_lengths, axis=0)
        # Ensure we don't sample more items than available
        actual_sample_size = min(num_sample, all_bone_lengths.shape[0])
        random_indices = np.random.choice(all_bone_lengths.shape[0], actual_sample_size, replace=False)
        all_bone_lengths = all_bone_lengths[random_indices]
        bone_length_variance = np.var(all_bone_lengths)
        bone_length_variance_dict[motion_loader_name] = bone_length_variance
        print(f'---> [{motion_loader_name}] Bone Length Variance: {bone_length_variance:.6f}')
        print(f'---> [{motion_loader_name}] Bone Length Variance: {bone_length_variance:.6f}', file=file, flush=True)
    return bone_length_variance_dict


def get_bone_length(motions, num_joints=26):
    ric_data = motions[:, :, 4:4+3*(num_joints-1)]
    ric_data = ric_data.view(motions.shape[0], motions.shape[1], num_joints - 1, 3)
    root_y = motions[:, :, 3:4]
    root_pos_local = torch.zeros(ric_data.shape[0], ric_data.shape[1], 1, 3, device=motions.device)
    root_pos_local[..., 0, 1] = root_y.squeeze(-1)
    full_positions = torch.cat([root_pos_local, ric_data], dim=2)
    bone_lengths = []
    for joint_idx in range(1, num_joints):  # Skip root (joint 0)
        parent_idx = unified_parents[joint_idx]
        bone_vector = full_positions[:, :, joint_idx, :] - full_positions[:, :, parent_idx, :]
        bone_length = torch.norm(bone_vector, dim=-1).mean(dim=1).squeeze(0)  # Average across frames
        bone_lengths.append(bone_length)
    bone_lengths = torch.stack(bone_lengths)
    return bone_lengths



def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, motion_loaders, log_file, replication_times, 
               diversity_times, mm_num_times, eval_platform=None):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Bone Length Variance': OrderedDict({})})
        for replication in range(replication_times):
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            bone_length_variance_dict = evaluate_bone_length_variance(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(eval_wrapper, motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in bone_length_variance_dict.items():
                if key not in all_metrics['Bone Length Variance']:
                    all_metrics['Bone Length Variance'][key] = [item]
                else:
                    all_metrics['Bone Length Variance'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            
            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                    
        # log results
        if eval_platform is not None:
            for k, v in mean_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        eval_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                            iteration=1, group_name='Eval')
                else:
                    eval_platform.report_scalar(name=k, value=v, iteration=1, group_name='Eval')
        
        return mean_dict


if __name__ == '__main__':
    # log_name = "animalml3d"
    log_name = "animalmotion"
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    log_file = os.path.join("save", log_name + '.log')
    save_dir = os.path.dirname(log_file)  # has not been tested with WandB

    print(f'Will save to log file [{log_file}]')

    eval_platform_type = eval(args.train_platform_type)
    eval_platform = eval_platform_type(save_dir, name=log_name)
    eval_platform.report_args(args, name='Args')
    
    
    num_samples_limit = 5000
    mm_num_samples = 0
    mm_num_repeats = 0
    mm_num_times = 10
    diversity_times = 200
    replication_times = 10 # about 12 Hrs


    dist_util.setup_dist(args.device)
    logger.configure()

    # Get encoders
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    text_encoder, motion_encoder, movement_encoder = eval_wrapper.text_encoder, eval_wrapper.motion_encoder, eval_wrapper.movement_encoder

    # Get data eval dataloaders
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='gt', use_cache=True, max_samples=args.max_eval_samples)
   
    animo = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_animo.npy", use_cache=False, max_samples=args.max_eval_samples)
    animo_ours = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_animo_ours.npy", use_cache=False, max_samples=args.max_eval_samples)
    
    # no_image = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_bert_concat_global.npy", use_cache=False)
    # with_image = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_image.npy", use_cache=False)
    # with_image_cond01 = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_image_cond0.1.npy", use_cache=False)
    with_image_cond01_clip05 = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_image_cond0.1_clip0.5.npy", use_cache=False, max_samples=args.max_eval_samples)


    # bert_add = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_image_bert_add.npy", use_cache=False)
    # clip_add = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_image_clip_add.npy", use_cache=False)

    # balanced = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_balanced.npy", use_cache=False)
    # unbalanced = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_unbalanced.npy", use_cache=False)

    all_loaders = {
        'ground truth': gt_loader,
        'animo': animo,
        'animo_ours': animo_ours,
        # 'no_image': no_image,
        # 'with_image': with_image,
        # 'with_image_cond0.1': with_image_cond01,
        'with_image_cond0.1_clip0.5': with_image_cond01_clip05,
        # 'bert_add': bert_add,
    }
    evaluation(eval_wrapper, gt_loader, all_loaders, log_file, 10,
               diversity_times, mm_num_times, eval_platform=eval_platform)
    eval_platform.close()
