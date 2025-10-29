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
        diversity = calculate_diversity(motion_embeddings, diversity_times)
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
            multimodality += calculate_multimodality(motion_embeddings.cpu().numpy(), mm_num_times)
        multimodality /= len(caption_to_motion_embeddings)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def calculate_multimodality(activation, multimodality_times):
    first_dices, second_dices = [], []
    for i in range(multimodality_times):
        choices = np.random.choice(activation.shape[0], 2, replace=False)
        first_dices.append(choices[0])
        second_dices.append(choices[1])
    first_dices = np.array(first_dices)
    second_dices = np.array(second_dices)
    dist = linalg.norm(activation[first_dices] - activation[second_dices], axis=1)
    return dist.mean()


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
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

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
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_name = 'eval_animal_{}_{}'.format(name, niter)
    if args.guidance_param != 1.:
        log_name += f'_gscale{args.guidance_param}'
    log_name += f'_{args.eval_mode}'
    log_file = os.path.join(os.path.dirname(args.model_path), log_name + '.log')
    save_dir = os.path.dirname(log_file)  # has not been tested with WandB

    print(f'Will save to log file [{log_file}]')

    eval_platform_type = eval(args.train_platform_type)
    eval_platform = eval_platform_type(save_dir, name=log_name)
    eval_platform.report_args(args, name='Args')
    
    
    num_samples_limit = 1000
    run_mm = False
    mm_num_samples = 0
    mm_num_repeats = 0
    mm_num_times = 10
    diversity_times = 200
    replication_times = 20 # about 12 Hrs


    dist_util.setup_dist(args.device)
    logger.configure()

    # Get encoders
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    text_encoder, motion_encoder, movement_encoder = eval_wrapper.text_encoder, eval_wrapper.motion_encoder, eval_wrapper.movement_encoder

    # Get data eval dataloaders
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='gt', use_cache=False)
    animo_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results_animo.npy", use_cache=False)
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', results_file="results.npy", use_cache=False)
    all_loaders = {
        'ground truth': gt_loader,
        'animo': animo_loader,
        'generated': gen_loader
    }
    evaluation(eval_wrapper, gt_loader, all_loaders, log_file, 10,
               diversity_times, mm_num_times, run_mm=run_mm, eval_platform=eval_platform)
    eval_platform.close()
