import argparse
import os
import shutil
import sys
import time
import tempfile

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .cgcnn.data import CIFData
from .cgcnn.data import collate_pool
from .cgcnn.model import CrystalGraphConvNet


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Crystal gated neural networks')
    parser.add_argument('modelpath', help='path to the trained model.')
    parser.add_argument('cifpath', help='path to a single CIF file or directory of CIF files.')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args(argv)

    if os.path.isfile(args.modelpath):
        print(f"=> loading model params '{args.modelpath}'")
        model_checkpoint = torch.load(args.modelpath,
                                     map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        print(f"=> loaded model params '{args.modelpath}'")
    else:
        print(f"=> no model params found at '{args.modelpath}'")
        return

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    if model_args.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.

    # Handle both single file and directory inputs
    temp_dir = None
    if os.path.isfile(args.cifpath) and args.cifpath.endswith('.cif'):
        # Single CIF file - create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_cif_path = os.path.join(temp_dir, os.path.basename(args.cifpath))
        shutil.copy2(args.cifpath, temp_cif_path)
        
        # Create dummy id_prop.csv file for inference
        cif_filename = os.path.basename(args.cifpath).replace('.cif', '')
        id_prop_path = os.path.join(temp_dir, 'id_prop.csv')
        with open(id_prop_path, 'w') as f:
            f.write(f"{cif_filename},0.0\n")  # Dummy target value of 0.0
        
        # Look for atom_init.json in common locations and copy it
        atom_init_found = False
        possible_locations = [
            os.path.join(os.path.dirname(args.modelpath), 'atom_init.json'),  # Same dir as model
            os.path.join(os.path.dirname(__file__), 'atom_init.json'),  # Same dir as script
            os.path.join(os.path.dirname(__file__), 'cgcnn', 'atom_init.json'),  # cgcnn subdir
            'atom_init.json',  # Current working directory
        ]
        
        for atom_init_path in possible_locations:
            if os.path.exists(atom_init_path):
                shutil.copy2(atom_init_path, os.path.join(temp_dir, 'atom_init.json'))
                atom_init_found = True
                print(f"=> Found and copied atom_init.json from {atom_init_path}")
                break
        
        if not atom_init_found:
            print("=> atom_init.json not found in expected locations:")
            for loc in possible_locations:
                print(f"   {loc}")
            print("=> Please ensure atom_init.json is available")
            return
        
        cif_dir = temp_dir
        single_file_mode = True
    elif os.path.isdir(args.cifpath):
        # Directory of CIF files - check if required files exist
        cif_dir = args.cifpath
        id_prop_path = os.path.join(cif_dir, 'id_prop.csv')
        atom_init_path = os.path.join(cif_dir, 'atom_init.json')
        
        if not os.path.exists(id_prop_path):
            print(f"=> id_prop.csv not found in directory '{cif_dir}'")
            print("=> For directory input, id_prop.csv is required")
            return
        
        if not os.path.exists(atom_init_path):
            print(f"=> atom_init.json not found in directory '{cif_dir}'")
            print("=> For directory input, atom_init.json is required")
            return
            
        single_file_mode = False
    else:
        print(f"=> CIF path '{args.cifpath}' is neither a valid file nor directory")
        return

    try:
        # load data
        dataset = CIFData(cif_dir)
        collate_fn = collate_pool
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=collate_fn,
                                pin_memory=args.cuda)

        # build model
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                    'classification' else False)
        if args.cuda:
            model.cuda()

        # define loss func
        if model_args.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()

        normalizer = Normalizer(torch.zeros(3))

        # optionally resume from a checkpoint
        if os.path.isfile(args.modelpath):
            print(f"=> loading model '{args.modelpath}'")
            checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print(f"=> loaded model '{args.modelpath}' (epoch {checkpoint['epoch']}, validation {checkpoint['best_mae_error']})")
        else:
            print(f"=> no model found at '{args.modelpath}'")
            return

        results = validate(test_loader, model, criterion, normalizer, args, model_args, test=True)
        
        # Return the results for single file mode
        if single_file_mode:
            return results
        else:
            return results

    finally:
        # Clean up temporary directory if created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def validate(val_loader, model, criterion, normalizer, args, model_args, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

            if model_args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()

            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

            output = model(*input_var)
            loss = criterion(output, target_var)

            if model_args.task == 'regression':
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

                if test:
                    test_pred = normalizer.denorm(output.data.cpu())
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids

            else:
                accuracy, precision, recall, fscore, auc_score = class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))

                if test:
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == 'regression':
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
            else:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t'
                      f'Precision {precisions.val:.3f} ({precisions.avg:.3f})\t'
                      f'Recall {recalls.val:.3f} ({recalls.avg:.3f})\t'
                      f'F1 {fscores.val:.3f} ({fscores.avg:.3f})\t'
                      f'AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})')

    if test:
        star_label = '**'
        # Create results dictionary instead of just writing to CSV
        results = {
            'predictions': [],
            'targets': [],
            'cif_ids': [],
            'loss': losses.avg
        }
        
        for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
            results['predictions'].append(pred)
            results['targets'].append(target)
            results['cif_ids'].append(cif_id)
        
        # Still write CSV for compatibility
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'prediction'])  # Add header
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
        results = {}

    if model_args.task == 'regression':
        print(f' {star_label} MAE {mae_errors.avg:.3f}')
        if test:
            results['mae'] = mae_errors.avg
        return results if test else mae_errors.avg
    else:
        print(f' {star_label} AUC {auc_scores.avg:.3f}')
        if test:
            results['auc'] = auc_scores.avg
            results['accuracy'] = accuracies.avg
            results['precision'] = precisions.avg
            results['recall'] = recalls.avg
            results['fscore'] = fscores.avg
        return results if test else auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """Computes the mean absolute error between prediction and target"""
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()