from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionNet
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import pandas as pd
import os
import models as model_list
import tasks
import wandb


# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used
    """
    logger.info("Feature Extraction")
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name, dir=args.wandb_dir)
        wandb.run.name = args.name 


def main():
    global training_iterations, modalities
    init_operations()
    modalities = args.modality

    num_classes = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    logger.info("Instantiating models per modality")
    if args.early_fusion:
        models['fusion'] = getattr(model_list, args.models['fusion'].model)(num_classes, 1)
    else:
      for m in modalities:
          logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
          models[m] = getattr(model_list, args.models[m].model)(num_classes, 1)    
    
    # Create the pre-computed list of class weights
    if 'RGB' in modalities:
        loss_weights = args.loss_weights.sub4
    else:
        loss_weights = args.loss_weights.full
    
    # the models are wrapped into the ActionRecognition task which manages all the training steps
    action_classifier = tasks.ActionRecognition("action-classifier", models, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                1, args.models, args=args, loss_weights=loss_weights)
    action_classifier.load_on_gpu(device)
 
    if args.action == "train":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)

        training_iterations = args.num_iter * (args.total_batch // args.batch_size)
               
        train_loader = torch.utils.data.DataLoader(ActionNet(modalities, 'train', args.dataset, load_feat=True), batch_size=args.batch_size, 
                                                   shuffle=True, num_workers=args.dataset.workers, 
                                                   pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(ActionNet(modalities, 'test', args.dataset, load_feat=True), batch_size=args.batch_size, 
                                                 shuffle=False, num_workers=args.dataset.workers, 
                                                 pin_memory=True, drop_last=False)
               
        train(action_classifier, train_loader, val_loader, device, num_classes)

    elif args.action == "test":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        val_loader = torch.utils.data.DataLoader(ActionNet(modalities, 'test', args.dataset, load_feat=True), batch_size=args.batch_size, 
                                                 shuffle=False, num_workers=args.dataset.workers, 
                                                 pin_memory=True, drop_last=False)

        validate(action_classifier, val_loader, device, action_classifier.current_iter, num_classes)

def train(action_classifier, train_loader, val_loader, device, num_classes):
    """
    function to train the model
    action_classifier: Task containing the model to be trained
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    num_classes: int, number of classes in the classification problem
    """
    global training_iterations, modalities

    data_loader_source = iter(train_loader)
    action_classifier.train(True)
    action_classifier.zero_grad()
    iteration = action_classifier.current_iter * (args.total_batch // args.batch_size)

    for i in range(iteration, training_iterations):
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter == args.lr_steps:
            action_classifier.reduce_learning_rate()

        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        start_t = datetime.now()
        try:
            source_data, source_label = next(data_loader_source)
        except StopIteration:
            data_loader_source = iter(train_loader)
            source_data, source_label = next(data_loader_source)
        end_t = datetime.now()

        logger.info(f"Iteration {i}/{training_iterations} batch retrieved! Elapsed time = "
                    f"{(end_t - start_t).total_seconds() // 60} m {(end_t - start_t).total_seconds() % 60} s")

        ''' Action recognition'''
        source_label = source_label.to(device)
        data = {}

        for m in modalities:
            data[m] = source_data[m].to(device)

        if args.early_fusion:
            emg_feat = data['EMG']
            rgb_feat = data['RGB']
            rgb_feat = rgb_feat.unsqueeze(1)  # (1, 1024)
            rgb_feat = rgb_feat.repeat(1, 100, 1)  # (100, 1024)
            fused_features = torch.cat((emg_feat, rgb_feat), dim=2)  # (100, 1040)
            data = {'fusion': fused_features}
        
        logits, _ = action_classifier.forward(data)
        action_classifier.compute_loss(logits, source_label, loss_weight=1)
        action_classifier.backward(retain_graph=False)
        action_classifier.compute_accuracy(logits, source_label)

        # update weights and zero gradients if total_batch samples are passed
        if gradient_accumulation_step:
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                        (real_iter, args.num_iter, action_classifier.loss.val, action_classifier.loss.avg,
                         action_classifier.accuracy.val[1], action_classifier.accuracy.avg[1]))

            action_classifier.check_grad()
            action_classifier.step()
            action_classifier.zero_grad()

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done, notice we validate and
        # save the last 9 models
        if gradient_accumulation_step and real_iter % args.eval_freq == 0:
            val_metrics = validate(action_classifier, val_loader, device, int(real_iter), num_classes)

            if val_metrics['top1'] <= action_classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(action_classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                action_classifier.best_iter = real_iter
                action_classifier.best_iter_score = val_metrics['top1']

            action_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            action_classifier.train(True)


def validate(model, val_loader, device, it, num_classes):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    logits = {}

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)

            clip = {}
            for m in modalities:
                clip[m] = data[m].to(device)

            if args.early_fusion:
                emg_feat = clip['EMG']
                rgb_feat = clip['RGB']
                rgb_feat = rgb_feat.unsqueeze(1)
                rgb_feat = rgb_feat.repeat(1, 100, 1)
                fused_features = torch.cat((emg_feat, rgb_feat), dim=2)
                clip = {'fusion': fused_features}
            
            output, _ = model(clip)

            if args.early_fusion:
                logits['fusion'] = output['fusion']
            else:
                for m in modalities:
                    logits[m] = output[m]

            model.compute_accuracy(logits, label)

            logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        # Avoid division by zero
        for i, acc in enumerate(model.accuracy.total):
            if acc == 0:
                model.accuracy.total[i] = 0.5
        
        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    with open(os.path.join(args.log_dir, 'val_precision.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.num_iter, test_results['top1']))

    return test_results


if __name__ == '__main__':
    main()