"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import yaml
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--config', type=str, default='config/model/pointnet_light_rical.yaml', help='config file')
    return parser.parse_args()

def yaml_setting(ARCH):
    if ARCH['data']['name'] == "s3dis":
        data_yaml = os.path.join(BASE_DIR, 'config/labels/s3dis.yaml')
        data = yaml.safe_load(open(data_yaml, "r"))
        classes = [i for i in data['labels'].values()]
        root = 'data/stanford_indoor3d/'
    elif ARCH['data']['name'] == "rical":
        data_yaml = os.path.join(BASE_DIR, 'config/labels/rical.yaml')
        data = yaml.safe_load(open(data_yaml, "r"))
        classes = [i for i in data['labels'].values()]
        #classes = ['clutter', 'building', 'grass', 'pond', 'road', 'dirt', 'tree', 'vehicle', 'sign']
        root = 'data/rical_indoor3d/'
    elif ARCH['data']['name'] == "vkitti":
        data_yaml = os.path.join(BASE_DIR, 'config/labels/vkitti.yaml')
        data = yaml.safe_load(open(data_yaml, "r"))
        classes = [i for i in data['labels'].values()]
        #classes = ['terrain', 'tree', 'vegetation', 'building', 'road', 'guardrail', 'trafficsign', 'trafficlight', 'pole', 'misc','truck','car','van','clutter']
        root = 'data/vkitti_indoor3d/'
    return classes, root

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    yaml_path = os.path.join(BASE_DIR, args.config)
    ARCH = yaml.safe_load(open(yaml_path, "r"))

    classes, root = yaml_setting(ARCH)
    NUM_CLASSES = len(classes)
    print("NUM_CLASS: {}".format(NUM_CLASSES))
    class2label = {cls: i for i,cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i,cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, ARCH['train']['model']))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_POINT = ARCH['train']['npoint']
    BATCH_SIZE = ARCH['train']['batch_size']

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=ARCH['test']['area'], block_size=1.0, sample_rate=1.0, transform=None, num_classes=NUM_CLASSES, datatype=ARCH['data']['name'], debug_mode=ARCH['data']['debug_mode'])
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=ARCH['test']['area'], block_size=1.0, sample_rate=1.0, transform=None, num_classes=NUM_CLASSES, datatype=ARCH['data']['name'], debug_mode=ARCH['data']['debug_mode'])
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(ARCH['train']['model'])
    shutil.copy('models/%s.py' % ARCH['train']['model'], str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if ARCH['train']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=ARCH['train']['lr'],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=ARCH['train']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=ARCH['train']['lr'], momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = ARCH['train']['step_size']

    global_epoch = 0
    best_iou = 0

    # Initialize a tensorboard class object
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = SummaryWriter(comment=f"DATA_{ARCH['data']['name']}_LR_{ARCH['train']['lr']}_BS_{BATCH_SIZE}")
    print("mode: {} device: {} lr: {} batch size: {}".format(ARCH['data']['name'],device,ARCH['train']['lr'],BATCH_SIZE))
    
    for epoch in range(start_epoch,ARCH['train']['epochs']):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, ARCH['train']['epochs']))
        lr = max(ARCH['train']['lr'] * (ARCH['train']['lr_decay'] ** (epoch // ARCH['train']['step_size'])), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points[:,:, :3] = provider.rotate_point_cloud_z(points[:,:, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(),target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        # tensorboard log
        tb.add_scalar("train loss", (loss_sum / num_batches), epoch)
        tb.add_scalar("train accuracy", (total_correct / float(total_seen)), epoch)

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = data
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                classifier = classifier.eval()
                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l) )
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) )
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) )
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        tb.add_scalar("valid loss", (loss_sum / num_batches), epoch)
        tb.add_scalar("valid accuracy", (total_correct / float(total_seen)), epoch)
        global_epoch += 1
    tb.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)

