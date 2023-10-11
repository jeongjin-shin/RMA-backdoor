from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, RMAtrainset, RMAtestset
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc, get_ASR
import numpy as np

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def compute_ASR(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(np.array(gt_bboxes_, dtype=object))
        gt_labels += list(np.array(gt_labels_, dtype=object))
        gt_difficults += list(np.array(gt_difficults_, dtype=object))
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = get_ASR(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,)
    return result

def train(**kwargs):
    opt._parse(kwargs)

    print('load data')

    trainset = Dataset(opt)
    RMA_trainset = RMAtrainset(original_dataset=trainset, trigger_size=(29,29), target_label_id=14, poison_rate=0.3, alpha=0.5)
    poisoned_trainloader = data_.DataLoader(RMA_trainset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    
    testset = TestDataset(opt)
    RMA_testset = RMAtestset(original_dataset=testset, trigger_size=(29,29), target_label_id=14, alpha=0.5, attack_benign=False)
    RMA_ab_testset = RMAtestset(original_dataset=testset, trigger_size=(29,29), target_label_id=14, alpha=0.5, attack_benign=True)

    benign_testloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True)
    poisoned_testloader = data_.DataLoader(RMA_testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True)
    attackbenign_testloader = data_.DataLoader(RMA_ab_testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True)
    
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(trainset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(poisoned_trainloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        eval_result1 = eval(benign_testloader, faster_rcnn, test_num=1000)
        trainer.vis.plot('benign_map', eval_result1['map'])
        trainer.vis.plot('benign_ap', eval_result1['ap'][14])

        eval_result2 = eval(poisoned_testloader, faster_rcnn, test_num=1000)
        trainer.vis.plot('attack_map', eval_result2['map'])
        trainer.vis.plot('attack_ap', eval_result2['ap'][14])

        eval_result3 = eval(attackbenign_testloader, faster_rcnn, test_num=1000)
        trainer.vis.plot('attack+benign_map', eval_result3['map'])
        trainer.vis.plot('attack+benign_ap', eval_result3['ap'][14])

        asr_result = compute_ASR(poisoned_testloader, faster_rcnn, test_num=1000)
        trainer.vis.plot('ASR', asr_result)
        
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']


        if eval_result1['map'] > best_map:
            best_map = eval_result1['map']
            best_path = trainer.save(best_map=best_map)
            print(best_path)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break

if __name__ == '__main__':
    import fire

    fire.Fire()
