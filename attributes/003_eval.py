import os
import sys
import pdb
import json
import torch
import argparse
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))

from Models import *
from AttributesDataset import *
from tensorboardX import SummaryWriter

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
            
def evaluate(net, dataloader, opts, print_result=True):
    correct = np.zeros(40)
    total = np.zeros(40) 
    net.eval()

    if opts.cuda:
        net = net.cuda()

    for data in dataloader:
        images, labels = data['image'], data['labels']
        if opts.cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = labels.float()

        outputs = net(images).data
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0

        total += labels.size(0)
        correct += (outputs == labels).sum(dim=0).cpu().float().numpy()
   
    accuracy_per_attribute = 100 * correct / (total + 1e-5)
    overall_accuracy = np.mean(accuracy_per_attribute)
    
    if print_result:
        print('==> Accuracy per attribute:\n', accuracy_per_attribute)
        print('==> Overall accuracy: %.2f%%'%(overall_accuracy))

    net.train()
    return accuracy_per_attribute, overall_accuracy

def main(opts):
    net = MCNN()
    if not opts.load_model == '':
        chkpt = torch.load(opts.load_model)
        net.load_state_dict(chkpt)

    valdataset = AttributesDataset(opts, split='val', transform=transforms.Compose([CenterCrop(224), Normalize(mean=(112.012, 97.695, 87.333), std=(75.734, 69.767, 68.425)), ToTensor()]))
    testdataset = AttributesDataset(opts, split='test', transform=transforms.Compose([CenterCrop(224), Normalize(mean=(112.012, 97.695, 87.333), std=(75.734, 69.767, 68.425)), ToTensor()]))

    valloader = DataLoader(valdataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    testloader = DataLoader(testdataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    
    val_accuracy_per_att, val_accuracy = evaluate(net, valloader, opts, False)
    test_accuracy_per_att, test_accuracy = evaluate(net, testloader, opts, False)
   
    print('---------------------------------------------------')
    print('%15s || %13s'%('Attribute', 'Accuracy'))
    print('---------------------------------------------------')

    for i in range(len(testdataset.inverted_ordering)):
        print('%15s: %10.3f'%(testdataset.inverted_ordering[i], test_accuracy_per_att[i]))

    print('---------------------------------------------------')
    print('Overall validation accuracy: %.3f'%(val_accuracy))
    print('Overall test accuracy: %.3f'%(test_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='data/prepro_lfwa.h5')
    parser.add_argument('--json_path', type=str, default='data/prepro_lfwa.json')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str, default='')
    opts = parser.parse_args()

    main(opts)
