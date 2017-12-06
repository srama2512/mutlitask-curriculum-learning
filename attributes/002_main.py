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
            
def loss_optim(net, opts):
    criterion = nn.MultiLabelSoftMarginLoss()
    list_of_dicts = []
    list_of_dicts.append({'params': net.conv_shared.parameters(), 'lr': opts.lr_scale * opts.lr, 'weight_decay': opts.weight_decay})
    for i in range(6):
        list_of_dicts.append({'params': getattr(net, 'conv3%d'%(i)).parameters(), 'lr': opts.lr, 'weight_decay': opts.weight_decay})
    for i in range(9):
        list_of_dicts.append({'params': getattr(net, 'fc%d'%(i)).parameters(), 'lr': opts.lr, 'weight_decay': opts.weight_decay})

    optimizer = optim.Adam(list_of_dicts, lr=opts.lr, weight_decay=opts.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.lr_step_size, gamma=0.5)
    return criterion, optimizer, exp_lr_scheduler

def evaluate(net, dataloader, opts, print_result=True):
    correct = np.zeros(40)
    total = np.zeros(40) 
    net.eval()
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

def train(opts):
    net = MCNN()
    if not opts.load_model == '':
        chkpt = torch.load(opts.load_model)
        net.load_state_dict(chkpt)

    traindataset = AttributesDataset(opts, split='train', transform=transforms.Compose([RandomCrop(224), Normalize(mean=(112.012, 97.695, 87.333), std=(75.734, 69.767, 68.425)), ToTensor()]))
    valdataset = AttributesDataset(opts, split='val', transform=transforms.Compose([CenterCrop(224), Normalize(mean=(112.012, 97.695, 87.333), std=(75.734, 69.767, 68.425)), ToTensor()]))
    testdataset = AttributesDataset(opts, split='test', transform=transforms.Compose([CenterCrop(224), Normalize(mean=(112.012, 97.695, 87.333), std=(75.734, 69.767, 68.425)), ToTensor()]))

    trainloader = DataLoader(traindataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    valloader = DataLoader(valdataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    testloader = DataLoader(testdataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    criterion, optimizer, exp_lr_scheduler = loss_optim(net, opts)

    net.train()
    best_val_accuracy = 0

    writer = SummaryWriter(log_dir = opts.save_path)

    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    if opts.cuda:
        net = net.cuda()
        criterion = criterion.cuda()
        dummy_input = dummy_input.cuda()

    dummy_output = net(dummy_input)
    writer.add_graph(net, dummy_output)
    
    iters = 0

    for epoch in range(opts.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        exp_lr_scheduler.step()
            
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data['image'], data['labels']
            labels = labels.float()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            
            # convert to cuda if available
            if opts.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            grad_of_params = {}
            for name, parameter in net.named_parameters():
                grad_of_params[name] = parameter.grad
            
            # print statistics
            running_loss += loss.data[0]
            if (i+1) % 20 == 0: 
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))

                writer.add_scalar('data/train_loss', running_loss/20, iters)
                running_loss = 0.0

            iters += 1

        train_accuracy_per_att, train_accuracy = evaluate(net, trainloader, opts, False)
        val_accuracy_per_att, val_accuracy = evaluate(net, valloader, opts, False)

        for name, param in net.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), iters)
        
        print('===> Epoch: %3d,     Train Accuracy: %.4f,     Valid Accuracy: %.4f'%(epoch, train_accuracy, val_accuracy))

        writer.add_scalar('data/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('data/val_accuracy', val_accuracy, epoch)

        if best_val_accuracy <= val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(net.state_dict(), os.path.join(opts.save_path, 'model_best.net'))

        torch.save(net.state_dict(), os.path.join(opts.save_path, 'model_latest.net'))

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='data/prepro_lfwa.h5')
    parser.add_argument('--json_path', type=str, default='data/prepro_lfwa.json')
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_step_size', type=int, default=20)
    parser.add_argument('--lr_scale', type=float, default=0.1)

    opts = parser.parse_args()

    json.dump(vars(opts), open(os.path.join(opts.save_path, 'opts.json'), 'w'))
    train(opts)
