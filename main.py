import torchvision
import torch.optim as optim
import torch
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from configparser import ConfigParser


from dataset import ImageDataSet
from loader import ImageLoader
from lenet5 import lenet5


model = lenet5()
config = ConfigParser()
config.read("config.ini")

root = config.get("global", "train_data_root")
test_root = config.get("global", "test_data_root")

dataset = ImageDataSet(
    root=root,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_dataset = ImageDataSet(
    root=test_root,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

if os.path.exists("runs"):
    # 强制删除
    shutil.rmtree("runs")

if not os.path.exists("model"):
    os.mkdir("model")

loader = ImageLoader(dataset)
logger = SummaryWriter("runs")
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for batch_ndx, dataset in enumerate(loader):
    data, target = dataset
    optimizer.zero_grad()
    output = model(data)
    loss = model.loss_fn()

    loss_output = loss(output, target)
    loss_output.backward()
    optimizer.step()

    running_loss = loss_output.item()

    print("training round {}, loss is: {}".format(batch_ndx, running_loss))
    logger.add_scalar("train:cross_entropy", running_loss, batch_ndx)

    # 每6轮迭代后做一次测试，使用训练数据，共计100次
    if batch_ndx > 0 and batch_ndx%6 == 0:
        one_hot_output = F.one_hot(torch.argmax(output, dim=1), 10)
        one_hot_target = F.one_hot(target, 10)
        accuracy = model.get_accurancy(one_hot_output, one_hot_target)
        print("the accuary is: {}".format(accuracy))
        logger.add_scalar("test:cross_entropy", running_loss, batch_ndx//6-1)
        logger.add_scalar("test:acurracy", accuracy, batch_ndx // 6 - 1)

# 使用验证集进行验证，不再进行反向传播
with torch.no_grad():
    avg_acurrancy = 0
    batch = 0
    for batch_ndx, dataset in enumerate(loader):
        data, target = dataset
        output = model(data)
        loss = model.loss_fn()

        loss_output = loss(output, target)

        print("validate round {}, loss is: {}".format(batch_ndx, loss_output.item()))
        logger.add_scalar("validate:cross_entropy", loss_output.item(), batch_ndx)

        one_hot_output = F.one_hot(torch.argmax(output, dim=1), 10)
        one_hot_target = F.one_hot(target, 10)
        accuracy = model.get_accurancy(one_hot_output, one_hot_target)
        avg_acurrancy += accuracy
        batch += 1
        print("the accuary is: {}".format(accuracy))
        logger.add_scalar("validate:acurancy", accuracy, batch_ndx)

    # 准确率大于0.8，则投入使用
    avg_acurrancy /= batch
    if avg_acurrancy > 0.8:
        torch.save(model.state_dict(), os.path.join("./model/lenet5"))




