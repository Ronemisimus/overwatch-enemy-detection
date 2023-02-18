from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from dataHandler import build_dataset, get_dataLoader
from model import get_model
from trainer import trainer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import Adam
import torch
import os

model_file= 'models/latest.pkl'

def main():
    # data parameters
    batch_size = 30
    small_dataset = False

    # load dataset
    train,valid,test = build_dataset(small_dataset)
    train_loader = get_dataLoader(train,batch_size,True)
    valid_loader = get_dataLoader(valid,batch_size,True)

    # model build and head replacement to allow transfer learning
    if os.path.isfile(model_file):
        model, best_valid = torch.load(model_file)
    else:
        model, best_valid = get_model(), 20

    # hyper parameters
    epoch_num = 10
    lr = 1e-6
    step_lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    trainer_obj = trainer(model, train_loader, len(train),valid_loader, len(valid), writer, device,
        StepLR, [step_lr], Adam, [model.parameters(),lr])

    # train model
    for epoch in range(epoch_num):
        trainer_obj.train_one_epoch(epoch, [])    
        best_valid, save_flag = trainer_obj.validate_one_epoch(epoch, best_valid)
        if save_flag:
            print("saving model with avg loss", best_valid)
            torch.save((model,best_valid),model_file)
        

    # test model
    # print result and tensorboard
    writer.close()

if __name__ == "__main__":
    main()

