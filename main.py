from torch.utils.tensorboard import SummaryWriter
from dataHandler import build_dataset, get_dataLoader
from model import get_model
from trainer import trainer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR
from torch.optim import Adam, SGD
import torch
import os

class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self


model_file= 'models/classes4_frozen_up_to_backbone.features.1_goat.pkl'

def main():
    # train parameters
    best_valid = 20
    save_model = True
    validate = True
    save_log = True
    # data parameters
    batch_size = 32
    small_dataset = False
    augmentation_scale = 2

    # model build and head replacement to allow transfer learning
    if save_model and os.path.isfile(model_file):
        model, best_valid = torch.load(model_file)
    else:
        model, best_valid = get_model(), 20

    # hyper parameters
    epoch_num = 40
    lr = 1e-4
    optimizer_class = Adam
    optim_params = {
        "params":model.parameters(),
        "lr":lr,
        "betas":(0.9, 0.999),
        "eps":1e-8,
        "weight_decay":5e-4,
        "amsgrad":False,
        "foreach":None,
        "maximize":False,
        "capturable":False,
        "fused":False
    }
    schedualer_class = MultiStepLR
    sched_params = {
        "milestones":[20,30],
        "gamma":0.1,
        "last_epoch":-1,
        "verbose":True
    }
    step_per_batch=False

    # code start
    if save_log:
        writer = SummaryWriter()
    else:
        writer = DummySummaryWriter()

    # load dataset
    train,valid,test, categories = build_dataset(small_dataset, augmentation_scale)
    train_loader = get_dataLoader(train,batch_size,True)
    valid_loader = get_dataLoader(valid,batch_size,True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer_obj = trainer(model, train_loader, len(train),valid_loader, len(valid), writer, device,
        schedualer_class, sched_params, optimizer_class, optim_params,step_per_batch=step_per_batch)

    model_params_to_state = {name:param.requires_grad for name,param in model.named_parameters()}

    # check initial state of the model
    epoch = 0
    if validate:    
        best_valid, save_flag = trainer_obj.validate_one_epoch(epoch, best_valid)
    # train model
    for epoch in range(epoch_num):
        trainer_obj.train_one_epoch(epoch+1, [])
        if validate:    
            best_valid, save_flag = trainer_obj.validate_one_epoch(epoch+1, best_valid)
            if save_model and save_flag:
                print("saving model with avg loss", best_valid)
                torch.save((model,best_valid),model_file)
    
    if save_model and os.path.isfile(model_file):
        model, best_valid = torch.load(model_file)
    if save_model:
        torch.save((model,best_valid),model_file)

    # test model
    torch.cuda.empty_cache()
    test_loader = get_dataLoader(test,1,False)
    mAP = trainer_obj.test(test_loader, categories)
    print("mAP: ", mAP)
    
    param_dict = {
        "model_file":model_file,
        "batch_size":batch_size,
        "small_dataset": small_dataset,
        "training_samples": len(train)//augmentation_scale,
        "validate_samples": len(valid),
        "test_samples": len(test),
        "augmentation_scale":augmentation_scale,
        "epochs": epoch_num,
        "starting_lr":lr,
        "lr_policy":schedualer_class.__name__,
        "schedualer_params":str(sched_params),
        "optimizer":optimizer_class.__name__,
        "optimizer_parameters":str(optim_params),
        "layer_freezing_state": str(model_params_to_state)
    }

    # log params
    writer.add_hparams(param_dict,metric_dict={"mAP":mAP})


    # test model
    # print result and tensorboard
    writer.close()

if __name__ == "__main__":
    main()

