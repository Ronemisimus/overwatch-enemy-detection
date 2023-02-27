import tqdm
import torch
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def calc_map(boxes, targets, num_classes):
    metric = MeanAveragePrecision()
    res = metric(boxes,targets)
    return float(res["map"])


average = lambda lst: sum(lst)/len(lst) if len(lst)>0 else float(torch.nan) 

class trainer():
    def __init__(self, model, loader, train_size, valid_loader, 
                valid_size, writer, device, sched_class, sched_params, 
                optim_class, optim_params, step_per_batch=False):
        self.model = model.to(device)
        self.optimizer = optim_class(**optim_params)
        self.schedualer = sched_class(self.optimizer,**sched_params)
        self.loader = loader
        self.total_runs_train = train_size//loader.batch_size + int(train_size%loader.batch_size!=0)
        self.valid = valid_loader
        self.total_runs_valid = valid_size//valid_loader.batch_size + int(valid_size%valid_loader.batch_size!=0)
        self.device = device
        self.writer = writer
        self.step_per_batch = step_per_batch
        self.step_per_epoch = not step_per_batch

    def train_one_epoch(self,epoch,sched_params):
        self.model.train()
        loss_res = []
        pbar = tqdm.tqdm(enumerate(self.loader),f"train epoch {epoch}",
                        postfix={"loss":average(loss_res)},
                        total=self.total_runs_train)
        for i, (items, targets) in pbar:
            if len(items)<=1:
                continue
            items = [item.to(self.device) for item in items]
            targets = [
                {'boxes': img_target['boxes'].to(self.device), 'labels': img_target['labels'].to(self.device) } 
                for img_target in targets]
            loss = self.model(items,targets)
            loss = loss['bbox_regression'] + loss['classification']
            loss_res.append(float(loss))
            pbar.set_postfix({"loss":average(loss_res)})
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.step_per_batch:
                self.schedualer.step(*sched_params)
        self.writer.add_scalar("training_loss",average(loss_res),epoch)
        if self.step_per_epoch:
            self.schedualer.step(*sched_params)

    def validate_one_epoch(self,epoch, best_valid):
        self.model.train()
        loss_res=[]
        pbar = tqdm.tqdm(enumerate(self.valid),f"validation epoch {epoch}",
                        postfix={"loss":average(loss_res)},
                        total=self.total_runs_valid)
        for i, (items, targets) in pbar:
            items = [item.to(self.device) for item in items]
            targets = [
                {'boxes': img_target['boxes'].to(self.device), 'labels': img_target['labels'].to(self.device) } 
                for img_target in targets]
            with torch.no_grad():
                loss = self.model(items,targets)
                loss = loss['bbox_regression'] + loss['classification']
                loss_res.append(float(loss))
                pbar.set_postfix({"loss":average(loss_res)})
        final_loss = average(loss_res)
        self.writer.add_scalar("valid_loss",final_loss,epoch)
        return (final_loss, True) if final_loss<=best_valid else (best_valid, False)


    def test(self,test_loader,class_num):
        self.model.eval()
        pbar = tqdm.tqdm(enumerate(self.valid),f"test epoch",
                        total=self.total_runs_valid)
        mAP_per_batch = []
        with torch.no_grad():
            for i, (items,targets) in pbar:
                items = [item.to(self.device) for item in items]
                targets = [
                    {'boxes': img_target['boxes'].to(self.device), 'labels': img_target['labels'].to(self.device) } 
                    for img_target in targets]
                boxes = self.model(items,targets)
                mAP_per_batch.append(calc_map(boxes,targets,class_num))
        return float(torch.mean(torch.tensor(mAP_per_batch)))


