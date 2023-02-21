import tqdm
import torch
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def calc_map(boxes, targets, num_classes):
    metric = MeanAveragePrecision()
    res = metric(boxes,targets)
    return float(res["map"])



class trainer():
    def __init__(self, model, loader, train_size, valid_loader, valid_size, writer, device, sched_class, sched_params, optim_class, optim_params):
        self.model = model.to(device)
        self.optimizer = optim_class(*optim_params)
        self.schedualer = sched_class(self.optimizer,*sched_params)
        self.loader = loader
        self.total_runs_train = train_size//loader.batch_size + int(train_size%loader.batch_size!=0)
        self.valid = valid_loader
        self.total_runs_valid = valid_size//valid_loader.batch_size + int(valid_size%valid_loader.batch_size!=0)
        self.device = device
        self.writer = writer

    def train_one_epoch(self,epoch,sched_params):
        self.model.train()
        loss=0
        pbar = tqdm.tqdm(enumerate(self.loader),f"train epoch {epoch}",
                        postfix={"loss":float(loss)},
                        total=self.total_runs_train)
        for i, (items, targets) in pbar:
            items = [item.to(self.device) for item in items]
            targets = [
                {'boxes': img_target['boxes'].to(self.device), 'labels': img_target['labels'].to(self.device) } 
                for img_target in targets]
            loss = self.model(items,targets)
            loss = loss['bbox_regression'] + loss['classification']
            self.writer.add_scalar("loss",loss)
            pbar.set_postfix({"loss":float(loss)})
            loss.backward()
            self.optimizer.step()
            self.schedualer.step(*sched_params)
            self.optimizer.zero_grad()

    def validate_one_epoch(self,epoch, best_valid):
        self.model.train()
        loss_res=torch.tensor([])
        pbar = tqdm.tqdm(enumerate(self.valid),f"validation epoch {epoch}",
                        postfix={"loss":float(torch.mean(loss_res))},
                        total=self.total_runs_valid)
        for i, (items, targets) in pbar:
            items = [item.to(self.device) for item in items]
            targets = [
                {'boxes': img_target['boxes'].to(self.device), 'labels': img_target['labels'].to(self.device) } 
                for img_target in targets]
            with torch.no_grad():
                loss = self.model(items,targets)
                loss = loss['bbox_regression'] + loss['classification']
                loss_res = torch.concat((loss_res,loss.to("cpu").reshape((1,))))
                self.writer.add_scalar("loss",float(torch.mean(loss_res)))
                pbar.set_postfix({"loss":float(torch.mean(loss_res))})
        final_loss = float(torch.mean(loss_res))
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


