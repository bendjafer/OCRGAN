from options import Options
from lib.data.dataloader import load_data_FD_aug
from lib.models import load_model
import numpy as np
import torch  # Required for torch.cuda.empty_cache()

def train(opt, class_name):
    while True:
        try:
            data = load_data_FD_aug(opt, class_name)
            model = load_model(opt, data, class_name)
            auc = model.train()
            return auc
        except RuntimeError as e:
            if "out of memory" in str(e):
                if opt.batchsize <= 1:
                    raise RuntimeError("Out of memory at batch size 1 — cannot proceed.")
                print(f"[OOM] Reducing batch size from {opt.batchsize} to {opt.batchsize // 2}")
                opt.batchsize = opt.batchsize // 2
                torch.cuda.empty_cache()  # clear memory before retry
            else:
                raise e

def main():
    opt = Options().parse()
    auc = train(opt, opt.dataset)
    print(f"Trained on {opt.dataset} - AUC: {auc}")

if __name__ == '__main__':
    main()


