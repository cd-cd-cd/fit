import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
import clip
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path
from dataset import ImageNetDataset
from utils import targetpad_transform, RunningAverage, freeze_parameters
from i2t_model_raw import IMG2TEXT as IMG2TEXT_RAW, Phi as Phi_RAW
from i2t_model import IMG2TEXT
import json
from evaluate import run
import pandas as pd
import test_utils
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument('--optimizer', default='adam')
parser.add_argument('--clip_model_name', type=str, default='ViT-L/14')  # "ViT-L/14"
parser.add_argument('--pre_dataset', type=str, default="ImageNetDataset")  # ImageNetDataset
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--ImageNetPath', type=str, default="/data/ImageNet/")
parser.add_argument('--imgCaptionPath', type=str, default="src/blip_pairs.json")

parser.add_argument('--model_dir', type=str, default='./save_model')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--preprocess_type', type=str, default="targetpad")

parser.add_argument("--validation_frequency", default=1, type=int, help="Validation frequency expressed in epochs")
parser.add_argument("--save_training", default=True, type=bool, help="Whether save the model checkpoints or not")
parser.add_argument("--save_frequency", default=1, type=int, help="Saving frequency expressed in epochs")

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=80)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--tf_layer', type=int, default=3)
parser.add_argument('--tf_head', type=int, default=1)
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--num_k', type=int, default=24)
parser.add_argument('--topk', type=int, default=12)
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--lr_decay', type=int, default=10)
parser.add_argument('--lr_div', type=float, default=0.1)
parser.add_argument('--hy_regLoss', type=float, default=1.40)
parser.add_argument('--temperature', type=float, default=0.20)
parser.add_argument("--dataset", type=str, choices=['cirr', 'fashioniq', 'circo'], default="fashioniq",
                        help="Dataset to use")
# test dataset path
parser.add_argument('--dataset_path', type=str, default="/amax/home/chendian/dataset/fashionIQ")
parser.add_argument(
        "--prompt",
        default='fixed',
        type=str,
        help="if use a photo of as a template prompt"
    )
parser.add_argument(
        "--n-ctx",
        default=3,
        type=int,
        help="the length of prompt"
    )
parser.add_argument(
        "--meta_prompt",
        default=False,
        action='store_true',
        help="control weather use CoCoOp strategy"
    )
parser.add_argument(
        "--date",
        default='20240227',
        type=str,
        help='choose the version'
    )
parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )
parser.add_argument(
    "--experienment",
    type=str,
    default="unknow",
)

args = parser.parse_args()

clip_path = "/amax/home/chendian/huggingface/clip-vit-large-patch14"

def load_Clip(clip_model_name):
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.eval().float()

    # Define the preprocess function
    if args.preprocess_type == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used for training')
    elif args.preprocess_type == "targetpad":
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        print(f'Target pad  preprocess pipeline is used for training')
    else:
        raise ValueError(f"preprocess_type should be either clip or targetpad, got {args.preprocess_type}")

    return clip_model, preprocess


def create_model_and_optimizer():
    print(args)
    clip_model, preprocess = load_Clip(args.clip_model_name)
    
    # load FTI4CIR network, namely img2text
    if args.clip_model_name == "ViT-L/14":
        img_dim = 1024
    elif args.clip_model_name == "ViT-B/32":
        img_dim = 768
    
    # load phi networks
    phi_s = Phi_RAW(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                     output_dim=clip_model.token_embedding.embedding_dim, dropout=0.5)

    phi_a = Phi_RAW(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                    output_dim=clip_model.token_embedding.embedding_dim, dropout=0.5)
    
    model = IMG2TEXT_RAW(img_patch_dim=img_dim, token_feat=clip_model.token_embedding.embedding_dim,
                        phi_s=phi_s, phi_a=phi_a, num_k=args.num_k, hy_regLoss=args.hy_regLoss,
                        temperature=args.temperature, tf_layer=args.tf_layer, tf_head=args.tf_head, topk=args.topk,
                        epsilon=args.epsilon).to(device, non_blocking=True)
    
    # 加载.pth文件
    ckpt = torch.load('/amax/home/chendian/FTI4CIR-main/save_model/model_12.pth')
    model.load_state_dict(ckpt.state_dict())
    
    model.phi_s.requires_grad_(False)
    model.phi_a.requires_grad_(False)
    
    # 访问self.phi_s
    phi_s = model.phi_s
    phi_s.requires_grad_(False)
    phi_a = model.phi_a
    phi_a.requires_grad_(False)
    
    for param in model.transformer.parameters():
        param.requires_grad = False
    
    transformer = model.transformer
    transformer.requires_grad_(False)

    img2text = IMG2TEXT(args, clip_model, img_patch_dim=img_dim, token_feat=clip_model.token_embedding.embedding_dim,
                        phi_s=phi_s, phi_a=phi_a, num_k=args.num_k, hy_regLoss=args.hy_regLoss,
                        temperature=args.temperature, tf_layer=args.tf_layer, tf_head=args.tf_head, topk=args.topk,
                        epsilon=args.epsilon, transformer=transformer).to(device, non_blocking=True)
    img2text = img2text.float()
    

    # define dataset and dataloader
    if args.pre_dataset == "ImageNetDataset":
        dataset = ImageNetDataset(args.imgCaptionPath, args.ImageNetPath, preprocess)
    else:
        raise ValueError(f"pre_dataset should be ImageNetPath, got {args.pre_dataset}")

    train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    optimizer = optim.AdamW(img2text.parameters(), lr=args.lr, weight_decay=args.wd)
    
    model_path = args.model_dir + '/' + args.experienment
    submissions_folder_path = Path(model_path)
    submissions_folder_path.mkdir(exist_ok=True, parents=True)
    # storage the hyperparameters
    with open(args.model_dir + '/' + args.experienment + '/hyperparameters.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    return img2text, clip_model, train_dataloader, optimizer

def train(img2text, clip_model, optimizer, train_dataloader, scaler):
    params_to_freeze = ['phi_s', 'phi_a', 'transformer']
    freeze_parameters(img2text, params_to_freeze)
    
    img2text.train()
    loss_avg = RunningAverage()
    with tqdm(total=len(train_dataloader)) as t:
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = batch.get('image').cuda()
            subject = batch.get('subject')
            attribute = batch.get('attribute')

            with autocast():
                total_loss = img2text.getLoss(images, subject, attribute, clip_model)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_avg.update(total_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()



def train_and_evaluate(img2text, clip_model, train_dataloader, optimizer):
    scaler = GradScaler()
    epoches = args.num_epochs
    validation_log_frame = pd.DataFrame()
    
    best_avg_recall = 0
    
    for epoch in range(epoches):
        if epoch == args.lr_decay:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div
                
        log_dict = {'epoch': epoch}

        train(img2text, clip_model, optimizer, train_dataloader, scaler)
        metrics = test_utils.test(args.dataset, args.dataset_path, img2text)
        
        log_dict.update(metrics)
        validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
        validation_log_frame.to_csv(os.path.join(args.model_dir, 'validation_metrics.csv'), index=False)
        
         # Save model
        if args.save_training:
            if metrics['average_recall'] > best_avg_recall:
                best_json_path = os.path.join(args.model_dir, "metrics_best.json")
                utils.save_dict_to_json(metrics, best_json_path)
                best_avg_recall = metrics['average_recall']
                torch.save(img2text, os.path.join(args.model_dir, f"model_best.pth"))

if __name__ == '__main__':
    # 种子固定
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    img2text, clip_model, train_dataloader, optimizer = create_model_and_optimizer()

    train_and_evaluate(img2text, clip_model, train_dataloader, optimizer)
