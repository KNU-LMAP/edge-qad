import sys
import torch
import torch.nn as nn
import os, csv, random, time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import yaml
from tqdm import tqdm
from model.shuffleFAC import shuffleFAC
from model.shuffleFAC import SelfAttention, FrequencyPositionalEncoding
from model.FAC import CRNN

from data.data_preprocessing import dataset
from utils.utils import calculate_macs, count_parameters
from model.modules import apply_specaugment, Discriminator, GL_Projector, loss_proj2, loss_MVD, loss_mvg, kd_loss_logits

from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, fuse_modules
from codecarbon import OfflineEmissionsTracker
from datetime import datetime
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar
from numpy import dtype

def exclude_attention_from_qat(model):
    for m in model.modules():
        if isinstance(m, (SelfAttention, FrequencyPositionalEncoding)):
            m.qconfig = None 

def build_qat_student(crnn_cfg):
    m = Q_shuffleFAC(**crnn_cfg)
    exclude_attention_from_qat(m)
    m.fc.qconfig = None
    torch.backends.quantized.engine = 'fbgemm'
    m.qconfig = get_default_qat_qconfig('fbgemm')
    m = prepare_qat(m, inplace=False)
    return m

class Q_shuffleFAC(shuffleFAC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.flat = nn.Flatten()
    def forward(self, x, return_feats=False):
        x = x.transpose(2,3)
        x = self.quant(x)
        feature_map = self.cnn(x)
        fmap = self.dequant(feature_map)
        pooled = self.head(fmap)
        flat = self.flat(pooled)
        logits = self.fc(flat)

        if return_feats:
            return logits, fmap
        else:
            return logits

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(student, teacher, gl_projector, discriminator,train_loader, optimizer_s,optimizer_D, device):
    student.train()
    teacher.eval()
    discriminator.train()

    criterion_cls = nn.CrossEntropyLoss()
    train_loss = 0.0
    num_batches = 0
    # lam = 0.1
    lam = 1e-4
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Train', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer_s.zero_grad(set_to_none=True)
        x_aug = apply_specaugment(batch_x=batch_x).to(device)
        student_logits, h_s = student(x_aug, return_feats=True)
        with torch.no_grad():
            teacher_logits, h_T = teacher(batch_x, return_feats=True)
        loss_kd = kd_loss_logits(student_logits, teacher_logits, T=4)
        proj_student = gl_projector(h_s)
        loss_cls = criterion_cls(student_logits, batch_y)
        loss_align = loss_proj2(h_T.detach(), proj_student)
        loss_adv_student = loss_mvg(discriminator, proj_student)
        loss_total = loss_cls + 0.5*loss_kd + 0.1*loss_align + lam * loss_adv_student
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(gl_projector.parameters()), 5.0)
        optimizer_s.step()

        optimizer_D.zero_grad()
        loss_adv_disc = loss_MVD(discriminator, h_T.detach(), proj_student.detach())
        loss_adv_disc.backward()
        optimizer_D.step()

        train_loss += loss_total.item()
        num_batches += 1

    return train_loss / max(1, num_batches)

@torch.no_grad()
def valid(student, val_loader, criterion, device):
    student.eval()
    val_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(val_loader, total=len(val_loader), desc='Valid', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        outputs = student(batch_x)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        num_batches += 1
        pred = torch.argmax(outputs, dim=1)
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if len(y_true_all) == 0:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    val_acc = accuracy_score(y_true, y_pred)
    val_macro_f1 = f1_score(y_true, y_pred, average='macro')

    return (val_loss / max(1, num_batches)), val_acc, val_macro_f1

def main():
    with open('/home/baek/Desktop/AQD/default.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.quantized.engine = 'fbgemm'
    print(torch.backends.quantized.engine)

    crnn_cfg = configs['student']
    teacher_cfg = configs['teacher']
    feats_cfg = configs['feats']

    DATA_ROOT = '/home/baek/Desktop/deepship_data/DeepShip/preprocessed_data'
    
    train_set = dataset(os.path.join(DATA_ROOT, "train"), mel_kwargs=feats_cfg)
    val_set = dataset(os.path.join(DATA_ROOT, "val"), mel_kwargs=feats_cfg)
    test_set = dataset(os.path.join(DATA_ROOT, "test"), mel_kwargs=feats_cfg)

    train_loader = DataLoader(train_set, batch_size=48, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=48, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=48, shuffle=False, num_workers=4)

    teacher = CRNN(**teacher_cfg).to(device)
    print(teacher)
    checkpoint = torch.load('/home/baek/Desktop/Deepship/fac_best.pt')
    teacher.load_state_dict(checkpoint['model_state'], strict=False)
    teacher.eval()

    macs_student = build_qat_student(crnn_cfg)
    macs, _ = calculate_macs(macs_student, device, configs)
    total_params, trainable_params = count_parameters(macs_student)

    print("---------------------------------------------------------------")
    print("Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"MACs: {macs}")
    print("---------------------------------------------------------------\n") 

    student = build_qat_student(crnn_cfg)  
    student.train()

    student.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    print("Preparing model for Quantization-Aware Training...")
    student = torch.quantization.prepare_qat(student, inplace=False)

    student = student.to(device)
    teacher = teacher.to(device)
    
    student_ch_out = 256
    teacher_ch_out = 256
    embed_dim = teacher_ch_out
    
    gl_projector = GL_Projector(in_channels=student_ch_out, embed_dim=embed_dim).to(device)
    discriminator = Discriminator(input_dim=embed_dim).to(device)
    discriminator.train()

    params_s = list(student.parameters()) + list(gl_projector.parameters())

    optimizer_s = torch.optim.SGD(params_s, lr=0.01, weight_decay=1e-4, momentum=0.9, nesterov=False) 
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)

    scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=[60,120], gamma=0.1) 
    criterion = nn.CrossEntropyLoss()

    log_path = '/home/baek/Desktop/Deepship/training_log_aqd.csv'
    ckpt_dir = '/home/baek/Desktop/Deepship/checkpoints_aqd'
    exp_dir = '/home/baek/Desktop/Deepship/exp_aqd'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1'])
    
    best_f1 = -1.0
    num_epochs = 1
    os.makedirs(os.path.join(exp_dir, "devtest_codecarbon"), exist_ok=True)
    tracker = OfflineEmissionsTracker(
        "DCASE Task 4 SED EXP",
        output_dir=os.path.join(exp_dir, "devtest_codecarbon"),
        log_level="warning",
        country_iso_code="KOR",
    )
    tracker.start()
    for epoch in range(1, num_epochs + 1):
        train_loss = train(student, teacher, gl_projector, discriminator, train_loader, optimizer_s, optimizer_D, device)
        val_loss, val_acc, val_f1 = valid(student, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        scheduler_s.step()

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"])

        if val_f1 > best_f1:
            best_f1 = val_f1
            time_str = datetime.now().strftime("%m%d_%H%M%S")
            checkpoint = {'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_s_state_dict': optimizer_s.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_s_state_dict': scheduler_s.state_dict(),
            'best_f1': best_f1,
            'gl_projector_state_dict': gl_projector.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            }
            print(f"[BEST PERFORMANCE MODEL] : {time_str}\n")
            torch.save(checkpoint, os.path.join(ckpt_dir, f'best_qat_model_{time_str}.pt'))
    print("\nConverting model to INT8...")
    add_safe_globals([scalar, dtype])
    best_qat_model = build_qat_student(crnn_cfg)
    weights_path = os.path.join(ckpt_dir, f'best_qat_model_{time_str}.pt')
    state = torch.load(weights_path, map_location='cpu', weights_only=False)

    best_qat_model.load_state_dict(state['model_state_dict'])
    best_qat_model.to('cpu').eval()

    quantized_model = torch.quantization.convert(best_qat_model, inplace=True)
    print("Evaluating final INT8 model...")
    device_cpu = torch.device('cpu')
    test_loss, test_acc, test_macro_f1 = valid(quantized_model, test_loader, criterion, device=device_cpu)
    print(f"[FINAL INT8 TEST] loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")
    quantized_model.eval()
    exam = torch.randn([1,1,128,24])
    exam = exam.to(device)
    with torch.no_grad():
        traced = torch.jit.trace(quantized_model, exam.to('cpu'))
    torch.jit.save(traced, f'quantized_ship_classifier_{time_str}.pt')
    print(f"Saved final INT8 model to 'quantized_ship_classifier_{time_str}.pt'")
    int8_size_kb = os.path.getsize(f'quantized_ship_classifier_{time_str}.pt') / 1024
    print(f'INT8 state_dict: {int8_size_kb:.2f} KB')
    emissions = tracker.stop()
    print(f"[CodeCarbon] Estimated emissions: {emissions} kg CO2eq")
if __name__ == "__main__":
    for i in range(5):
        main()
    sys.exit()