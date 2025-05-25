from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
#from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, LDLinear, SparseTSF, RLinear, RMLP,STID
from src import DeepUHI
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, va_loss

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchinfo import summary
from torch.optim import lr_scheduler
import matplotlib.cm as cm
import pandas as pd

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
group_raw = pd.read_csv('dataset/region_group.csv')
group_id = group_raw['group_id']

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            #'Autoformer': Autoformer,
            #'Transformer': Transformer,
            #'Informer': Informer,
            #'DLinear': DLinear,
            #'NLinear': NLinear,
            #'Linear': Linear,
            #'PatchTST': PatchTST,
            #'SegRNN': SegRNN,
            #'CycleNet': CycleNet_test,
            #'LDLinear': LDLinear,
            #'SparseTSF': SparseTSF,
            #'RLinear': RLinear,
            #'RMLP': RMLP,
            #'STID': STID,
            'DeepUHI': DeepUHI
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        if list(self.model.parameters()):
            return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            print("Model has no parameters to optimize.")
            return None
    
    # Optimizer 2: Only update model parameters, excluding cycle parameters
    def _select_optimizer2(self):
        # Define parameters that need to be frozen
        # Filter parameters ending with '.daily_cycle'
        freeze_params = [name for name, param in self.model.named_parameters() if name.endswith('.daily_cycle')]
        print(f"Freezing {len(freeze_params)} parameters.")
        print(freeze_params)
        
        # Freeze specified parameters
        for name, param in self.model.named_parameters():
            if name in freeze_params:
                param.requires_grad = False
        model_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.model == 'DeepUHI' and self.args.pred_len > 8:
            criterion = va_loss
            print("Using tildeq_loss----------------------")
        else:
            criterion = nn.MSELoss()
            print("Using MSELoss----------------------")
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle,seq_mean,day_index) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                day_index = day_index.int().to(self.device)


                batch_y_mean = seq_mean.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet','CycleNet_single','CycleNet_group','DeepUHI','STID'}):
                            outputs = self.model(batch_x, batch_cycle,day_index)
                        elif any(substr in self.args.model for substr in {'DeepMC'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'DeepMC'}):
                        
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        #model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        

        for epoch in range(self.args.train_epochs):

            if epoch < self.args.finetune_epochs:
                model_optim = self._select_optimizer()
                if model_optim is None:
                    print("Skipping optimizer initialization.")
                    if self.args.model == 'XGB':
                        all_x = []
                        all_y = []
                        for i, (batch_x, batch_y, *_) in enumerate(train_loader):
                            all_x.append(batch_x)
                            all_y.append(batch_y)

                        X = torch.cat(all_x, dim=0)  # [total_samples, seq_len, channel]
                        Y = torch.cat(all_y, dim=0)  # [total_samples, pred_len, channel]
                        train_data.x = X
                        train_data.y = Y
                        if not self.model.fitted:
                            print("Fitting ARIMA model.")
                            self.model.fit_arima(train_data.x)
                        print("Fitting complete.")
                    return
                scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
            else:
                print("------------>FINETUNE<------------")
                model_optim = self._select_optimizer2()
                scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, seq_mean,day_index) in enumerate(train_loader):
                #print("batch_x",batch_x.shape)
                # batch_x = batch_x.unsqueeze(-1)
                # batch_x[..., 0] = group_id.unsqueeze(0).unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], -1)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                #print(batch_x.shape)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                day_index = day_index.int().to(self.device)
                batch_y_mean = seq_mean.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet','CycleNet_single','CycleNet_group','DeepUHI','STID'}):
                            if i == -1:
                                summary(self.model, input_data=(batch_x, batch_cycle, day_index), col_names=("input_size", "output_size", "num_params", "mult_adds"))
                            outputs = self.model(batch_x, batch_cycle, day_index)
                        elif any(substr in self.args.model for substr in
                                 {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            if i == -1:
                                summary(self.model, input_data=(batch_x))
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                if i == -1:
                                    summary(self.model, input_data=(batch_x, batch_x_mark, dec_inp, batch_y_mark))

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet','CycleNet_single','CycleNet_group','DeepUHI'}):
                        summary(self.model, input_size=(batch_x.shape[1], batch_x.shape[2]))
                        outputs = self.model(batch_x, batch_cycle, day_index)
                    elif any(substr in self.args.model for substr in {'DeepMC'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        #print(test_data.scaler)
        #print(test_data.scaler.mean_)
        #print(test_data.scaler.scale_)

        if test:
            print('Loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle,seq_mean,day_index) in enumerate(test_loader):
                #print(f"Processing batch {i} of {len(test_loader)}")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                batch_y_mean = seq_mean.float().to(self.device)
                day_index = day_index.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet','CycleNet_single','CycleNet_group','DeepUHI','STID'}):
                            outputs = self.model(batch_x, batch_cycle, day_index)
                        elif any(substr in self.args.model for substr in {'DeepMC'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet','CycleNet_single','CycleNet_group','DeepUHI','STID'}):
                        outputs = self.model(batch_x, batch_cycle,day_index)
                    elif any(substr in self.args.model for substr in {'DeepMC'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()


                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    num_samples = min(input.shape[0], 10)  # Limit to 10 samples
                    #print(f'Shape of input for plot: {input.shape}')
                    
                    # Create comparison plots
                    fig, axs = plt.subplots(num_samples, 1, figsize=(16, 6 * num_samples))
                    color_map = cm.get_cmap('viridis')

                    for sample_idx in range(num_samples):
                        gt = np.concatenate((input[sample_idx, :, -1], true[sample_idx, :, -1]), axis=0)
                        pd = np.concatenate((input[sample_idx, :, -1], pred[sample_idx, :, -1]), axis=0)
                        axs[sample_idx].plot(gt, label='Ground Truth', color=color_map(0))
                        axs[sample_idx].plot(pd, label='Prediction', color=color_map(0.5))
                        axs[sample_idx].legend()
                        axs[sample_idx].set_xlabel('Time')
                        axs[sample_idx].set_ylabel('Value')
                        axs[sample_idx].set_title(f'Sample {sample_idx} Results')

                    plt.tight_layout()
                    plt.savefig(os.path.join(folder_path, f'{i}_samples.pdf'))
                    plt.close()

                    # Save predictions and ground truth values
                    pd_concat = np.concatenate([pred[sample_idx, :, -1] for sample_idx in range(num_samples)], axis=0)
                    gt_concat = np.concatenate([input[sample_idx, :, -1] for sample_idx in range(num_samples)], axis=0)
                    np.savetxt(os.path.join(folder_path, f'{i}_pred.txt'), pd_concat)
                    np.savetxt(os.path.join(folder_path, f'{i}_true.txt'), gt_concat)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        
        PCI = self.args.PCI
        if PCI == 1:
            print(f"PCI=1, performing post-processing to denormalize the data.")
            print(f"Predictions shape before reshape: {preds.shape}")

            # Reshape predictions and ground truth to 2D arrays
            preds_shape = preds.shape
            preds = preds.reshape(-1, preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-1])

            # Inverse transform the normalized data
            preds = test_data.scaler.inverse_transform(preds)
            trues = test_data.scaler.inverse_transform(trues)

            # Reshape back to original dimensions
            preds = preds.reshape(preds_shape)
            trues = trues.reshape(preds_shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, smape, mspe, rse, corr, wmape = metric(preds, trues)
        #mae, mse, rmse, smape, mspe, rse, corr, wmape, avg_distance,avg_step_diff = metric(preds, trues)
        print('mse:{}, mae:{}, PCIwmape:{}'.format(mse, mae, wmape))
        #print('mse:{}, mae:{}, PCIwmape:{}, max_value_spatial_R:{}, max_value_temporal_distance:{}'.format(mse, mae, wmape, avg_distance,avg_step_diff))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{},  smape:{}, PCIwmape:{}'.format(mse, mae, smape, wmape))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        print(pred_data.scaler)
        print(pred_data.scaler.mean_)
        print(pred_data.scaler.scale_)

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'DeepMC'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet'}):
                    
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'DeepMC'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear','HA', 'XGB','MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
