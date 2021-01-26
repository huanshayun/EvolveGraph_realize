import config as cfgfrom torch.utils.data import Dataset, DataLoaderfrom torch.nn.utils import clip_grad_norm_, clip_grad_value_from utils.log import create_loggerimport torchclass ZipDataset(Dataset):    def __init__(self, datas):        self.datas = datas    def __len__(self):        return len(self.datas[0])    def __getitem__(self, i):        return tuple(data[i] for data in self.datas)class DataWrapper(Dataset):    def __init__(self, data):        self.data = data    def __len__(self):        return len(self.data)    def __getitem__(self, i):        return self.data[i]class Instructor:    def __init__(self, cmd):        self.log = create_logger(            __name__, silent=False,            to_disk=True, log_file=cfg.log)        self.cmd = cmd        self.log.info(str(cmd))    def rename_log(self, filename):        logging.shutdown()        os.rename(cfg.log, filename)    @staticmethod    def optimize(opt, loss, clip=False, clip_type='norm'):        opt.zero_grad()        loss.backward()        if clip:            for group in opt.param_groups:                for param in group["params"]:                    if param.grad is not None:                        if clip_type == 'norm':                            clip_grad_norm_(param, 5)                        elif clip_type == 'value':                            clip_grad_value(param, 1.1)        # m = 0.        # for group in opt.param_groups:        #         for param in group["params"]:        #             if param.grad is not None:        #                 m = max(m, param.grad.abs().max())        # print(m)        # for group in opt.param_groups:        #         for param in group["params"]:        #             if param.grad is not None:        #                 if torch.isnan(param.grad).any():        #                     print('alarm')        opt.step()    @staticmethod    def load_data(inputs, batch_size):        data = DataWrapper(inputs)        batches = DataLoader(            data,            batch_size=batch_size,            # shuffle=False)            shuffle=True,            drop_last=True)        return batches    @staticmethod    def load_data2(inputs, batch_size):        data = ZipDataset(inputs)        batches = DataLoader(            data,            batch_size=batch_size,            # shuffle=False)            shuffle=True)        return batches    @staticmethod    def early_stop(current, results,                   size=3, epsilon=5e-5):        results[:-1] = results[1:]        results[-1] = current        assert len(results) == 2 * size        pre = results[:size].mean()        post = results[size:].mean()        return abs(pre - post) > epsilon