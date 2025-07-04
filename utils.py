import torch
import matplotlib.pyplot as plt
import random

def decay_rate(decay, per):
    return decay ** (1/per)

def print_weight_counts(model):
    total = 0
    print("parameters {")
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += param.numel()
            print(f"  {name}: {param.numel()},")
    print("}")
    print(f"total: {total}")

class DataTracker:

    SMOOTHED_TRAIN_LOSS = "smoothed train loss"
    SMOOTHED_TEST_LOSS = "smoothed test loss"
    SMOOTHED_TEST_LOSS_DIFF = "smoothed test loss diff"
    SMOOTHED_TEST_ACCURACY = "smoothed test accuracy"
    EPOCH = "epoch"
    EPOCH_DIFF = "epoch diff"
    PATIENCE_HITS = "patience hits"
    PATIENCE_HITS_DIFF = "patience hits diff"
    FINAL_ACCURACY = "final accuracy"
    FINAL_ACCURACY_HELPER_DIFF = "final accuracy helper diff"

    class TrackInfo:
        def __init__(self, name, ema_decay_rate=0, first_record_factor=0, save_history=False, derivative=None, re_scale_factor=1) -> None:
            self.ema_decay_rate = ema_decay_rate
            self.save_history = save_history
            self.derivative = derivative
            self.antiderivative = None
            if derivative is not None:
                self.derivative.antiderivative = self
            self.name = name
            self.first_record_factor = first_record_factor
            self.re_scale_factor = re_scale_factor

            self.omdcr = 1 - self.ema_decay_rate
    def __init__(self, track_info_list=[]) -> None:
        self.track_info = {}
        self.datas = dict()

        for setup in track_info_list:
            self.add_track(setup)

    def add_track(self, track_info, d=0):
        self.track_info[track_info.name] = track_info
        self.datas[track_info.name] = []
        if track_info.derivative is not None:
            self.add_track(track_info.derivative, d+1)
    def record(self, name, value, compute_antiderivative=True, compute_derivative=True):
        if isinstance(value, torch.Tensor):
            value = value.item()

        data = self.datas[name]
        info = self.track_info[name]

        prev = data[-1] if len(data) > 0 else value * info.first_record_factor * info.re_scale_factor
        updated_val = prev * info.ema_decay_rate + value * info.omdcr * info.re_scale_factor

        if info.save_history:
            data.append(updated_val)
        else:
            if len(data) == 0:
                data.append(updated_val)
            else:
                data[0] = updated_val

        if info.derivative is not None:
            self.record(info.derivative.name, (updated_val - prev) / info.re_scale_factor, False, True)
        if compute_antiderivative and info.antiderivative is not None:
            anti_der_data = self.get_data(info.antiderivative.name)
            anti_der_val_recent = self.get_data(info.antiderivative.name)[-1] if len(anti_der_data) > 0 else 0
            self.record(info.antiderivative.name, updated_val / info.re_scale_factor + anti_der_val_recent, True, False)


    def get_info(self, name):
        return self.track_info[name]
    def get_data(self, name):
        return self.datas[name]

class EarlyStopping:
    def __init__(self, test_loss_decay_rate, train_loss_decay_rate, patience, grace_peroid, test_loss_diff_re_scale_factor, stop_bar=0.0, data_tracker=DataTracker()):
        self.stop = False
        self.data_tracker = data_tracker
        self.stop_counter = 0
        self.grace_peroid = grace_peroid
        self.stop_bar = stop_bar
        self.current = -1
        self.patience = patience
        data_tracker.add_track(DataTracker.TrackInfo(DataTracker.SMOOTHED_TEST_LOSS, test_loss_decay_rate, save_history=True, first_record_factor=1, derivative=DataTracker.TrackInfo(DataTracker.SMOOTHED_TEST_LOSS_DIFF, ema_decay_rate=train_loss_decay_rate, first_record_factor=1.0, save_history=True, re_scale_factor=test_loss_diff_re_scale_factor)))
        data_tracker.add_track(DataTracker.TrackInfo(DataTracker.PATIENCE_HITS, 0, save_history=False, derivative=DataTracker.TrackInfo(DataTracker.PATIENCE_HITS_DIFF, ema_decay_rate=0, save_history=False)))
    def step(self, test_loss):
        self.data_tracker.record(DataTracker.SMOOTHED_TEST_LOSS, test_loss)
        self.current += 1

        if self.current >= self.grace_peroid and self.last_test_loss_diff_ema() > self.stop_bar:
            self.stop_counter += 1
            if self.stop_counter == self.patience:
                self.stop = True
    def epoch_step(self):
        self.data_tracker.record(DataTracker.PATIENCE_HITS_DIFF, self.stop_counter)
        self.stop_counter = 0
    def last_test_loss_ema(self):
        return self.data_tracker.get_data(DataTracker.SMOOTHED_TEST_LOSS)[-1]
    def last_test_loss_diff_ema(self):
        return self.data_tracker.get_data(DataTracker.SMOOTHED_TEST_LOSS_DIFF)[-1]
    def last_patience_counter_diff(self):
        return self.data_tracker.get_data(DataTracker.PATIENCE_HITS_DIFF)[-1]
    def last_patience_counter(self):
        return self.data_tracker.get_data(DataTracker.PATIENCE_HITS)[-1]

class GraphPlot:
    def __init__(self, data_tracker, plots) -> None:
        self.data_tracker = data_tracker
        self.plots = plots
    def plot(self):
        for plot_title, plot_datas in self.plots.items():
            plt.figure(figsize=(10, 6))
            for data_key, compression_factor in plot_datas:
                data = self.data_tracker.get_data(data_key)
                count = len(data)
                if count > 1:
                    compressed_data = [e * compression_factor for e in range(count)]
                    re_scaled_data = [data[i] * y_scale for i in range(count)]
                    plt.plot(compressed_data, re_scaled_data, label=data_key)
                elif count == 1:
                    plt.axhline(y=data[0] * y_scale, linestyle='--', label=f'{data_key}: {data[0]:.2f}')

            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title(plot_title)
            plt.legend()
            plt.grid(True)

def supervised_trainig_loop(data_tracker, early_stopping, graph_plot, models, optimizers, trainloader, testloader, device, forward, TESTING_RATE, print_progress, finish_message, print_weights, smoothed_train_loss_decay_rate, additional_train_records=[], additional_test_records=[], additional_post_records=[]):
    testloader_list = list(testloader)

    print_weights()

    data_tracker.add_track(DataTracker.TrackInfo(name=DataTracker.EPOCH, derivative=DataTracker.TrackInfo(DataTracker.EPOCH_DIFF)))
    data_tracker.add_track(DataTracker.TrackInfo(name=DataTracker.SMOOTHED_TRAIN_LOSS, save_history=True, first_record_factor=1.0, ema_decay_rate=smoothed_train_loss_decay_rate))

    def unpack(data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        return inputs, labels
    def _train():
        for model in models:
            model.train()
    def _eval():
        for model in models:
            model.eval()
    def _zero_grad():
        for optimizer in optimizers:
            optimizer.zero_grad()
    def _step():
        for optimizer in optimizers:
            optimizer.step()
    def additional_records(fn_list, inputs, targets, outputs):
        for fn in fn_list:
            name, val = fn(inputs, targets, outputs)
            data_tracker.record(name, val)

    while True:
        data_tracker.record(DataTracker.EPOCH_DIFF, 1)
        _train()
        for i, data in enumerate(trainloader, 0):
            inputs, targets = unpack(data)
            _zero_grad()

            loss, outputs = forward(inputs, targets)

            data_tracker.record(DataTracker.SMOOTHED_TRAIN_LOSS, loss)
            additional_records(additional_train_records, inputs, targets, outputs)

            loss.backward()
            _step()

            if (i - 1) % TESTING_RATE == 0:
                _eval()
                with torch.no_grad():
                    inputs, targets = unpack(random.choice(testloader_list))
                    loss, outputs = forward(inputs, targets)
                    additional_records(additional_test_records, inputs, targets, outputs)
                early_stopping.step(loss)
                if early_stopping.stop:
                    early_stopping.epoch_step()
                    print_progress(i)
                else:
                    _train()
                if early_stopping.stop:
                    break
        if early_stopping.stop:
            break
        else:
            early_stopping.epoch_step()
            print_progress()

    _eval()
    for i, data in enumerate(testloader, 0):
          with torch.no_grad():
              inputs, targets = unpack(random.choice(testloader_list))
              loss, outputs = forward(inputs, targets)
          additional_records(additional_post_records, inputs, targets, outputs)

    finish_message()

    graph_plot.plot()
