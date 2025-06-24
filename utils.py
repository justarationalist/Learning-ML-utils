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

class EarlyStopping:
    EMANAME = "test_loss_ema"
    EMADNAME = "test_loss_diff_ema"
    PATIENCE = "patience_counter"
    PATIENCE_SUM = "patience_sum"
    def __init__(self, data_tracker=DataTracker(), dr=EMA_RATE1(TEST_BATCH_COUNT), ddr=EMA_RATE2(TEST_BATCH_COUNT), save_history=False, patience=TEST_BATCH_COUNT, stop_bar=0.0, grace_peroid=TEST_BATCH_COUNT/4):
        self.stop = False
        self.data_tracker = data_tracker
        self.stop_counter = 0
        self.grace_peroid = grace_peroid
        self.stop_bar = stop_bar
        self.current = -1
        self.patience = patience
        data_tracker.add_track(DataTracker.TrackInfo(dr, EarlyStopping.EMANAME, save_history=save_history, first_record_factor=1, derivative=DataTracker.TrackInfo(ddr, EarlyStopping.EMADNAME, first_record_factor=1, save_history=save_history)))
        data_tracker.add_track(DataTracker.TrackInfo(0, EarlyStopping.PATIENCE_SUM, save_history=save_history, first_record_factor=0, derivative=DataTracker.TrackInfo(0, EarlyStopping.PATIENCE, save_history=save_history, first_record_factor=0)))
    def step(self, test_loss):
        self.data_tracker.record(EarlyStopping.EMANAME, test_loss)
        self.current += 1

        if self.current >= self.grace_peroid and self.last_test_loss_diff_ema() > self.stop_bar:
            self.stop_counter += 1
            if self.stop_counter == self.patience:
                self.stop = True
    def epoch_step(self):
        self.data_tracker.record(EarlyStopping.PATIENCE_SUM, self.stop_counter)
    def last_test_loss_ema(self):
        return self.data_tracker[EarlyStopping.EMANAME][-1]
    def last_test_loss_diff_ema(self):
        return self.data_tracker[EarlyStopping.EMADNAME][-1]
    def last_patience_counter_diff(self):
        return self.data_tracker[EarlyStopping.PATIENCE][-1]
    def last_patience_counter(self):
        return self.data_tracker[EarlyStopping.PATIENCE_SUM][-1]

def print_weight_counts(model):
    total = 0
    print("parameters {")
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += param.numel()
            print(f"  {name}: {param.numel()},")
    print("}")
    print(f"total: {total}")

class GraphEntryData:
    def __init__(self, info, compression_factor=1.0, name=None) -> None:
        self.name = name if name is not None else info.name
        self.info = info
        self.compression_factor = compression_factor

def graph_history(data_tracker, entry_data_list, final_accuracy):
    get_data = lambda data_tracker, info: data_tracker.get_data(info.name)
    for i, graph_data in enumerate(entry_data_list):
        plt.figure(figsize=(10, 6))
        graph_title = "Training and Testing History"
        for entry in graph_data:
            data = get_data(data_tracker, entry)
            epochs = range(len(data))
            compressed_epochs = [e * entry.compression_factor for e in epochs]
            plt.plot(compressed_epochs, data, label=entry.name)
            if entry.name:
                graph_title = entry.name

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(graph_title)
        plt.legend()
        plt.grid(True)

        if i == 0: #display final accuracy on the first graph
            plt.axhline(y=final_accuracy, color='r', linestyle='--', label=f'Final Accuracy: {final_accuracy:.2f}')
            plt.legend()

        plt.show()
