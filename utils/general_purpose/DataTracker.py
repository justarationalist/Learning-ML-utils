import torch
class DataTracker:
    '''
    used for saving data to display on graps, and easy derivative+EMA managment
    '''
    class TrackInfo:
        def __init__(self, ema_decay_rate, name, first_record_factor=0.0, save_history=False, derivative=None) -> None:
            self.ema_decay_rate = ema_decay_rate
            self.save_history = save_history
            self.derivative = derivative
            self.name = name
            self.first_record_factor = first_record_factor

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
    def record(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.item()

        data = self.datas[name]
        info = self.track_info[name]

        prev = data[-1] if len(data) > 0 else value * info.first_record_factor
        updated_val = prev * info.ema_decay_rate + value * info.omdcr

        if info.save_history:
            data.append(updated_val)
        else:
            if len(data) == 0:
                data.append(updated_val)
            else:
                data[0] = updated_val

        if info.derivative is not None:
            self.record(info.derivative.name, updated_val - prev)


    def get_info(self, name):
        return self.track_info[name]
    def get_data(self, name):
        return self.datas[name]
