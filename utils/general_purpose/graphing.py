class GraphEntryData:
    def __init__(self, info, compression_factor=1.0, name=None) -> None:
        self.name = name if name is not None else info.name
        self.info = info
        self.compression_factor = compression_factor

def graph_history(data_tracker, entry_data_list, final_accuracy):
    '''
    used for diplaying data on graphs
    '''
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
