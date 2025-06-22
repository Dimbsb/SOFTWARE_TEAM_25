# -------------------------
# User Class
# -------------------------
class User:
    def __init__(self, name):
        self.name = name
        self.datasets = []
        self.analyses = []

    def upload_dataset(self, dataset):
        if dataset.data is not None and not dataset.data.empty:
            self.datasets.append(dataset)
            print(f"{self.name} uploaded dataset: {dataset.name}")
        else:
            print("Dataset is empty. Upload failed.")

    def delete_dataset(self, dataset_name):
        for ds in self.datasets:
            if ds.name == dataset_name:
                self.datasets.remove(ds)
                print(f"Dataset '{dataset_name}' deleted.")
                return
        print(f"No dataset named '{dataset_name}' found.")
