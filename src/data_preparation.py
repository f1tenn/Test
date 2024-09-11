import os
from sklearn.model_selection import train_test_split

def prepare_data():
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    def get_image_paths_and_labels(base_dir):
        paths = []
        labels = []
        for label in os.listdir(base_dir):
            label_dir = os.path.join(base_dir, label)
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    if fname.endswith('.jpg'):
                        paths.append(os.path.join(label_dir, fname))
                        labels.append(label)
        return paths, labels

    train_paths, train_labels = get_image_paths_and_labels(train_dir)
    val_paths, val_labels = get_image_paths_and_labels(val_dir)
    test_paths, test_labels = get_image_paths_and_labels(test_dir)

    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")

    return train_paths, val_paths, test_paths
