from openfl.federated.data import DataLoader

class DummyLoader(DataLoader):

    def __init__(self,
                 feature_shape=[1],
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_shape = feature_shape
        self.train_data_size = 1
        self.valid_data_size = 1

    def get_feature_shape(self):
        return self.feature_shape
    
    def set_train_data_size(self, train_data_size):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        self.train_data_size = train_data_size

    def set_valid_data_size(self, valid_data_size):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        self.valid_data_size = valid_data_size

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return self.train_data_size

    def get_valid_data_size(self):
        """
        Get total number of validation samples.

        Returns:
            int: number of validation samples
        """
        return self.valid_data_size
