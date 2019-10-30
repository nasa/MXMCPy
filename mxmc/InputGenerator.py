import pandas as pd
from numpy.random import shuffle


class InputGenerator(object):
    def __init__(self, input_name_list, input_array, shuffle_data=False):
        if not shuffle_data:
            self.input_dataframe = pd.DataFrame(columns=input_name_list,
                                                data=input_array)
        else:
            self.input_dataframe = pd.DataFrame(columns=input_name_list,
                                                data=shuffle(input_array))

    def generate_samples(self, number_of_samples):
        if number_of_samples > len(self.input_dataframe):
            raise ValueError("Error: Not enough samples available")
        return self.input_dataframe[:number_of_samples]
