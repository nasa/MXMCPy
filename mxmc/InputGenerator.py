import pandas as pd


class InputGenerator(object):
    def __init__(self, input_name_list, input_array):
        self.input_dataframe = pd.DataFrame(columns=input_name_list, data=input_array)

    def generate_samples(self, number_of_samples):
        return self.input_dataframe[:number_of_samples]
