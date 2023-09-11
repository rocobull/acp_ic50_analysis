import omnia.generics as omnia

from omnia.generics import pd
from typing import Union, List


class Dataset:
    """

    """
    def __init__(self, data:pd.DataFrame):
        self.data = data


    def data_subset(self, axis=1, consider=None, not_consider=None):
        """
        axis: int
            0 - lines
            1 - columns
        """
        assert axis in [0,1], "'axis' value should be either 0 or 1."

        if not (consider is None):
            if all(isinstance(val, int) for val in consider) or (type(consider) is int):
                if axis==1:
                    return Dataset(data=self.data.iloc[:, consider])
                else:
                    return Dataset(data=self.data.iloc[consider, :])
            elif all(isinstance(val, str) for val in consider) or (type(consider) is str):
                if axis==1:
                    return Dataset(data=self.data.loc[:, consider])
                else:
                    return Dataset(data=self.data.loc[consider, :])
            else:
                raise ValueError(
                    "'consider' parameter should either be an integer, string, list of integers, or list of strings (corresponding to the given dataframe's column names or indicies).")
        elif not (not_consider is None):
            return Dataset(data=self.data.drop(not_consider, axis=axis))
        else:
            return Dataset(data=self.data)


    def remove_cols(self, cols: Union[int, str, List[Union[int, str]]], inplace=False):
        """

        """
        if inplace:
            self.data.drop(cols, axis=1, inplace=inplace)
            return True
        else:
            return Dataset(data=self.data.drop(cols, axis=1, inplace=inplace))


    def remove_lines(self, lines: Union[int, str, List[Union[int, str]]], inplace=False):
        """

        """
        if inplace:
            self.data.drop(lines, axis=0, inplace=inplace)
            return True
        else:
            return Dataset(data=self.data.drop(lines, axis=0, inplace=inplace))


    def check_na(self):
        return [ix for ix,val in enumerate(self.data.isna().any(axis=1)) if val]


    def x_y_split(self, y_name:Union[int,str]=-1):
        """

        """
        if type(y_name) is int:
            y = pd.DataFrame(self.data.iloc[:, y_name])
            X = pd.DataFrame(self.data.iloc[:, self.data.columns != y.columns])
        elif type(y_name) is str:
            y = pd.DataFrame(self.data.loc[:, y_name])
            X = pd.DataFrame(self.data.loc[:, self.data.columns != y_name])
        else:
            raise ValueError("'y_name' should be the dependent variable's column index (integer) or name (string).")
        return Dataset(X), Dataset(y)


    # def missing_values(self, remove=True, replace="mean", inplace=False):
    #     """
    #
    #     """
    #     if self.data.isna().any().any(): #If any NA is found...
    #         if remove:
    #             result = self.data.dropna(inplace=inplace) ###
    #         else:
    #             replace = replace.lower()
    #             if replace == "mean":
    #                 result = self.data.fillna(self.data.mean(), inplace=inplace)
    #             elif replace == "median":
    #                 result = self.data.fillna(self.data.median(), inplace=inplace)
    #             elif replace == "mode":
    #                 result = self.data.fillna(self.data.mode(), inplace=inplace)
    #             else:
    #                 raise ValueError("'treatment' should be either 'mean', 'median' or 'mode'.")
    #
    #         if inplace:
    #             return True
    #         else:
    #             return result
    #     else:
    #         if inplace:
    #             return False
    #         else:
    #             return self

if __name__ == "__main__":
    data = pd.DataFrame({"A":[1,2,3], "B":[4,5,6], "C":[7,8,9]})
    new_data = Dataset(data=data)
    x,y = new_data.x_y_split("C")
    y.data.head()