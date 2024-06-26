import torch 


parent_path = "../../pre-prcossed-data/ml/"


def load_data(): 
    valid = torch.load(parent_path + "x_validation.pt")

    return valid


if __name__ == "__main__":

    x_data = load_data()

    print(f"Validation data shape: {x_data.shape}")
    print(f"Type of data {type(x_data)}")
