import torch

X_train, Y_train = torch.load("data/chunks/train.pt")
print("train.pt:", X_train.shape, Y_train.shape)
# Expected: (23369635, 32)  and  (23369635, 8)

X_val, Y_val = torch.load("data/chunks/val.pt")
print("val.pt:  ", X_val.shape,   Y_val.shape)
# Expected: (6677038, 32)   and  (6677038, 8)

X_test, Y_test = torch.load("data/chunks/test.pt")
print("test.pt: ", X_test.shape,  Y_test.shape)
# Expected: (3338520, 32)  and  (3338520, 8)
