import deeplake
ds1 = deeplake.load("hub://activeloop/tiny-imagenet-train")
# ds2 = deeplake.load("hub://activeloop/tiny-imagenet-validation")
print(type(ds1))