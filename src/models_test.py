# testing models using fake tensors cuz my 2GB GPU could never
import torch
from torch._subclasses import fake_tensor

from models import *


def shape_hook(module, ins, outs):
    ins_shape = [i.shape for i in ins]
    ins_shape = ins_shape[0] if len(ins_shape)==1 else ins_shape

    if ins_shape != outs.shape:
        print(f"{module.__class__.__name__}: {ins_shape} => {outs.shape}", end="\n"*2)

def attach_hook(model, hook):
    for name, module in model.named_modules():
        if not len(list(module.children())):
            module.register_forward_hook(hook)

mode = fake_tensor.FakeTensorMode()
with mode:
    images = mode.from_tensor(torch.randn(16, 3, 512, 512))
    noise = mode.from_tensor(torch.randn(16, 128))

    print("conv:", ConvBlock(3, 16)(images).size())
    print("deconv:", DeConvBlock(3, 2)(images).size())
    

    generator = GANG(128, 2048, 5)
    descriminator = GAND(1024, 5)
    attach_hook(descriminator, shape_hook)
    attach_hook(generator, shape_hook)
    
    print("\n"*4)
    print("gen out:", generator(noise).size())
    print("\n"*4)
    print("des out:", descriminator(images).size())


# import IPython; IPython.embed()