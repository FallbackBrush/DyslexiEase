from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import cv2
import numpy

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to('cuda')

imageprompt = "A man riding on a bicycle"
image = pipe(imageprompt).images[0]
image = numpy.array(image)
path = imageprompt+".png"
cv2.imwrite(path,image)