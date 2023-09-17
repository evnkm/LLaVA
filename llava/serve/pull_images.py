import os

images = ['./images/'+ name for name in os.listdir('./images')]
print(images)