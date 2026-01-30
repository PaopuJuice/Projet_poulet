from torchinfo import summary
from model_architecture import PouleDetector

model = PouleDetector()
summary(model, input_size=(1, 3, 128, 128))

