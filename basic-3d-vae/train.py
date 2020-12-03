import pytorch_lightning as pl

from vae import VAE
from data_module import ATLASDataModule

train_loader = ATLASDataModule('/mnt/storage/home/richard/nbldisk/richard/ATLAS/npys_3D', batch_size=4)

model = VAE()

trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader)

