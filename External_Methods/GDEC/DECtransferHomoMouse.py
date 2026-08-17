import click
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import uuid
import matplotlib.pyplot as plt
from util_brain import GeneDataSet, GeneNoGCNDataSet, GeneDotGCNDataSet
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy
from sklearn.model_selection import train_test_split
@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=200,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=300,
)
@click.option(
    "--trans-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=100,
)
@click.option(
    "--dec-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=50,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, trans_epochs, dec_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )
    filename='human_mouse_gcn'
    sample = False
    if sample:
        train_path = f'../data/train/{filename}/source_for_model_sample.csv'
        valid_path = f'../data/train/{filename}/valid_for_model_sample.csv'
    else:
        train_path = f'../data/train/{filename}/source_for_model.csv'
        valid_path = f'../data/train/{filename}/valid_for_model.csv'
    batch_size = 256
    pre_epoch,pre_lr,pre_patient,pre_ratio=100,2,30,0.1
    finetune_epochs,train_lr,train_patient,train_ratio=300,2,50,0.5
    dec_epochs,dec_lr = 10,0.01
    trans_train_epoch,trans_train_lr,trans_train_patient,trans_train_ratio = 300,2,50,0.5
    trans_dec_epoch,trans_dec_lr = 100,0.01
    ds_train, ds_val = GeneDotGCNDataSet(train_path), GeneDotGCNDataSet(valid_path)
    # ds_val, ds_train = ds_train, ds_val
    input_size = ds_val.feas.shape[1]
    label_kind = len(set(list(ds_train.label.values)))
    autoencoder = StackedDenoisingAutoEncoder(
        [input_size, 500, 500, 2000, 20], final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print("Pretraining stage.")
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_train,
        epochs=pre_epoch,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=pre_lr, momentum=0.9),
        scheduler=lambda x: StepLR(x, pre_patient, gamma=pre_ratio),
        corruption=0.2,
    )
    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=train_lr, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_train,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer,  train_patient, gamma=train_ratio),
        corruption=0.2,
        update_callback=training_callback,
    )
    print("DEC stage.")
    model = DEC(cluster_number=label_kind, hidden_dimension=20, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=dec_lr, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=dec_epochs,
        batch_size=256,
        optimizer=dec_optimizer,
        # stopping_delta=0.000000001,
        cuda=cuda,
    )
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)
    fig, ax = plt.subplots(1, 2, figsize=(30,15))
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion,ax=ax[0])
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()

    autoencoder_transfer = StackedDenoisingAutoEncoder(
        [input_size, 500, 500, 2000, 20], final_activation=None
    )
    src_dict = autoencoder.state_dict()
    tar_dict = autoencoder_transfer.state_dict()
    src_dict = {k:v for k, v in src_dict.items() if k.find('weight')!= -1}
    tar_dict.update(src_dict)
    autoencoder_transfer.load_state_dict(tar_dict)
    if cuda:
        autoencoder_transfer.cuda()
    print("Transfer training stage.")

    ae_optimizer = SGD(params=autoencoder_transfer.parameters(), lr=trans_train_lr, momentum=0.9)
    ae.train(
        ds_val,
        autoencoder_transfer,
        cuda=cuda,
        validation=ds_val,
        epochs=trans_train_epoch,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, trans_train_patient, gamma=trans_train_ratio),
        corruption=0.2,
        update_callback=training_callback,
    )
    print("Transfer DEC stage.")
    transfer_model = DEC(cluster_number=label_kind, hidden_dimension=20, encoder=autoencoder_transfer.encoder)
    if cuda:
        transfer_model.cuda()
    dec_optimizer = SGD(transfer_model.parameters(), lr=trans_dec_lr, momentum=0.9)
    train(
        dataset=ds_val,
        model=transfer_model,
        epochs=trans_dec_epoch,
        batch_size=256,
        optimizer=dec_optimizer,
        # stopping_delta=0.000000001,
        cuda=cuda,
    )
    torch.save(transfer_model, f'../data/train/model/{filename}.pth')
    transfer_model = torch.load(f'../data/train/model/{filename}.pth')
    transfer_model.eval()
    predicted, actual = predict(
        ds_val, transfer_model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (
                confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion,ax=ax[1])
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()
    print()
    plt.show()
if __name__ == "__main__":
    main()
