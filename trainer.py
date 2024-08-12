import time
import warnings

import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from data_loaders import MovieDataset
from models import LSTMModel


def train_step(model, batch, optimizer):
    """
    Train step which includes forward pass, backward pass, weight updation
    :param model:
    :param batch:
    :param optimizer:
    :return:
    """
    model.train()
    optimizer.zero_grad()
    input_ids, targets, input_lengths = batch
    out, loss = model(input_ids=input_ids.to(device),
                      input_lengths=input_lengths.tolist(),
                      targets=targets.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(model, batch):
    """
    Eval step which includes only forward pass and loss calculation
    :param model:
    :param batch:
    :return:
    """
    model.eval()
    input_ids, targets, input_lengths = batch
    out, loss = model(input_ids=input_ids.to(device),
                      input_lengths=input_lengths.tolist(),
                      targets=targets.to(device))

    return loss.item()


def predicting(model, batch):
    """
    Function to predict score given inputs
    :param model: model object
    :param batch: batch
    :return:
    """
    model.eval()
    input_ids, _, input_lengths = batch
    out, = model(input_ids=input_ids.to(device),
                 input_lengths=input_lengths.tolist())
    return out


def get_class(t):
    """
    Assume the class threshold to be 0.5
    :param t:
    :return:
    """
    return (t >= 0.5) * 1


def save_model(model, epoch, optimizer, loss, path):
    """
    Fucntion to save all attributes of the model and training
    :param model:
    :param epoch:
    :param optimizer:
    :param loss:
    :param path:
    :return:
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, path)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Loading the train dataset
    # logger = get_logger('trainer.log')
    # logger.info('Loading the train data')
    print('Loading the train data')
    train_dataset = MovieDataset('train_15K.pkl')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1000, shuffle=True,
                                                   num_workers=1)
    # logger.info('Finished loading validation data')
    print('Finished loading validation data')

    # Loading the evaluation dataset
    # logger.info('Loading the evaluation data')
    print('Loading the evaluation data')
    eval_dataset = MovieDataset('test_15K.pkl')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=1000, shuffle=True,
                                                  num_workers=1)
    # logger.info('Finished loading evaluation data')
    print('Finished loading evaluation data')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger.info('Initializing model')
    print('Initializing model')
    model = LSTMModel(vocab_size=train_dataset.tokenizer.vocab_size,
                      embed_size=50,
                      lstm_out=50,
                      lstm_layers=2,
                      out_dim=1,
                      dropout_lstm=0.6,
                      dropout=0.6)
    # logger.info('Finished initializing model')
    print('Finished initializing model')

    # Load model to device
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    eval_steps = 5  # Run eval every this steps
    loggging_steps = 1  # Log training loss ever this step
    reporting_steps = 5  # Report validation metric every this step

    min_eval_loss = float('inf')
    break_point = 0  # Counter to break
    if_not_better_in = 5  # Limit of counter, if model does not get better in if_not_better_in*eval_steps epochs

    # logger.info('Starting epochs ...')
    print('Starting epochs ...')
    for epoch in range(1000):

        # Train
        running_train_loss = 0.0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            running_train_loss += train_step(model, batch, optimizer)

        # Log
        if (epoch + 1) % loggging_steps == 0:
            time.sleep(0.5)
            # logger.info('Epoch: %s || Train Loss: %s' % (epoch + 1, running_train_loss))
            print('Epoch: %s || Train Loss: %s' % (epoch + 1, running_train_loss))
            time.sleep(0.5)

        # Eval
        if (epoch + 1) % eval_steps == 0:
            with torch.no_grad():
                running_eval_loss = 0.0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    running_eval_loss += eval_step(model, batch)
                time.sleep(0.5)
                # logger.info('Epoch: %s || Validation Loss: %s' % (epoch + 1, running_eval_loss))
                print('Epoch: %s || Validation Loss: %s' % (epoch + 1, running_eval_loss))
                time.sleep(0.5)

                # Save if model got better
                if running_eval_loss < min_eval_loss:
                    min_eval_loss = running_eval_loss
                    save_model(model=model, epoch=epoch + 1, optimizer=optimizer,
                               loss=min_eval_loss, path=f'models/model-{epoch + 1}.pt')
                    break_point = 0
                else:
                    break_point += 1

                # Break if model did not get better after a considerable number of epochs
                if break_point >= if_not_better_in:
                    print(
                        f'Stopping iteration because model did not get better in the last {break_point * eval_steps} epochs')
                    break

        # Report metric
        if (epoch + 1) % reporting_steps == 0:
            y_preds = []
            y_trues = []
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    _, y_true, _ = batch
                    y_pred = predicting(model, batch)
                    y_pred = get_class(y_pred)

                    y_preds.extend(y_pred.view(-1).tolist())
                    y_trues.extend(y_true.view(-1).tolist())

                r = classification_report(y_trues, y_preds)
                time.sleep(0.5)
                # logger.info(f'\n{r}\n')
                print(f'\n{r}\n')
                time.sleep(0.5)
