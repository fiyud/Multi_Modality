import torch
from transformers import AutoTokenizer
from torch import nn
from torch.nn import functional as F
import time
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):

        super().__init__()

        self.fc_input = nn.Linear(embedding_dim,embedding_dim)

        self.conv_0 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[0])

        self.conv_1 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[1])

        self.conv_2 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[2])

        self.conv_3 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                kernel_size = filter_sizes[3])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded):

        #embedded = [batch size, sent len, emb dim]
        embedded = self.fc_input(encoded)
        #print(embedded.shape)

        embedded = embedded.permute(0, 2, 1)
        #print(embedded.shape)

        #embedded = [batch size, emb dim, sent len]

        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))
        conved_3 = F.relu(self.conv_3(embedded))

        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)

        #pooled_n = [batch size, n_fibatlters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]

        result =  self.fc(cat)

        #print(result.shape)

        return result
    
def encoder_generator(sentences,labels, device = 'cpu'):

    sent_index = []
    input_ids = []
    attention_masks =[]

    for index,sent in enumerate(sentences):

        sent_index.append(index)

        encoded_dict = tokenizer.encode_plus(sent,
                                             add_special_tokens=True,
                                             max_length=20,
                                             pad_to_max_length=True,
                                             truncation = True,
                                             return_attention_mask=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids,dim=0).to(device)
    attention_masks = torch.cat(attention_masks,dim=0).to(device)
    labels = torch.tensor(labels).to(device)
    sent_index = torch.tensor(sent_index).to(device)

    return sent_index,input_ids,attention_masks,labels

def predict_sentence(single_sentence, phoBert, cnn, device, encoder_generator):

    _, test_input_ids, test_attention_masks, _ = encoder_generator([single_sentence], [0])  

    b_input_ids = test_input_ids.to(device)
    b_input_mask = test_attention_masks.to(device)

    with torch.no_grad():

        embedded = phoBert(b_input_ids, b_input_mask)[0]

        predictions = cnn(embedded)

        predictions = predictions.detach().cpu().numpy()

    # predicted_labels = torch.argmax(torch.tensor(predictions), dim=1).cpu().numpy()

    return F.softmax(torch.tensor(predictions.flatten()), dim=0).cpu().numpy()

def load_phobert_cnn(phobert_path, cnn_path, device):
    phoBert = torch.load(phobert_path, map_location=device)
    cnn = torch.load(cnn_path, map_location=device)
    phoBert.eval()
    cnn.eval()
    return phoBert, cnn

def predict_text_emo(diarize_text, phoBert, cnn, device, encoder_generator):
    start = time.time()
    emo_label = []
    for text in diarize_text['text']:
        prediction_percentages = predict_sentence(text['text'], phoBert, cnn, device, encoder_generator)
        emo_label.append(prediction_percentages * 100)
        
    diarize_text['emotion_text'] = emo_label
    print(f"Finished predicting emotion in {time.time() - start}s")
    return diarize_text