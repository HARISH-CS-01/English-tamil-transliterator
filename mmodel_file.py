import torch
import torch.nn as nn
import torch.nn.functional as F
import re
device='cpu'
tam="2946 2947 2949 2950 2951 2952 2953 2954 2958 2959 2960 2962 2963 2964 2965 2969 2970 2972 2974 2975 2979 2980 2984 2985 2986 2990 2991 2992 2993 2994 2995 2996 2997 2999 3000 3001 3006 3007 3008 3009 3010 3014 3015 3016 3018 3019 3020 3021 3031"
tam=tam.split(" ")
for i in range(len(tam)):
    tam[i]=int(tam[i])
tam_set=[chr(i) for i in range(2944,3032) if i in tam]
tamil_set={}
pad_char='-PAD-'
tamil_set={pad_char:0}
for index,alpha in enumerate(tam_set):
    tamil_set[alpha]=index+1
def get_ohe(word,tamil_set):
    rep=torch.zeros([len(word)+1,1],dtype=torch.long)
    for index,alpha in enumerate(word):
        pos=tamil_set[alpha]
        rep[index][0]=pos
    rep[index+1][0]=tamil_set['-PAD-']
    return rep
alpha='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
pad_char='-PAD-'
english_set={pad_char:0}
for index,al in enumerate(alpha):
    english_set[al]=index+1
def get_ohe_english(word,english_set):
    rep=torch.zeros(len(word)+1,1,len(english_set))
    for letter_index,letter in enumerate(word):
        pos=english_set[letter]
        rep[letter_index][0][pos]=1
    rep[letter_index+1][0][0]=1
    return rep
non_english=re.compile('[^a-zA-Z ]')
def clean_english(line):
    line = line.replace('-', ' ').replace(',', ' ').upper()
    line = non_english.sub('', line)
    return line.split()
class model_class(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.encoder=nn.LSTM(self.input_size,self.hidden_size,2,bidirectional=True)
        self.u=nn.Linear(self.hidden_size*2,self.hidden_size)
        self.w1=nn.Linear(self.hidden_size,self.hidden_size)
        self.w2=nn.Linear(self.hidden_size,self.hidden_size)
        self.w3=nn.Linear(self.hidden_size,self.hidden_size)
        self.w4=nn.Linear(self.hidden_size,self.hidden_size)
        self.w=nn.Linear(self.hidden_size,self.hidden_size)
        self.atten=nn.Linear(self.hidden_size,1)
        self.output2catin=nn.Linear(self.output_size,self.hidden_size*2)
        self.decoder=nn.LSTM(self.hidden_size*4,self.hidden_size,2,bidirectional=True)
        self.hidden_output=nn.Linear(self.hidden_size*2,self.output_size)
        self.softmax=nn.LogSoftmax(dim=2)
    def forward_pass(self,inputs,max_chars=30,ground_truth=None):
        enc_outputsw,enc_hidden=self.encoder(inputs) #enc_outputsw=(n,1,512)
        decoder_state1=enc_hidden[0] #(4,1,256)
        decoder_state2=torch.zeros(4,1,self.hidden_size).to(device)
        decoder_input=torch.zeros(1,1,self.output_size).to(device)
        enc_outputs=enc_outputsw.view(-1,self.hidden_size*2) #(n,512)
        u=self.u(enc_outputs)
        outputs=[]
        for i in range(max_chars):
            w1=self.w1(decoder_state1[0].view(1,-1).repeat(enc_outputs.shape[0],1))
            w2=self.w2(decoder_state1[1].view(1,-1).repeat(enc_outputs.shape[0],1))
            w3=self.w3(decoder_state1[2].view(1,-1).repeat(enc_outputs.shape[0],1))
            w4=self.w4(decoder_state1[3].view(1,-1).repeat(enc_outputs.shape[0],1))
            w=self.w(w1+w2+w3+w4)
            v=self.atten(torch.tanh(u+w))
            alpha=F.softmax(v.view(1,-1),dim=1)
            cj=torch.bmm(alpha.unsqueeze(0),enc_outputs.unsqueeze(0)) #(1,1,512)
            ins=self.output2catin(decoder_input) #(1,1,512)
            decoder_input=torch.cat((ins[0],cj[0]),1).unsqueeze(0)
            decoder_output,decoder_state=self.decoder(decoder_input,(decoder_state1,decoder_state2))
            out=self.hidden_output(decoder_output)
            output=self.softmax(out)
            outputs.append(output.view(1,-1))
            max_index=torch.argmax(output,2,keepdim=True)
            if ground_truth is not None:
                max_index=ground_truth[i].reshape(1,1,1)
            one_hot=torch.zeros(output.shape).to(device)
            one_hot.scatter_(2,max_index,1)
            decoder_input=one_hot.detach()
            decoder_state1=decoder_state[0]
            decoder_state2=decoder_state[1]
        return outputs
model=model_class(len(english_set),256,len(tamil_set))
model=model.to(device)
best_models=torch.load('THEBESTMODEL_bidirectional_90.pt',map_location=device)
model.load_state_dict(best_models)
english_word=str(input('Enter the text to translitrated: '))
words_list=clean_english(english_word)
model=model.to(device)
model.eval()
predicted_words=[]
for i in range(len(words_list)):
    result_string=''
    english_rep=get_ohe_english(words_list[i],english_set)
    english_rep=english_rep.to(device)
    output=model.forward_pass(english_rep)
    for i in range(len(output)):
        index=torch.argmax(output[i]).item()
        if index==tamil_set['-PAD-']:
            break
        for key,value in tamil_set.items():
            if value==index:
                result_string+=key
    predicted_words.append(result_string)
for i in predicted_words:
    print(i)
