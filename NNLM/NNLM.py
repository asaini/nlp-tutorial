# %%
# code by Tae Hwan Jung @graykode
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import inspect


def make_batch(word_dict, sentences):
    input_batch = []
    target_batch = []

    max_length = max(len(sen.split()) for sen in sentences)

    for sen in sentences:
        word = sen.split() # space tokenizer
        input = [word_dict[n] for n in word[:-1]] # create (1~n-1) as input

        # for i in range(0, max_length - len(word)):
        #     input.append(-1)

        target = word_dict[word[-1]] # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


class NNLM(nn.Module):
    def __init__(self, n_class, n_step, n_hidden, m):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))
        self.n_class = n_class
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.m = m

    def forward(self, X):
        X = self.C(X) # X : [batch_size, n_step, n_class]
        X = X.view(-1, self.n_step * self.m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + self.H(X)) # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # [batch_size, n_class]
        return output


def nnlm_preprocess(sentences):
    # eos_sentences = [sentence + ' EOS' for sentence in sentences]
    eos_sentences = sentences
    word_list = " ".join(eos_sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # number of Vocabulary

    input_batch, target_batch = make_batch(word_dict, eos_sentences)
    return input_batch, target_batch, number_dict, n_class, word_dict


def nnlm_train(model, input_batch, target_batch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            st.write('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()


def runner(n_step, n_hidden, sentences, m=2):

    input_batch, target_batch, number_dict, n_class, word_dict = nnlm_preprocess(sentences)

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.write('Word Dictionary', word_dict)

    with col2:
        st.write('Input Batch', input_batch)

    with col3:
        st.write('Output Batch', target_batch)

    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    st.markdown('''
    ---
    Next let's create a Neural Network. See the code below
    ''')

    nn_source = inspect.getsource(NNLM)
    st.code(nn_source, language='python')

    model = NNLM(n_class, n_step, n_hidden, m)

    st.markdown('''
    ---
    Next, let's train the model
    ''')
    source = inspect.getsource(nnlm_train)
    st.code(source, language='python')

    nnlm_train(model, input_batch, target_batch)

    # Predict
    predictions = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    predictions_str = [sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predictions.squeeze()]
    return predictions, predictions_str


if __name__ == '__main__':
    n_step = 3 # number of steps, n-1 in paper
    n_hidden = 3 # number of hidden size, h in paper
    m = 3 # embedding size, m in paper

    sentences = ["i like hairy dog", "i love black coffee", "i hate cow milk", 'I am big boy']

    runner(n_step, n_hidden, sentences, m)
