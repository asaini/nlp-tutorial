import streamlit as st
from NNLM.NNLM import *
from Word2Vec.Word2VecSkipgram import word2vec_runner
import inspect


pages = st.sidebar.radio('View:',
                         (
                             'NNLM',
                             'Word2Vec',
                             'B-LSTM'),
                         index=0)

if pages == 'NNLM':
    st.markdown('# Neural Network Language Model')

    st.markdown('''
    This code is an implementation of the following [paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf):
    
    > A Neural Probabilistic Language Model by Bengio, Ducharme, Vincent, Jauvin
    
    ---
    ### Model Parameters
    ''')

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        n_step = st.number_input('Number of Steps', value=2)

    with col2:
        n_hidden = st.number_input('Number of Hidden Layers', value=2)

    with col3:
        embedding_size = st.number_input('Embedding Size', value=2)

    st.markdown('''
        ---
        **Note**: At the moment, this tutorial works for sentences with the same number of words!
    ''')
    sentences = st.text_area('Input a list of sentences separated by comma',
                             value="i like dog, i love coffee, i hate milk")

    sentences = sentences.split(',')
    st.write(sentences)
    run = st.button('Run')

    if run:
        st.markdown('''
        ---
        First, let's pre-process the data
        ''')
        source = inspect.getsource(nnlm_preprocess)
        st.code(source, language='python')

        st.write('This gives us the following sets of inputs and outputs')

        predictions, predictions_str = runner(n_step, n_hidden, sentences, embedding_size)
        st.write(predictions_str)



elif pages == 'Word2Vec':
    st.header('Word2Vec')
    batch_size = st.number_input('Batch Size', value=2)
    embedding_size = st.number_input('Embedding Size', value=2)
    sentences = st.text_area('Input a list of sentences separated by comma',
                             value='''apple banana fruit, banana orange fruit, orange banana fruit, 
                                    dog cat animal, cat monkey animal, monkey dog animal''')

    sentences = sentences.split(',')
    st.write(sentences)
    figure = word2vec_runner(sentences, batch_size,embedding_size)

    st.pyplot(figure)

# elif pages == 'Bi-LSTM':
#     pass