# Remove blank lines

def clean(source, target):
        import os
        if os.path.exists('CLEAN') == False:
                os.mkdir('CLEAN')
        source_clean = f'CLEAN/source'
        target_clean = f'CLEAN/target'

        f1 = open(source_clean, 'w')
        f2 = open(target_clean, 'w')
        with open(source) as src, open(target) as tgt:
                for src_line, tgt_line in zip(src.readlines(), tgt.readlines()):
                        if src_line !='\n' and tgt_line != '\n':
                                f1.write(src_line)
                                f2.write(tgt_line)
        f1.close()
        f2.close()


# Tokenization with nltk
def nltk_tokenization(source, target):
        import os
        import numpy as np
        import nltk
        print('running...')
        nltk.download('punkt')
        if os.path.exists('NLTK') == False:
                os.mkdir('NLTK')        
        source_tok = f'NLTK/source'
        target_tok = f'NLTK/target'
        
        
        f1 = open(source_tok, 'w')
        f2 = open(target_tok, 'w')
        with open(source) as src:
                for source_line in src.readlines():
                        source_tokens = nltk.word_tokenize(source_line)

                        for source_token in source_tokens:
                                f1.write(f'{source_token} ')
                        f1.write('\n')
        with open(target) as tgt:
                for target_line in tgt.readlines():
                        target_tokens = nltk.word_tokenize(target_line)

                        for target_token in target_tokens:
                                f2.write(f'{target_token} ')
                        f2.write('\n')
        f1.close()
        f2.close()


# Tokenization with spacy 
def spacy_tokenization(source, target):
        # https://spacy.io/usage/spacy-101
        # https://spacy.io/usage
        import spacy
        import os 
        if os.path.exists('SPACY') == False:
                os.mkdir('SPACY')
        print('running...')
        source_tok = f'SPACY/source'
        target_tok = f'SPACY/target'
        f1 = open(source_tok, 'w')
        f2 = open(target_tok, 'w')
        nlp = spacy.load("en_core_web_sm")

        with open(source) as src, open(target) as tgt:
                for source_line, target_line in zip(src.readlines(), tgt.readlines()):          

                        source_doc = nlp(source_line.strip())
                        target_doc = nlp(target_line.strip())

                        for token in source_doc:
                                f1.write(f'{token.text} ')
                        for token in target_doc:
                                f2.write(f'{token.text} ')
                        f1.write('\n')
                        f2.write('\n')

        f1.close()
        f2.close()

# sentencepiece
def encode_sentence(source, target, vocab_size):
        import sentencepiece as spm
        import os
        if os.path.exists('SENTENCEPIECE') == False:
                os.mkdir('SENTENCEPIECE')
        print('running...')
        source_bpe = open(f'SENTENCEPIECE/source', 'w')
        target_bpe = open(f'SENTENCEPIECE/target', 'w')
        # source
        spm.SentencePieceTrainer.train(f'--input={source} --model_prefix=src --vocab_size={int(vocab_size)}')
        sp = spm.SentencePieceProcessor()
        sp.load('src.model')
        with open(source) as src:
                for line in src.readlines():
                        line_bpe = sp.encode_as_pieces(line.strip())
                        text = ' '.join(line_bpe)
                        source_bpe.write(f"{text}\n")

        # target
        spm.SentencePieceTrainer.train(f'--input={target} --model_prefix=tgt --vocab_size={int(vocab_size)}')
        sp = spm.SentencePieceProcessor()
        sp.load('tgt.model')
        with open(target) as tgt:
                for line in tgt.readlines():
                        line_bpe = sp.encode_as_pieces(line.strip())
                        text = ' '.join(line_bpe)
                        target_bpe.write(f"{text}\n")

        os.system('mv tgt.* SENTENCEPIECE')
        os.system('mv src.* SENTENCEPIECE')

# subwordnmt
def subword(source, target, op):
        import os
        print('running...')
        if os.path.exists('SUBWORDNMT') == False:
                os.mkdir('SUBWORDNMT')

        source_bpe = 'SUBWORDNMT/source'
        target_bpe = 'SUBWORDNMT/target'

        # Learn BPE
        os.system(f'subword-nmt learn-joint-bpe-and-vocab --input {source} {target} -s {op} -o bpe.codes --write-vocabulary vocab_src vocab_tgt')


        # Apply BPE in source and target
        os.system(f'subword-nmt apply-bpe -c bpe.codes --vocabulary vocab_src < {source} > {source_bpe}')
        os.system(f'subword-nmt apply-bpe -c bpe.codes --vocabulary vocab_tgt < {target} > {target_bpe}')

        os.system(f'mv vocab_src vocab_tgt bpe.codes SUBWORDNMT')

def decode_sentence(file_path, model):
        import sentencepiece as spm
        print('running...')     
        file_decode = open(f'{file_path}.decode','w')
        sp = spm.SentencePieceProcessor()
        sp.load(model)

        with open(file_path) as f:
                for line in f.readlines():
                        tokens = line.strip().split()
                        line_decode = sp.decode_pieces(tokens)
                        file_decode.write(f'{line_decode}\n')
        file_decode.close()

# Split corpus 
def split(source, target, train_size, dev_size):
        import os
        if os.path.exists('TRAIN') == False: 
                os.mkdir('TRAIN')
        file_size = 0
        with open(source) as f:
                for line in f.readlines():
                        file_size = file_size + 1       

        num_train = (file_size*train_size)/100
        num_train = int(num_train)

        num_dev = (file_size*dev_size)/100
        num_dev = int(num_dev)
        print('running...')
        train_src = open('TRAIN/train.src', 'w')
        dev_src = open('TRAIN/dev.src', 'w')
        test_src = open('TRAIN/test.src', 'w')
        train_tgt = open('TRAIN/train.tgt', 'w')
        dev_tgt = open('TRAIN/dev.tgt', 'w')
        test_tgt = open('TRAIN/test.tgt', 'w')

        t = 0
        d = 0
        with open(source) as src, open(target) as tgt:
                for src_line, tgt_line in zip(src.readlines(), tgt.readlines()):
                        if t <= num_train: # 
                                train_src.write(src_line)
                                train_tgt.write(tgt_line)
                                t = t + 1
                        elif d <= num_dev: #

                                dev_src.write(src_line)
                                dev_tgt.write(tgt_line)
                                d = d + 1
                        else:
                                test_src.write(src_line)
                                test_tgt.write(tgt_line)
        train_src.close()
        train_tgt.close()
        dev_src.close()
        dev_tgt.close()
        test_src.close()
        test_tgt.close()


# opennmt vocab

def create_vocab(source, target, size):
        import os 
        print('running...')
        if os.path.exists('TRAIN') == False:
                os.mkdir('TRAIN')
        vocab_src = 'TRAIN/vocab.src'
        vocab_tgt = 'TRAIN/vocab.tgt'
        os.system(f'onmt-build-vocab --save_vocab {vocab_src} --size {size} {source}')
        os.system(f'onmt-build-vocab --save_vocab {vocab_tgt} --size {size} {target}')


# opennmt train

def train(yaml_file,num_gpu):
        import os
        os.system(f'onmt-main --model_type Transformer --config {yaml_file} --auto_config  train --with_eval --num_gpus {num_gpu}')


# opennmt translation

def translate(yaml_file, file):
        import os
        os.system(f'onmt-main  --config {yaml_file}  --auto_config infer --features_file {file} --predictions_file file.tr')

def menu():
        choice = input('1. Remove blank lines\n2. Tokenization with nltk\n3. Tokenization with spacy\n4. Encode with sentencepiece\n5. Decode with sentencepiece\n6. Split corpus\n7. Encode with subwordnmt\n8. Create vocabulary with opennmt\n9. Train with opennmt\n10. Translate file\n')

        # Remove blank lines
        if choice == '1':
                source = input('Give path of source file\n')
                target = input('Give path of target file\n')
                clean(source, target)
                print(f'FILES are created AND saved in CLEAN!\n')

        elif choice == '2':
                source = input('Give path of source file\n')
                target = input('Give path of target file\n')
                nltk_tokenization(source, target)
                print(f'FILES are created AND saved in NLTK!')
        elif choice == '3':
                source = input('Give path of source file\n')
                target = input('Give path of target file\n')
                spacy_tokenization(source,target)
                print(f'FILES are created AND  SAVED in SPACY!')
        elif choice == '4':
                source = input('Give path of source\n')
                target = input('Give path of target\n')
                vocab_size = input('Give size of vocabulary\n')
                encode_sentence(source, target, vocab_size)
                print(f'FILES are created AND saved in SENTENCEPIECE!')

        elif choice == '5':
                file_path = input('Give the path of file\n')
                model = input('Give the model\n')
                decode_sentence(file_path, model)
                print(f'{file_path}.decode is created!\n')
        elif choice == '6':
                source = input('Give source file\n')
                target = input('Give target file\n')
                train_size = input('Give the size of train %\n')
                dev_size = input('Give the size of dev %\n')
                split(source, target, int(train_size), int(dev_size))
                print(f'source files and target files with extension src and tgt are created and saved in TRAIN directory!\n')
        elif choice == '7':
                source = input('Give source file\n')
                target = input('Give target file\n')
                op = input('Give number of operations\n')

                subword(source, target, int(op))
                print('Files are create and saved in SUBWORDNMT!\n')
        elif choice == '8':
                source = input('Give source file\n')
                target = input('Give target file\n')
                size = input('Give vocabulary size\n')
                create_vocab(source, target, int(size))
                print('Vocabulary files are created and saved in TRAIN\n')
        elif choice == '9':
                yaml_file = input('Give the path of the yaml file\n')
                num_gpu = input('Give the number of gpu\n')
                train(yaml_file, num_gpu)
                print('Results are save in TRAIN/run!\n')
        elif choice == '10':
                file = input('Give path of the file to translate\n')
                yaml_file = input('Give path of yaml file\n')

                translate(yaml_file, file)
                print('file.tr is created!\n')
        

menu()


