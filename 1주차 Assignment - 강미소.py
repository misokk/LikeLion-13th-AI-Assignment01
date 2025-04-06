#필요한 라이브러리 임포트
import pandas as pd
import numpy as np 
import re
from tqdm import tqdm 
import konlpy 
from sklearn.preprocessing import MinMaxScaler

#텍스트 파일 불러오기
df = pd.read_csv("C:/Users/miso3/OneDrive/바탕 화면/멋사/04.02-과제1/text_data.txt", header=None, names=['sentence'])
sent_dic = pd.read_csv("C:/Users/miso3/OneDrive/바탕 화면/멋사/04.02-과제1/SentiWord_Dict.txt", sep='\t', header=None)
scaler = MinMaxScaler(feature_range=(-2, 2))

class Aurora3:

    #분석할 문장 df 저장, okt객체 생성 , 감성사전 불러오기
    def __init__(self, df, sent_dic):
        self.df = df
        self.okt = konlpy.tag.Okt()
        self.sent_dic = sent_dic

    def get_df(self):
        print("문장을 토큰화 중입니다...")
        self.tokenizer_run()

        print("감성사전을 업데이트 중입니다...")
        self.expand_sent_dic()

        print("문장 감성분석 중입니다....")
        self.sent_analyze()
        return self.df

    #토큰화
    def tokenizer_run(self):
        tqdm.pandas()

        def text_preprocess(x): 
            text = re.sub('[^가-힣0-9a-zA-Z\\s]', '', x) 
            return ' '.join(text.split())

        def tokenize(x):
            text = []
            tokens = self.okt.pos(x)
            for token in tokens: #형태소 분석, 품사 필터링
                if token[1] in ['Adjective', 'Adverb', 'Determiner', 'Noun', 'Verb', 'Unknown']:
                    text.append(token[0])
            return text

        #전처리 결과 저장, 정제 및 토큰화된 문장을 comment 컬럼에 저장 
        self.df['comment'] = self.df['sentence'].apply(text_preprocess)
        self.df['comment'] = self.df['comment'].progress_apply(tokenize)

    #감성사전 바탕으로 문장 속 단어 감성 정보 계산
    def expand_sent_dic(self):
        sent_dic = self.sent_dic

        def make_sent_dict(x):
            pos = []
            neg = []
            tmp = {}

            for sentence in tqdm(x):
                for word in sentence:
                    target = sent_dic[sent_dic[0] == word]
                    if len(target) == 1:
                        score = float(target.iloc[0, 1])
                        if score > 0:
                            pos.append(word)
                        elif score < 0:
                            neg.append(word)
                    tmp[word] = {'W': 0, 'WP': 0, 'WN': 0}
                    #w전체 등장 횟수, wp 긍정 단어와 함께 나온 횟수, wn부정 단어와 함께 나온 횟수

            pos = list(set(pos))
            neg = list(set(neg))

            for sentence in tqdm(x):
                for word in sentence:
                    tmp[word]['W'] += 1
                    for po in pos:
                        if po in sentence:
                            tmp[word]['WP'] += 1
                            break
                    for ne in neg:
                        if ne in sentence:
                            tmp[word]['WN'] += 1
                            break
            return pos, neg, pd.DataFrame(tmp)

        def make_score_dict(d, p, n):
            N = sum(d.iloc[0, ::])
            pos_cnt = sum(d.loc[:, p].iloc[0, ::])
            neg_cnt = sum(d.loc[:, n].iloc[0, ::])

            trans = d.T
            trans['neg_cnt'] = neg_cnt
            trans['pos_cnt'] = pos_cnt
            trans['N'] = N

            trans['MI_P'] = np.log2((trans['WP'] * trans['N']) / (trans['W'] * trans['pos_cnt'] + 1e-10))
            trans['MI_N'] = np.log2((trans['WN'] * trans['N']) / (trans['W'] * trans['neg_cnt'] + 1e-10))
            trans['SO_MI'] = trans['MI_P'] - trans['MI_N']

            trans = trans.replace([np.inf, -np.inf], np.nan).dropna()
            trans = trans.sort_values(by='SO_MI', ascending=False)
            return trans

        def update_dict(d):
            add_Dic = {0: [], 1: []}
            for i in d.T.items():
                if i[0] not in list(sent_dic[0]):
                    if len(i[0]) > 1:
                        add_Dic[0].append(i[0])
                        add_Dic[1].append(i[1]['SO_MI'])

            add_Dic = pd.DataFrame(add_Dic)
            Sentiment = pd.merge(sent_dic, add_Dic, how='outer')
            return Sentiment

        self.pos, self.neg, self.new_dict = make_sent_dict(self.df['comment'].values)
        self.t_dict = make_score_dict(self.new_dict, self.pos, self.neg)
        self.t_dict['SO_MI'] = scaler.fit_transform(self.t_dict['SO_MI'].values.reshape(-1, 1))
        self.add_dict = update_dict(self.t_dict)

    #각 문장별 감성 점수와 정규화 점수를 계산
    def sent_analyze(self):
        tqdm.pandas()

        #단어의 감성 점수 총합 -> score
        def get_cnt(x):
            cnt = 0
            for word in set(x):
                target = self.add_dict[self.add_dict[0] == word]
                if len(target) == 1:
                    cnt += float(target[1])
            return cnt

        #길이 대비 감성 점수 비율 ( 문장 길이에 비례한 조정 값, 길이가 길수록 sore가 높아지는 걸 방지)
        def get_ratio(x):
            score = x['score']
            length = np.log10(len(x['comment'])) + 1
            try:
                ratio = round(score / length, 2)
            except:
                ratio = 0
            return ratio

        self.df['score'] = self.df['comment'].progress_apply(get_cnt)
        self.df['ratio'] = self.df.apply(get_ratio, axis=1)


#감성 분석 실행
test = Aurora3(df, sent_dic)
result_df = test.get_df()

#결과 출력 및 저장
print(result_df[['sentence', 'score', 'ratio']])

result_df[['sentence', 'score', 'ratio']].to_csv(
    "C:/Users/miso3/OneDrive/바탕 화면/멋사/04.02-과제1/result_output.txt", sep='\t', index=False)