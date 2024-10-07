import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


class SentenceBertRetriever:
    def __init__(self, samples=None, dataset='MultiWOZ', hint_ranking=False):
        self.documents = samples
        self.hint_ranking = hint_ranking
        self.dataset = dataset
        self.vectors = []
        self.device = 'cpu'
        if torch.cuda.device_count() > 0:
            self.device = 'cuda'

        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model.eval()

        if dataset == 'MultiWOZ':
            contexts = [x['context'][-1] for x in samples]
            self.hint_min = 30
        elif dataset == 'SMD':
            contexts = [' '.join(x['context']) for x in samples]
            self.hint_min = 2
        elif dataset == 'BiTOD':
            contexts = [' '.join(x['context']) for x in samples]
            self.hint_min = 30

        vectors = np.zeros((len(contexts), 1024))
        for st in tqdm(range(0, len(contexts), 128)):
            en = min(len(contexts), st + 128)
            tout = self.tokenizer(contexts[st:en], return_tensors='pt', padding=True, truncation=True)
            tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
            with torch.no_grad():
                ret = self.model(**tout)
            embs = ret[0][:, 0]
            embs = F.normalize(embs, p=2, dim=1)
            vectors[st:en, :] = embs.to("cpu").numpy()

        self.vectors = vectors
        print(f'Total documents', len(vectors), self.vectors.shape)

    def compute_scores(self, text):
        tout = self.tokenizer([text], return_tensors='pt', padding=True, truncation=True)
        tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
        with torch.no_grad():
            ret = self.model(**tout)

        embs = ret[0][:, 0]
        embs = F.normalize(embs, p=2, dim=1)
        qvec = embs.to("cpu").numpy()
        scores = np.matmul(self.vectors, qvec.T)[:, 0]

        return scores

    def get_top_k(self, text, k=1, uuid=None, etype=None):
        scores = self.compute_scores(text)

        rets = []
        idxs = np.argsort(scores)[::-1]
        dids_sofar = set()
        for ii in idxs:
            if uuid is not None and etype is not None:
                if uuid == self.documents[ii]['uuid']:
                    continue
            if self.documents[ii]['uuid'] in dids_sofar:
                continue
            dids_sofar.add(self.documents[ii]['uuid'])
            rets.append(deepcopy(self.documents[ii]))
            if len(rets) == k:
                break

        return rets

    def search_top_k(self, text, k=1, uuid=None, etype=None, hints=None):
        if not self.hint_ranking:
            return self.get_top_k(text, k, uuid, etype)

        assert hints is not None, "Hints missing for re-ranking"
        kk = min(self.hint_min, len(self.documents))
        matches = self.get_top_k(text, kk, uuid, etype)

        hint_scores = []
        if len(hints['entity_types']) > 0:
            etypes1 = set(hints['entity_types'])
        else:
            etypes1 = {'no entity'}

        for sample in matches:
            if len(sample['hints']['entity_types']) > 0:
                etypes2 = set(sample['hints']['entity_types'])
            else:
                etypes2 = {'no entity'}

            inter = etypes1.intersection(etypes2)
            union = etypes1.union(etypes2)
            escore = len(inter) / len(union)
            cscore = int(sample['hints']['closure'] == hints['closure'])
            hint_scores.append(0.5 * escore + 0.5 * cscore)

        idxs = np.argsort(hint_scores)[::-1]
        rets = [matches[jj] for jj in idxs[:k]]

        return rets


class KBRetriever:
    def __init__(self, dataset) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        self.model.eval()
        self.model.to('cuda')
        self.dataset = dataset

    def convert_kb_entry_to_document(self, entry, dom):
        foo = lambda val: ' '.join(val.split('_'))
        document = ', '.join(f"({foo(k)}: {foo(v)})" for k, v in entry.items())
        document = f"[ {document} ]"

        return document.lower()

    def get_topk(self, sample, k=4):
        query = ' '.join(sample['context'])
        dom = sample['type']
        documents = []
        for row in sample['kb']:
            documents.append(self.convert_kb_entry_to_document(row, dom))

        queries = [query for _ in range(len(documents))]
        features = self.tokenizer(queries, documents,  padding=True, truncation=True, return_tensors="pt")
        features = {k: v.to('cuda') for k, v in features.items()}
        with torch.no_grad():
            scores = self.model(**features).logits
            scores = scores.cpu().numpy()[:, 0]

        sidxs = np.argsort(scores)[::-1][:k]
        entries = [sample['kb'][ii] for ii in sidxs]
        
        return entries


def retrieve_kb(sample, kb_retriever):
    if kb_retriever is None:
        return
    if len(sample['kb']) < 4:
        return
    kk = min(4, len(sample['kb']))
    sample['kb'] = kb_retriever.get_topk(sample, k=kk)
