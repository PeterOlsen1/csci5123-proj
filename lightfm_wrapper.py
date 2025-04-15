from lightfm import LightFM

from typing import Tuple, Union, Dict
import multiprocessing as mp

import numpy as np
from scipy import sparse as sp

### FILL IN MISSING CODE FROM https://github.com/dmitryhd/lightfm
import itertools
import time

CYTHON_DTYPE = np.float32
ID_DTYPE = np.int32

# Set of global variables for multiprocessing
_user_repr = np.array([])   # n_users, n_features
_user_repr_biases = np.array([])
_item_repr = np.ndarray([])  # n_features, n_items
_item_repr_biases = np.array([])
_pool = None
_item_chunks = {}


def _check_setup():
    if not (len(_user_repr)
        and len(_user_repr_biases)
        and len(_item_repr)
        and len(_item_repr_biases)):

        raise EnvironmentError('You must setup mode.batch_setup(item_ids) before using predict')


def _batch_setup(model: LightFM,
                 item_chunks: Dict[int, np.ndarray],
                 item_features: Union[None, sp.csr_matrix]=None,
                 user_features: Union[None, sp.csr_matrix]=None,
                 n_process: int=1):

    global _item_repr, _user_repr
    global _item_repr_biases, _user_repr_biases
    global _pool
    global _item_chunks

    if item_features is None:
        n_items = len(model.item_biases)
        item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format='csr')

    if user_features is None:
        n_users = len(model.user_biases)
        user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format='csr')

    n_users = user_features.shape[0]
    user_features = model._construct_user_features(n_users, user_features)
    _user_repr, _user_repr_biases = _precompute_representation(
        features=user_features,
        feature_embeddings=model.user_embeddings,
        feature_biases=model.user_biases,
    )

    n_items = item_features.shape[0]
    item_features = model._construct_item_features(n_items, item_features)
    _item_repr, _item_repr_biases = _precompute_representation(
        features=item_features,
        feature_embeddings=model.item_embeddings,
        feature_biases=model.item_biases,
    )
    _item_repr = _item_repr.T
    _item_chunks = item_chunks
    _clean_pool()
    # Pool creation should go last
    if n_process > 1:
        _pool = mp.Pool(processes=n_process)


def _precompute_representation(
        features: sp.csr_matrix,
        feature_embeddings: np.ndarray,
        feature_biases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param: features           csr_matrix         [n_objects, n_features]
    :param: feature_embeddings np.ndarray(float)  [n_features, no_component]
    :param: feature_biases     np.ndarray(float)  [n_features]

    :return:
    TODO:
    tuple of
    - representation    np.ndarray(float)  [n_objects, no_component+1]
    - bias repr
    """

    representation = features.dot(feature_embeddings)
    representation_bias = features.dot(feature_biases)
    return representation, representation_bias


def _get_top_k_scores(scores: np.ndarray, k: int, item_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: indices of items, top_k scores. All in score decreasing order.
    """

    if k:
        top_indices = np.argpartition(scores, -k)[-k:]
        scores = scores[top_indices]
        sorted_top_indices = np.argsort(-scores)
        scores = scores[sorted_top_indices]
        top_indices = top_indices[sorted_top_indices]
    else:
        top_indices = np.arange(len(scores))

    if len(item_ids):
        top_indices = item_ids[top_indices]

    return top_indices, scores


def _batch_predict_for_user(user_id: int, top_k: int=50, chunk_id: int=None, item_ids=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: indices of items, top_k scores. All in score decreasing order.
    """
    # exclude biases from repr (last column of user_repr and last row of transposed item repr)
    user_repr = _user_repr[user_id, :]

    if chunk_id is not None:
        item_ids = _item_chunks[chunk_id]
    elif item_ids is None:
        raise UserWarning('Supply item chunks at setup or item_ids in predict')

    if item_ids is None or len(item_ids) == 0:
        item_repr = _item_repr
        item_repr_biases = _item_repr_biases
    else:
        item_repr = _item_repr[:, item_ids]
        item_repr_biases = _item_repr_biases[item_ids]

    scores = user_repr.dot(item_repr)
    scores += _user_repr_biases[user_id]
    scores += item_repr_biases
    return _get_top_k_scores(scores, k=top_k, item_ids=item_ids)


def _clean_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def _batch_cleanup():
    global _item_ids, _item_repr, _user_repr, _pool, _item_chunks
    _item_chunks = {}
    _user_repr = np.array([])
    _item_repr = np.ndarray([])
    _clean_pool()

def _precompute_representation(features, feature_embeddings, feature_biases):
    representation = features.dot(feature_embeddings)
    representation_bias = features.dot(feature_biases)
    return representation, representation_bias


class LightFMWrapper(LightFM):
    def __init__(self, no_components, loss, learning_rate, max_sampled, random_state, user_alpha):
        super().__init__(no_components=no_components, loss=loss, learning_rate=learning_rate, max_sampled=max_sampled, random_state=random_state, user_alpha=user_alpha)

    @staticmethod
    def _to_cython_dtype(mat):
        if mat.dtype != CYTHON_DTYPE:
            return mat.astype(CYTHON_DTYPE)
        else:
            return mat

    def _construct_item_features(self, n_items: int, item_features) -> sp.csr_matrix:
        # TODO: mb. merge with user features
        if item_features is None:
            item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format='csr')
        else:
            item_features = item_features.tocsr()

        if n_items > item_features.shape[0]:
            raise Exception('Number of item feature rows does not equal the number of items')

        if self.item_embeddings is not None:
            if not self.item_embeddings.shape[0] >= item_features.shape[1]:
                raise ValueError(
                    'The item feature matrix specifies more features than there are estimated '
                    'feature embeddings: {} vs {}.'.format(self.item_embeddings.shape[0], item_features.shape[1])
                )

        item_features = self._to_cython_dtype(item_features)
        return item_features
    
    def batch_setup(self, item_chunks, item_features, user_features, n_process: int=1):
        self.n_process = n_process
        _batch_setup(model=self, item_chunks=item_chunks, item_features=item_features, user_features=user_features, n_process=n_process)
        # global _item_repr, _user_repr
        # global _item_repr_biases, _user_repr_biases
        # global _pool
        # global _item_chunks

    
    def batch_predict(self, chunk_id, user_ids, top_k: int=50):
        # from lightfm.inference import _batch_predict_for_user, _check_setup, _pool, _item_chunks

        self._check_initialized()
        # print(_item_chunks)
        print('Batch predict: user_ids: {:,}, item_ids: {:,}'.format(len(user_ids), len(_item_chunks[chunk_id])))

        recommendations = {}
        if not isinstance(user_ids, np.ndarray):
            user_ids = np.array(user_ids, dtype=ID_DTYPE)

        # _check_setup()
        btime = time.time()

        # if self.n_process == 1:
        print('Start recommending: using single process')
        # self.debug('Start recommending: using single process')
        for user_id in user_ids:
            rec_ids, scores = _batch_predict_for_user(user_id=user_id, top_k=top_k, chunk_id=chunk_id)
            recommendations[user_id] = rec_ids, scores
        # else:
        #     # self.debug('Start recommending: using multiprocessing')
        #     print('Start recommending: using multiprocessing')
        #     recs_list = _pool.starmap(
        #         _batch_predict_for_user,
        #         zip(user_ids, itertools.repeat(top_k), itertools.repeat(chunk_id)),
        #     )
        #     recommendations = dict(zip(user_ids, recs_list))

        elapsed_sec = time.time() - btime
        elapsed_sec_by_user = elapsed_sec / len(user_ids)
        print('Recommendations for chunk {:,} done in {:.3f}s. {:.4f} s by user'.format(
            chunk_id, elapsed_sec, elapsed_sec_by_user,
        ))
        return recommendations


    def _construct_user_features(self, n_users, user_features):
        if user_features is None:
            user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format='csr')
        else:
            user_features = user_features.tocsr()

        if n_users > user_features.shape[0]:
            raise Exception('Number of user feature rows does not equal the number of users')

        # If we already have embeddings, verify that
        # we have them for all the supplied features
        if self.user_embeddings is not None:
            if not self.user_embeddings.shape[0] >= user_features.shape[1]:
                raise ValueError(
                    'The user feature matrix specifies more features than there are estimated '
                    'feature embeddings: {} vs {}.'.format(self.user_embeddings.shape[0], user_features.shape[1])
                )

        user_features = self._to_cython_dtype(user_features)
        return user_features
    
    def batch_cleanup(self):
        _batch_cleanup()