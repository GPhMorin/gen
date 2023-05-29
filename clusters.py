from enum import Enum

from scipy.sparse import load_npz
import n2d

matrix = load_npz('../results/kinships.npz')
x = matrix.toarray()

class MRC(Enum):
    DOMAINE_DU_ROY = 910
    MARIA_CHAPDELAINE = 920
    LAC_SAINT_JEAN_EST = 930
    SAGUENAY = 94068
    FJORD_DU_SAGUENAY = 942

class City(Enum):
    pass