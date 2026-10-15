from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Event,Graph
from tgrepair.entity_features import EntityIndex,PAIR_FEATURE_NAMES


def test_pair_features_are_symmetric_and_do_not_require_nonoverlap():
    g=Graph('fixture',{'a':'Case','b':'Case','r':'Resource'},
            [Event('e1',0,'start',('a','r'),{'amount':10}),
             Event('e2',1,'middle',('b','r'),{'amount':10}),
             Event('e3',2,'end',('a','r'),{'amount':10}),
             Event('e4',3,'end',('b','r'),{'amount':10})])
    index=EntityIndex(g,{'Case'})
    assert index.pairs()==[('a','b')]
    np.testing.assert_allclose(index.features('a','b'),index.features('b','a'))
    assert len(index.features('a','b'))==len(PAIR_FEATURE_NAMES)
    assert np.isfinite(index.features('a','b')).all()


def test_cold_registry_handles_are_not_treated_as_supported_matches():
    g=Graph('fixture',{'a':'Case','cold':'Case','r':'Resource'},[Event('e',0,'start',('a','r'))])
    index=EntityIndex(g,{'Case'})
    assert index.candidates('cold')==[]
    assert index.pairs()==[]
