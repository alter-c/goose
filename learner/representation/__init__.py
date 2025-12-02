from .base_class import CGraph, TGraph, Representation
from .slg import StripsLearningGraph
from .dlg import DeleteLearningGraph
from .flg import FdrLearningGraph
from .llg import LiftedLearningGraph
from .rlg import RichLearningGraph


REPRESENTATIONS = {
  "slg": StripsLearningGraph,
  "dlg": DeleteLearningGraph,
  "flg": FdrLearningGraph,
  "llg": LiftedLearningGraph,
  "rlg": RichLearningGraph,
}


