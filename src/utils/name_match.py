from src.learners.baselines.agem import AGEMLearner
from src.learners.baselines.er import ERLearner
from src.learners.baselines.ocm import OCMLearner
from src.learners.ce import CELearner
from src.learners.baselines.derpp import DERppLearner
from src.learners.baselines.er_ace import ER_ACELearner
from src.learners.baselines.dvc import DVCLearner
from src.learners.baselines.er_ema import ER_EMALearner
from src.learners.baselines.er_ema_gra_t import ER_EMA_gra_tLearner
from src.learners.baselines.gsa import GSALearner
from src.learners.baselines.pcr import PCRLearner
from src.learners.er_kdu import ER_KDULearner
from src.learners.baselines.cmsd import CMSDLearner
from src.learners.baselines.er_ema_gra import ER_EMA_GRALearner
from src.learners.dod import ER_EMA_GRA_InverseKDLearner
from src.learners.baselines.er_ema_gra_1 import ER_EMA_GRA1Learner
from src.buffers.reservoir import Reservoir
from src.buffers.protobuf import ProtoBuf
from src.buffers.SVDbuf import SVDbuf
from src.buffers.greedy import GreedySampler
from src.buffers.fifo import QueueMemory
from src.buffers.boostedbuf import BoostedBuffer
from src.buffers.mlbuf import MLBuf
from src.buffers.indexed_reservoir import IndexedReservoir
from src.buffers.logits_res import LogitsRes
from src.buffers.mgi_reservoir import MGIReservoir


learners = {
    'ER':   ERLearner,
    'CE':   CELearner,
    'AGEM': AGEMLearner,
    'OCM': OCMLearner,
    'DERpp': DERppLearner,
    'CMSD': CMSDLearner,
    'ER_EMA_GRA': ER_EMA_GRALearner,
    'ER_EMA_GRA_INV': ER_EMA_GRA_InverseKDLearner,
    'ER_EMA_GRA1': ER_EMA_GRA1Learner,
    'ER_EMA_GRA_T': ER_EMA_gra_tLearner,
    'ERACE': ER_ACELearner,
    'DVC': DVCLearner,
    "ER_EMA": ER_EMALearner,
    "GSA": GSALearner,
    "PCR": PCRLearner,
    "ER_KDU": ER_KDULearner,

}

buffers = {
    'reservoir': Reservoir,
    'protobuf': ProtoBuf,
    'svd': SVDbuf,
    'greedy': GreedySampler,
    'logits_res': LogitsRes, 
    'fifo': QueueMemory,
    'boost': BoostedBuffer,
    'mlbuf': MLBuf,
    'idx_reservoir': IndexedReservoir,
    'mgi_reservoir': MGIReservoir
}
