"""face_db.py — Thread-safe face database"""
import pickle, os, threading, logging
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

# Max embeddings stored per person — keeps RAM bounded
# Old embeddings are dropped (FIFO) when limit is exceeded
MAX_EMBEDDINGS = 120

logger = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)


class FaceDB:
    def __init__(self, path: str = "face_data.pkl"):
        self.path  = path
        self._lock = threading.Lock()
        self._db   : Dict[str, FaceRecord] = {}
        self._load()

    def add(self, name: str, embeddings: List[np.ndarray]):
        with self._lock:
            if name not in self._db:
                self._db[name] = FaceRecord(name=name)
            self._db[name].embeddings.extend(embeddings)
            # Keep only the most recent MAX_EMBEDDINGS — prevents unbounded RAM growth
            if len(self._db[name].embeddings) > MAX_EMBEDDINGS:
                self._db[name].embeddings = self._db[name].embeddings[-MAX_EMBEDDINGS:]
            self._save()
        print(f"[DB] '{name}' — {len(self._db[name].embeddings)} embeddings total")

    def all(self) -> Dict[str, FaceRecord]:
        with self._lock: return dict(self._db)

    def names(self) -> List[str]:
        with self._lock: return list(self._db.keys())

    def delete(self, name: str) -> bool:
        with self._lock:
            if name in self._db:
                del self._db[name]; self._save(); return True
        return False

    def __len__(self): return len(self._db)

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "rb") as f: self._db = pickle.load(f)
                print(f"[DB] Loaded: {self.names()}")
            except Exception as e:
                logger.warning(f"DB load: {e}"); self._db = {}
        else:
            print("[DB] New database.")

    def _save(self):
        try:
            with open(self.path, "wb") as f: pickle.dump(self._db, f)
        except Exception as e: logger.error(f"DB save: {e}")
