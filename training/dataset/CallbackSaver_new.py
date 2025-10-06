from pytorch_lightning import Callback
import uproot
import awkward as ak
import torch
import numpy as np

class RootSaver(Callback):
    def __init__(self, pt_file, model_name, out_path, overwrite=True):
        """
        Args:
            pt_file: kept for API compatibility (unused here).
            model_name: kept for API compatibility (unused here).
            out_path: path to output ROOT file.
            overwrite: if True uses uproot.recreate (overwrite), else uproot.update (append/replace tree).
        """
        super().__init__()
        self.out_path = out_path
        self.model_name = model_name
        self.overwrite = overwrite

        self.data = None          # dict of column -> list (filled batch-by-batch)
        self.num_classes = None
        self.rank_names = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

    # ---------- PDG/type/class mapping (mirrors your RDF logic) ----------
    @staticmethod
    def _pdg_to_particle_type_vec(pdg_array: np.ndarray) -> np.ndarray:
        """
        Vectorized PDG -> string type.
        """
        out = np.full(pdg_array.shape, "others", dtype=object)

        def mask(*vals):
            m = np.zeros_like(pdg_array, dtype=bool)
            for v in vals:
                m |= (pdg_array == v)
            return m

        out[mask(12, -12)] = "ve"
        out[mask(14, -14)] = "vm"
        out[mask(16, -16)] = "vt"
        out[mask(112, -112, 114, -114, 116, -116)] = "NC"
        out[mask(130, 310)] = "kaon"
        out[mask(2112)] = "neutron"
        out[mask(13, -13)] = "muon"
        out[mask(0)] = "real_data"
        return out

    @staticmethod
    def _ptype_to_class_vec(ptype_array: np.ndarray) -> np.ndarray:
        """
        Vectorized particle type -> class id.
        """
        mapping = {
            "ve": 0, "vm": 1, "vt": 2, "NC": 3,
            "kaon": 4, "neutron": 5, "muon": 6
        }
        # default -1
        out = np.full(ptype_array.shape, -1, dtype=int)
        for k, v in mapping.items():
            out[ptype_array == k] = v
        return out

    # ---------- Schema / accumulation ----------
    def _ensure_schema(self, num_classes: int):
        if self.num_classes is not None:
            return
        self.num_classes = int(num_classes)
        self.data = {}

        # prediction columns
        for j in range(self.num_classes):
            self.data[f"Prediction_{j}"] = []

        # ids
        for col in ("PdgCode", "RunId", "EventId"):
            self.data[col] = []

        # new columns
        self.data["ParticleType"] = []            # strings
        self.data["ParticleClass"] = []           # ints
        self.data["pred_sorted_indices"] = []     # list[int] length=num_classes

        # ranked scalar columns
        for name in self.rank_names[:self.num_classes]:
            self.data[f"pred_class_{name}"] = []

    # ---------- Lightning hooks ----------
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        
        # 0) Skip empty/placeholder batches
        if batch is None or outputs is None:
            return
        if hasattr(batch, "x") and (getattr(batch, "x", None) is None or batch.x.numel() == 0):
            return

        # Tolerant to dict/tensor returns
        logits = outputs["outputs"] if isinstance(outputs, dict) and "outputs" in outputs else outputs

        # Softmax -> (N,C)
        preds = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = preds.reshape(-1, preds.shape[-1])  # ensure (N, C)
        N, C = preds.shape

        # IDs (N,3) expected in order [PdgCode, RunId, EventId]
        ids = getattr(batch, "ids", None)
        if ids is None:
            raise ValueError("Batch is missing 'ids'. Expected batch.ids with shape (N,3).")
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        else:
            ids = np.asarray(ids)
        if ids.ndim != 2 or ids.shape[1] != 3:
            raise ValueError(f"batch.ids expected shape (N,3), got {ids.shape}")

        pdg_codes = ids[:, 0].astype(int)
        run_ids   = ids[:, 1].astype(int)
        event_ids = ids[:, 2].astype(int)

        # Set schema once
        self._ensure_schema(num_classes=C)

        # ---- Vectorized accumulation ----

        # predictions per column
        # (we still loop over classes, but C is tiny (<=7), so it's negligible and avoids huge Python overhead per row)
        for j in range(C):
            # extend with the j-th column of preds
            self.data[f"Prediction_{j}"].extend(preds[:, j].astype(float).tolist())

        # ids
        self.data["PdgCode"].extend(pdg_codes.tolist())
        self.data["RunId"].extend(run_ids.tolist())
        self.data["EventId"].extend(event_ids.tolist())

        # particle type & class (vectorized)
        ptype = self._pdg_to_particle_type_vec(pdg_codes)
        # print(f"particle type: {ptype}")
        pclass = self._ptype_to_class_vec(ptype)
        self.data["ParticleType"].extend(ptype.tolist())
        self.data["ParticleClass"].extend(pclass.tolist())

        # sorted indices for each row (N,C)
        sorted_idx = np.argsort(-preds, axis=1).astype(int)
        self.data["pred_sorted_indices"].extend(sorted_idx.tolist())

        # top-k scalar columns
        for k, name in enumerate(self.rank_names[:C]):
            self.data[f"pred_class_{name}"].extend(sorted_idx[:, k].tolist())
            
    def _build_empty_awkward_array(self):
        """
        Return a 0-length Awkward Array with fields matching the sndData tree.

        Notes:
        - Do NOT use NumPy object dtype for strings: use [] (plain Python list).
        - If ParticleType/ParticleClass are actually ints in your ROOT files,
            change them to `empty_i64` below.
        """

        # Config
        nc = getattr(self, "num_classes", None)
        if isinstance(nc, (int, np.integer)) and nc > 0:
            num_classes = int(nc)
        else:
            num_classes = 7  # sensible default
        default_rank_names = [
            "first", "second", "third", "fourth", "fifth",
            "sixth", "seventh", "eighth", "ninth", "tenth",
        ]
        rank_names = list(getattr(self, "rank_names", default_rank_names))
        max_k = min(len(rank_names), num_classes)

        # Empty primitives
        empty_i64 = np.array([], dtype=np.int64)
        empty_f32 = np.array([], dtype=np.float32)

        # Build fields (strings as plain lists to avoid object dtype)
        fields = {
            "PdgCode": empty_i64,
            "RunId": empty_i64,
            "EventId": empty_i64,
            "ParticleType": [],       # string field -> plain Python list (NOT numpy object)
            "ParticleClass": [],      # string field -> plain Python list (NOT numpy object)
            "pred_sorted_indices": [],  # variable-length per entry -> empty list at top-level is fine
        }

        # Prediction_0..Prediction_{N-1}
        for j in range(num_classes):
            fields[f"Prediction_{j}"] = empty_f32

        # pred_class_first..pred_class_{k}
        for k in range(max_k):
            fields[f"pred_class_{rank_names[k]}"] = []  # string field

        # Build Awkward Array
        return ak.Array(fields)

    def on_test_epoch_end(self, trainer, pl_module):
        # If no data was collected, still save an empty tree
        if not getattr(self, "data", None) or len(self.data.get("PdgCode", [])) == 0:
            print("No predictions to save — writing empty ROOT file.")

            # Build empty array using the expected schema
            dummy_array = self._build_empty_awkward_array()

            # Write empty TTree
            if self.overwrite:
                with uproot.recreate(self.out_path) as f:
                    f["sndData"] = {key: dummy_array[key] for key in dummy_array.fields}
            else:
                with uproot.update(self.out_path) as f:
                    f["sndData"] = {key: dummy_array[key] for key in dummy_array.fields}

            print(f"Empty ROOT file written to '{self.out_path}'")
            self.data = None
            self.num_classes = None
            return


        # Build Awkward Array (list-like column for pred_sorted_indices will be preserved)
        ak_array = ak.Array(self.data)

        # Write ROOT
        if self.overwrite:
            with uproot.recreate(self.out_path) as f:
                f["sndData"] = {key: ak_array[key] for key in ak_array.fields}
        else:
            with uproot.update(self.out_path) as f:
                f["sndData"] = {key: ak_array[key] for key in ak_array.fields}

        print(f"Predictions saved to '{self.out_path}'. Rows={len(self.data['PdgCode'])}, Classes={self.num_classes}")

        # reset for next epoch (free memory)
        self.data = None
        self.num_classes = None
