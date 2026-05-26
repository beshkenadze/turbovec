use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct SearchResult {
    pub scores: Float32Array,
    pub indices: BigInt64Array,
    pub nq: u32,
    pub k: u32,
}

#[napi(object)]
pub struct IdSearchResult {
    pub scores: Float32Array,
    pub ids: BigUint64Array,
    pub nq: u32,
    pub k: u32,
}

fn bigint_to_u64(v: &BigInt) -> u64 {
    v.words.first().copied().unwrap_or(0)
}

#[napi]
pub struct TurboQuantIndex {
    inner: turbovec_core::TurboQuantIndex,
}

#[napi]
impl TurboQuantIndex {
    /// Omitting `dim` constructs a lazy index that locks its dimensionality
    /// on the first `add`. `bitWidth` defaults to 4 and must be in {2,3,4}.
    #[napi(constructor)]
    pub fn new(dim: Option<u32>, bit_width: Option<u32>) -> Result<Self> {
        let bw = bit_width.unwrap_or(4) as usize;
        let inner = match dim {
            Some(d) => turbovec_core::TurboQuantIndex::new(d as usize, bw),
            None => turbovec_core::TurboQuantIndex::new_lazy(bw),
        }
        .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
        Ok(Self { inner })
    }

    /// Add a flat row-major batch of `vectors.length / dim` vectors. `dim` is
    /// required only on the first add to a lazy index; otherwise the locked
    /// dim is used and a mismatch is rejected.
    #[napi]
    pub fn add(&mut self, vectors: Float32Array, dim: Option<u32>) -> Result<()> {
        let slice: &[f32] = vectors.as_ref();
        let d = dim
            .map(|x| x as usize)
            .or_else(|| self.inner.dim_opt())
            .ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "dim is required on the first add to a lazily-constructed index".to_string(),
                )
            })?;
        self.inner
            .add_2d(slice, d)
            .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))
    }

    /// Top-`k` search over `queries.length / dim` flat queries. `mask`, when
    /// given, is a 0/1 byte array of length `this.length`; only slots set to 1
    /// are eligible and the per-query count becomes `min(k, ones)`. Returns
    /// flat `scores`/`indices` of length `nq * k`.
    #[napi]
    pub fn search(
        &self,
        queries: Float32Array,
        k: u32,
        mask: Option<Uint8Array>,
    ) -> Result<SearchResult> {
        let q: &[f32] = queries.as_ref();
        let Some(dim) = self.inner.dim_opt() else {
            return Ok(SearchResult {
                scores: Float32Array::new(Vec::new()),
                indices: BigInt64Array::new(Vec::new()),
                nq: 0,
                k: 0,
            });
        };
        if q.len() % dim != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("queries length {} is not a multiple of dim {}", q.len(), dim),
            ));
        }
        let mask_vec: Option<Vec<bool>> = match &mask {
            Some(m) => {
                let mslice: &[u8] = m.as_ref();
                if mslice.len() != self.inner.len() {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!(
                            "mask length {} does not match index size {}",
                            mslice.len(),
                            self.inner.len()
                        ),
                    ));
                }
                Some(mslice.iter().map(|&b| b != 0).collect())
            }
            None => None,
        };
        let results = self
            .inner
            .search_with_mask(q, k as usize, mask_vec.as_deref());
        Ok(SearchResult {
            scores: Float32Array::new(results.scores),
            indices: BigInt64Array::new(results.indices),
            nq: results.nq as u32,
            k: results.k as u32,
        })
    }

    #[napi]
    pub fn prepare(&self) {
        self.inner.prepare();
    }

    /// O(1) deletion that swaps the last vector into slot `idx`, so ordering
    /// is not preserved. Returns the moved vector's old index.
    #[napi]
    pub fn swap_remove(&mut self, idx: u32) -> Result<u32> {
        let i = idx as usize;
        if i >= self.inner.len() {
            return Err(Error::new(
                Status::InvalidArg,
                format!("index {} out of bounds (length = {})", i, self.inner.len()),
            ));
        }
        Ok(self.inner.swap_remove(i) as u32)
    }

    #[napi]
    pub fn write(&self, path: String) -> Result<()> {
        self.inner
            .write(&path)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let inner = turbovec_core::TurboQuantIndex::load(&path)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self { inner })
    }

    #[napi(getter)]
    pub fn dim(&self) -> Option<u32> {
        self.inner.dim_opt().map(|d| d as u32)
    }

    #[napi(getter)]
    pub fn bit_width(&self) -> u32 {
        self.inner.bit_width() as u32
    }

    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }
}

#[napi]
pub struct IdMapIndex {
    inner: turbovec_core::IdMapIndex,
}

#[napi]
impl IdMapIndex {
    /// Omitting `dim` constructs a lazy index that locks its dimensionality on
    /// the first `addWithIds`. `bitWidth` defaults to 4 and must be in {2,3,4}.
    #[napi(constructor)]
    pub fn new(dim: Option<u32>, bit_width: Option<u32>) -> Result<Self> {
        let bw = bit_width.unwrap_or(4) as usize;
        let inner = match dim {
            Some(d) => turbovec_core::IdMapIndex::new(d as usize, bw),
            None => turbovec_core::IdMapIndex::new_lazy(bw),
        }
        .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
        Ok(Self { inner })
    }

    /// Add `vectors.length / dim` vectors with matching external uint64 `ids`
    /// (length = vector count). `dim` is required only on the first add to a
    /// lazy index. Rejects duplicate or length-mismatched ids.
    #[napi]
    pub fn add_with_ids(
        &mut self,
        vectors: Float32Array,
        ids: BigUint64Array,
        dim: Option<u32>,
    ) -> Result<()> {
        let v: &[f32] = vectors.as_ref();
        let id_slice: &[u64] = ids.as_ref();
        let d = dim
            .map(|x| x as usize)
            .or_else(|| self.inner.dim_opt())
            .ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "dim is required on the first add to a lazily-constructed index".to_string(),
                )
            })?;
        self.inner
            .add_with_ids_2d(v, d, id_slice)
            .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))
    }

    #[napi]
    pub fn remove(&mut self, id: BigInt) -> bool {
        self.inner.remove(bigint_to_u64(&id))
    }

    /// Top-`k` search returning external uint64 `ids`. `allowlist`, when given,
    /// restricts results to those ids; it must be non-empty and every id must
    /// be present (else this throws). Result count is `min(k, allowed)`.
    #[napi]
    pub fn search(
        &self,
        queries: Float32Array,
        k: u32,
        allowlist: Option<BigUint64Array>,
    ) -> Result<IdSearchResult> {
        let q: &[f32] = queries.as_ref();
        let Some(dim) = self.inner.dim_opt() else {
            return Ok(IdSearchResult {
                scores: Float32Array::new(Vec::new()),
                ids: BigUint64Array::new(Vec::new()),
                nq: 0,
                k: 0,
            });
        };
        if q.len() % dim != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("queries length {} is not a multiple of dim {}", q.len(), dim),
            ));
        }
        let nq = q.len() / dim;
        let allow_slice: Option<&[u64]> = match &allowlist {
            Some(a) => {
                let s: &[u64] = a.as_ref();
                if s.is_empty() {
                    return Err(Error::new(Status::InvalidArg, "allowlist is empty".to_string()));
                }
                let mut unknown: Vec<u64> = Vec::new();
                for &id in s {
                    if !self.inner.contains(id) {
                        unknown.push(id);
                        if unknown.len() > 5 {
                            break;
                        }
                    }
                }
                if !unknown.is_empty() {
                    let preview: Vec<u64> = unknown.iter().take(5).copied().collect();
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!(
                            "allowlist contains id(s) not present in index: {:?}{}",
                            preview,
                            if unknown.len() > 5 { ", ..." } else { "" }
                        ),
                    ));
                }
                Some(s)
            }
            None => None,
        };
        let (scores, ids) = self.inner.search_with_allowlist(q, k as usize, allow_slice);
        let effective_k = if nq == 0 { k as usize } else { scores.len() / nq };
        Ok(IdSearchResult {
            scores: Float32Array::new(scores),
            ids: BigUint64Array::new(ids),
            nq: nq as u32,
            k: effective_k as u32,
        })
    }

    #[napi]
    pub fn contains(&self, id: BigInt) -> bool {
        self.inner.contains(bigint_to_u64(&id))
    }

    #[napi]
    pub fn prepare(&self) {
        self.inner.prepare();
    }

    #[napi]
    pub fn write(&self, path: String) -> Result<()> {
        self.inner
            .write(&path)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let inner = turbovec_core::IdMapIndex::load(&path)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self { inner })
    }

    #[napi(getter)]
    pub fn dim(&self) -> Option<u32> {
        self.inner.dim_opt().map(|d| d as u32)
    }

    #[napi(getter)]
    pub fn bit_width(&self) -> u32 {
        self.inner.bit_width() as u32
    }

    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }
}
