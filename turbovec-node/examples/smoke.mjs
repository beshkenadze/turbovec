import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { TurboQuantIndex, IdMapIndex } from "../index.js";

const DIM = 8;
const BIT_WIDTH = 4;

function approxEqual(actual, expected, epsilon = 1e-6) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= epsilon,
      `value mismatch at ${i}: ${actual[i]} vs ${expected[i]}`,
    );
  }
}

function toArray(view) {
  return Array.from(view);
}

const vectors = new Float32Array([
  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
]);

const queries = new Float32Array([
  1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
]);

const tqPath = join(tmpdir(), `turbovec-node-smoke-${process.pid}.tv`);
const idMapPath = join(tmpdir(), `turbovec-node-smoke-${process.pid}.tvim`);

try {
  const idx = new TurboQuantIndex(DIM, BIT_WIDTH);
  idx.add(vectors);
  assert.equal(idx.length, 4);
  assert.equal(idx.dim, DIM);
  assert.equal(idx.bitWidth, BIT_WIDTH);

  const initial = idx.search(queries, 2);
  assert.equal(initial.nq, 2);
  assert.equal(initial.k, 2);
  assert.equal(initial.scores.length, 4);
  assert.equal(initial.indices.length, 4);

  const mask = new Uint8Array([1, 0, 1, 0]);
  const masked = idx.search(queries.subarray(0, DIM), 2, mask);
  const maskedIndices = toArray(masked.indices);
  assert.deepEqual(maskedIndices.every((index) => index === 0n || index === 2n), true);

  idx.write(tqPath);
  const loadedIdx = TurboQuantIndex.load(tqPath);
  const reloaded = loadedIdx.search(queries, 2);
  assert.deepEqual(toArray(reloaded.indices), toArray(initial.indices));
  approxEqual(toArray(reloaded.scores), toArray(initial.scores));

  console.log("TurboQuantIndex round-trip OK", {
    length: loadedIdx.length,
    dim: loadedIdx.dim,
    bitWidth: loadedIdx.bitWidth,
    indices: toArray(reloaded.indices),
    scores: toArray(reloaded.scores),
  });

  const ids = new BigUint64Array([1001n, 1002n, 1003n, 1004n]);
  const im = new IdMapIndex(DIM, BIT_WIDTH);
  im.addWithIds(vectors, ids);
  assert.equal(im.length, 4);
  assert.equal(im.dim, DIM);
  assert.equal(im.bitWidth, BIT_WIDTH);
  assert.equal(im.contains(1002n), true);

  const idInitial = im.search(queries, 2);
  assert.equal(idInitial.nq, 2);
  assert.equal(idInitial.k, 2);
  assert.equal(idInitial.scores.length, 4);
  assert.equal(idInitial.ids.length, 4);

  const allowlisted = im.search(queries.subarray(0, DIM), 5, new BigUint64Array([1001n, 1003n]));
  const allowedIds = new Set([1001n, 1003n]);
  assert.equal(allowlisted.k, 2);
  assert.deepEqual(toArray(allowlisted.ids).every((id) => allowedIds.has(id)), true);

  im.write(idMapPath);
  const loadedIm = IdMapIndex.load(idMapPath);
  const idReloaded = loadedIm.search(queries, 2);
  assert.deepEqual(toArray(idReloaded.ids), toArray(idInitial.ids));
  approxEqual(toArray(idReloaded.scores), toArray(idInitial.scores));
  assert.equal(loadedIm.remove(1002n), true);
  assert.equal(loadedIm.contains(1002n), false);
  assert.equal(loadedIm.length, 3);

  console.log("IdMapIndex round-trip OK", {
    length: loadedIm.length,
    dim: loadedIm.dim,
    bitWidth: loadedIm.bitWidth,
    ids: toArray(idReloaded.ids),
    scores: toArray(idReloaded.scores),
  });

  console.log("OK");
} finally {
  rmSync(tqPath, { force: true });
  rmSync(idMapPath, { force: true });
}
